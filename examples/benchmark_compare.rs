//! In-process half of tools/benchmark_python_rust.py; outputs JSON outside timing.
use anyhow::{Context, ensure};
use clap::{Parser, ValueEnum};
use openbo::{
    Matrix, Result, acquisition,
    core::{self, GaussianProcess, GpConfig},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, hint::black_box, path::PathBuf, time::Instant};

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Verify,
    Time,
}

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    workload: Option<String>,
    #[arg(long, default_value_t = 5)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    #[arg(long, default_value_t = 30.0)]
    min_sample_ms: f64,
}

#[derive(Clone, Deserialize, Serialize)]
struct Specification {
    seed: u64,
    objective_points: usize,
    ei_points: usize,
    gp_small_train: usize,
    gp_small_query: usize,
    gp_medium_train: usize,
    gp_medium_query: usize,
    hv_front: usize,
    hv_dominated: usize,
}

impl Specification {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=1_000_000).contains(&self.objective_points)
                && (1..=1_000_000).contains(&self.ei_points),
            "invalid vector workload size"
        );
        ensure!(
            [self.gp_small_train, self.gp_medium_train]
                .iter()
                .all(|n| (2..=4096).contains(n)),
            "invalid GP training size"
        );
        ensure!(
            [self.gp_small_query, self.gp_medium_query]
                .iter()
                .all(|n| (1..=100_000).contains(n)),
            "invalid GP query size"
        );
        ensure!(
            (2..=4096).contains(&self.hv_front) && self.hv_dominated <= 1_000_000,
            "invalid hypervolume workload size"
        );
        Ok(())
    }
}

/// SplitMix64 top-53-bit uniforms, mirrored exactly by the Python driver.
struct Generator(u64);
impl Generator {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 11) as f64 / 9_007_199_254_740_992.0
    }
    fn matrix(&mut self, n: usize, dim: usize) -> Matrix {
        (0..n)
            .map(|_| (0..dim).map(|_| self.uniform()).collect())
            .collect()
    }
}

fn gp_config() -> GpConfig {
    GpConfig {
        lengthscale: vec![0.2, 0.35, 0.5],
        variance: 1.3,
        noise: 1e-4,
        kernel_type: "rbf".into(),
        standardize_targets: false,
        optimize_hyperparameters: false,
        mean_constant: 0.2,
        ..Default::default()
    }
}

enum Workload {
    Objective(Matrix),
    Gp {
        x: Matrix,
        y: Vec<f64>,
        query: Matrix,
        pipeline: bool,
    },
    Ei {
        mean: Vec<f64>,
        variance: Vec<f64>,
    },
    Hypervolume(Matrix),
}

impl Workload {
    fn new(id: &str, spec: &Specification) -> Result<Self> {
        Ok(match id {
            "branin_batch" => {
                Self::Objective(Generator(spec.seed ^ 1).matrix(spec.objective_points, 2))
            }
            "gp_small" | "gp_medium" | "bo_candidate_scoring" => {
                let small = id == "gp_small";
                let (n, q) = if small {
                    (spec.gp_small_train, spec.gp_small_query)
                } else {
                    (spec.gp_medium_train, spec.gp_medium_query)
                };
                let mut generator = Generator(spec.seed ^ if small { 2 } else { 3 });
                let x = generator.matrix(n, 3);
                let y = x
                    .iter()
                    .map(|row| (row[0] * 2.0 + row[1] * 0.5) - row[2] * 0.25)
                    .collect();
                let query = generator.matrix(q, 3);
                Self::Gp {
                    x,
                    y,
                    query,
                    pipeline: id == "bo_candidate_scoring",
                }
            }
            "ei_batch" => {
                let mut generator = Generator(spec.seed ^ 4);
                let mut mean = Vec::with_capacity(spec.ei_points);
                let mut variance = Vec::with_capacity(spec.ei_points);
                for _ in 0..spec.ei_points {
                    mean.push(2.0 * generator.uniform() - 1.0);
                    variance.push(0.25 + generator.uniform());
                }
                Self::Ei { mean, variance }
            }
            "hypervolume_2d" => {
                let mut points = Vec::with_capacity(spec.hv_front + spec.hv_dominated);
                for i in 0..spec.hv_front {
                    let t = (i + 1) as f64 / (spec.hv_front + 1) as f64;
                    points.push(vec![0.5 + t, 1.5 - t]);
                }
                let mut generator = Generator(spec.seed ^ 5);
                for _ in 0..spec.hv_dominated {
                    points.push(vec![
                        0.05 + 0.35 * generator.uniform(),
                        0.05 + 0.35 * generator.uniform(),
                    ]);
                }
                Self::Hypervolume(points)
            }
            _ => anyhow::bail!("unknown workload {id}"),
        })
    }

    fn values(&self) -> Vec<f64> {
        match self {
            Self::Objective(x) | Self::Hypervolume(x) => x.iter().flatten().copied().collect(),
            Self::Gp { x, y, query, .. } => x
                .iter()
                .flatten()
                .chain(y)
                .chain(query.iter().flatten())
                .copied()
                .collect(),
            Self::Ei { mean, variance } => mean.iter().chain(variance).copied().collect(),
        }
    }

    fn digest(&self) -> String {
        let mut hash = 0xcbf29ce484222325u64;
        for value in self.values() {
            for byte in value.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        format!("fnv1a64-f64le:{hash:016x}")
    }

    fn gp_output(x: &Matrix, y: &[f64], query: &Matrix) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut gp = GaussianProcess::new(gp_config())?;
        gp.fit(x, y)?;
        gp.predict(query)
    }

    fn ei(mean: &[f64], variance: &[f64], best: f64) -> Result<Vec<f64>> {
        mean.iter()
            .zip(variance)
            .map(|(&m, &v)| acquisition::expected_improvement(m, v, best))
            .collect()
    }

    fn pipeline(x: &Matrix, y: &[f64], query: &Matrix) -> Result<(usize, f64)> {
        let (mean, variance) = Self::gp_output(x, y, query)?;
        let best = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scores = Self::ei(&mean, &variance, best)?;
        let mut index = 0;
        for i in 1..scores.len() {
            if scores[i] > scores[index] {
                index = i;
            }
        }
        Ok((index, scores[index]))
    }

    fn verify(&self) -> Result<Value> {
        let value = match self {
            Self::Objective(x) => json!({"values":core::synthetic("branin",x)?}),
            Self::Gp {
                x,
                y,
                query,
                pipeline: false,
            } => {
                let (mean, variance) = Self::gp_output(x, y, query)?;
                json!({"mean":mean,"variance":variance})
            }
            Self::Gp {
                x,
                y,
                query,
                pipeline: true,
            } => {
                let (index, score) = Self::pipeline(x, y, query)?;
                json!({"index":index,"score":score,"point":query[index]})
            }
            Self::Ei { mean, variance } => json!({"values":Self::ei(mean,variance,0.2)?}),
            Self::Hypervolume(y) => {
                json!({"value":acquisition::hypervolume(y,&[0.0,0.0])?,"front_size":acquisition::pareto_front(y)?.len()})
            }
        };
        Ok(json!({"input_digest":self.digest(),"output":value}))
    }

    fn execute(&self) -> Result<()> {
        match self {
            Self::Objective(x) => {
                black_box(core::synthetic("branin", black_box(x))?);
            }
            Self::Gp {
                x,
                y,
                query,
                pipeline: false,
            } => {
                black_box(Self::gp_output(
                    black_box(x),
                    black_box(y),
                    black_box(query),
                )?);
            }
            Self::Gp {
                x,
                y,
                query,
                pipeline: true,
            } => {
                black_box(Self::pipeline(
                    black_box(x),
                    black_box(y),
                    black_box(query),
                )?);
            }
            Self::Ei { mean, variance } => {
                black_box(Self::ei(black_box(mean), black_box(variance), 0.2)?);
            }
            Self::Hypervolume(y) => {
                black_box(acquisition::hypervolume(black_box(y), &[0.0, 0.0])?);
            }
        }
        Ok(())
    }
}

fn elapsed(workload: &Workload, repetitions: usize) -> Result<u128> {
    let start = Instant::now();
    for _ in 0..repetitions {
        workload.execute()?;
    }
    Ok(start.elapsed().as_nanos())
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let spec: Specification = serde_json::from_slice(&fs::read(&args.input)?)?;
    spec.validate()?;
    let output = match args.mode {
        Mode::Verify => {
            let mut values = serde_json::Map::new();
            for id in [
                "branin_batch",
                "gp_small",
                "gp_medium",
                "ei_batch",
                "hypervolume_2d",
                "bo_candidate_scoring",
            ] {
                values.insert(id.into(), Workload::new(id, &spec)?.verify()?);
            }
            json!({"backend":"rust","profile":if cfg!(debug_assertions){"debug"}else{"release"},"upstream_revision":openbo::UPSTREAM_REVISION,"outputs":values})
        }
        Mode::Time => {
            ensure!(
                !cfg!(debug_assertions),
                "compile benchmark_compare with --release before timing"
            );
            ensure!(
                args.samples > 0 && args.min_sample_ms.is_finite() && args.min_sample_ms > 0.0,
                "positive sample count/duration required"
            );
            let workload = Workload::new(
                args.workload
                    .as_deref()
                    .context("time mode requires --workload")?,
                &spec,
            )?;
            // Match Python's calibration headroom for warm CPU/cache effects.
            let target = (args.min_sample_ms * 1_000_000.0 * 2.0) as u128;
            let mut repetitions = 1usize;
            loop {
                let duration = elapsed(&workload, repetitions)?;
                if duration >= target {
                    break;
                }
                let factor =
                    ((target as f64 / duration.max(1) as f64).ceil() as usize).clamp(2, 16);
                repetitions = repetitions
                    .checked_mul(factor)
                    .context("repetition count overflow")?;
                ensure!(
                    repetitions <= 10_000_000,
                    "workload is too short for stable calibration"
                );
            }
            // Warm after calibration, before the recorded samples.
            for _ in 0..args.warmups {
                workload.execute()?;
            }
            let samples: Vec<u128> = (0..args.samples)
                .map(|_| elapsed(&workload, repetitions))
                .collect::<Result<_>>()?;
            json!({"backend":"rust","profile":"release","workload":args.workload,"input_digest":workload.digest(),"repetitions":repetitions,"raw_elapsed_ns":samples,"warmup_calls":args.warmups,"timer":"std::time::Instant","calibration_target_ms":args.min_sample_ms})
        }
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
