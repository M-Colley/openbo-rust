//! Real ask/tell workflows for tools/benchmark_workflows.py.
//! Inputs/fixtures and JSON serialization are outside the measured interval.
use anyhow::{Context, ensure};
use clap::Parser;
use openbo::{
    Matrix, Result,
    core::{self, GpConfig},
    neural::{NafConfig, NafOptimizer, NafTask, train_naf},
    optimizers::{OptimizerConfig, SequentialOptimizer},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    workload: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    verify: bool,
}

#[derive(Deserialize)]
struct Specification {
    budget: usize,
    n_init: usize,
    n_candidates: usize,
    optimizer_maxiter: usize,
    optimizer_restarts: usize,
    source_directory: PathBuf,
    runs: Vec<RunInput>,
    objective_checks: Vec<ObjectiveCheck>,
    training_tasks: Vec<TrainingTask>,
}

#[derive(Deserialize)]
struct RunInput {
    seed: u64,
    initial_2d: Matrix,
    initial_3d: Matrix,
    initial_6d: Matrix,
}

#[derive(Deserialize)]
struct ObjectiveCheck {
    function: String,
    x: Matrix,
}

#[derive(Deserialize)]
struct TrainingTask {
    translation: Vec<f64>,
    scaling: f64,
}

#[derive(Serialize)]
struct Observation {
    evaluation: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    cumulative_seconds: f64,
    evaluation_seconds: f64,
}

fn objective(function: &str, x: &Matrix) -> Result<Matrix> {
    if function == "oka2" {
        core::multi_objective(function, x)
    } else {
        Ok(core::synthetic(function, x)?
            .into_iter()
            .map(|v| vec![v])
            .collect())
    }
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        !cfg!(debug_assertions),
        "benchmark requires a release build"
    );
    let spec: Specification = serde_json::from_slice(&fs::read(&args.input)?)?;
    if args.verify {
        let values = spec
            .objective_checks
            .iter()
            .map(|check| {
                Ok(json!({"function":check.function,"y":objective(&check.function, &check.x)?}))
            })
            .collect::<Result<Vec<Value>>>()?;
        let checks = spec
            .objective_checks
            .iter()
            .find(|check| check.function == "normalized_branin")
            .context("missing neural objective checks")?;
        let training_checks = spec
            .training_tasks
            .iter()
            .map(|task| {
                let shifted = checks
                    .x
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(&task.translation)
                            .map(|(a, b)| a - b)
                            .collect()
                    })
                    .collect();
                Ok(core::synthetic("normalized_branin", &shifted)?
                    .into_iter()
                    .map(|v| task.scaling * v)
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        println!(
            "{}",
            json!({"objective_checks":values,"training_task_checks":training_checks,"upstream_revision":openbo::UPSTREAM_REVISION})
        );
        return Ok(());
    }
    let workload = args.workload.context("--workload required")?;
    let seed = args.seed.context("--seed required")?;
    let input = spec
        .runs
        .iter()
        .find(|run| run.seed == seed)
        .context("seed missing from input")?;
    ensure!(
        spec.budget > spec.n_init && spec.n_init > 0,
        "invalid budget/initial count"
    );
    let (function, dim) = match workload.as_str() {
        "bo_branin" | "taf_branin" | "naf_train_branin" => ("normalized_branin", 2),
        "bo_hartmann6" => ("hartmann6", 6),
        "mobo_oka2" => ("oka2", 3),
        _ => anyhow::bail!("unknown workflow {workload}"),
    };
    let gp = GpConfig {
        lengthscale: vec![1.5; dim],
        optimize_hyperparameters: true,
        optimizer_maxiter: spec.optimizer_maxiter,
        optimizer_restarts: spec.optimizer_restarts,
        optimizer_seed: Some(seed),
        ..GpConfig::default()
    };
    let mut config = OptimizerConfig {
        method: "bo_scratch_grid".into(),
        bounds: vec![(0.0, 1.0); dim],
        n_init: spec.n_init,
        n_iter: spec.budget - spec.n_init,
        seed,
        n_candidates: spec.n_candidates,
        n_starts: 1,
        refine: false,
        gp,
        ..OptimizerConfig::default()
    };
    if workload == "mobo_oka2" {
        let problem = core::mo_function_spec("oka2")?;
        config.method = "mobo_ieipv".into();
        config.reference_point = Some(problem.v_ref);
        config.upper_reference = Some(problem.w_ref);
        config.refine = true;
        config.gp.noise = 1e-4;
        config.gp.optimize_noise = true;
    } else if workload == "taf_branin" {
        config.method = "bo_taf_r".into();
        config.taf_run_dir = Some(spec.source_directory.clone());
        config.n_init = 0;
        config.n_iter = spec.budget;
    }
    let initial = match dim {
        2 => &input.initial_2d,
        3 => &input.initial_3d,
        _ => &input.initial_6d,
    };
    let mut observations = Vec::with_capacity(spec.budget);
    let mut training_seconds = 0.0;
    let mut training_details = Value::Null;
    let mut actual_naf_config = None;
    let mut source_names = Vec::new();
    ensure!(
        spec.training_tasks
            .iter()
            .all(|task| task.translation.len() == 2
                && task.translation.iter().all(|v| v.is_finite())
                && task.scaling.is_finite()
                && task.scaling > 0.0),
        "invalid neural training task"
    );
    // Preparation above does not fit models or construct any optimizer/policy.
    let started = Instant::now();
    if workload == "naf_train_branin" {
        let naf_config = NafConfig {
            dim: 2,
            budget: spec.budget,
            hidden: vec![8, 8],
            points_per_dim: 9,
            n_local_maxima: 2,
            n_local: 32,
            n_iterations: 2,
            steps_per_iteration: 2 * spec.budget,
            epochs: 2,
            minibatch_size: 8,
            seed,
            ..NafConfig::default()
        };
        let naf_gp = GpConfig {
            lengthscale: vec![0.235, 0.578],
            variance: 2.0,
            noise: 1e-10,
            kernel_type: "rbf".into(),
            standardize_targets: false,
            ..GpConfig::default()
        };
        let optimum = core::function_spec("normalized_branin")?.optimum;
        let tasks = spec
            .training_tasks
            .iter()
            .map(|task| {
                let translation = task.translation.clone();
                let scaling = task.scaling;
                NafTask::new(
                    move |x| {
                        let shifted =
                            vec![x.iter().zip(&translation).map(|(a, b)| a - b).collect()];
                        scaling
                            * core::synthetic("normalized_branin", &shifted)
                                .expect("validated finite two-dimensional Branin task")[0]
                    },
                    optimum * scaling,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let trained = train_naf(naf_config.clone(), naf_gp.clone(), &tasks, None, false)?;
        training_seconds = started.elapsed().as_secs_f64();
        training_details = json!({"iterations":2,"episodes_per_iteration":2,"episode_budget":spec.budget,
            "objective_evaluations":4*spec.budget,"mean_return":trained.mean_return,
            "mean_final_log_regret":trained.mean_final_log_regret,"policy":"final_af"});
        actual_naf_config = Some((naf_config.clone(), naf_gp.clone()));
        {
            let mut optimizer = NafOptimizer::new(naf_config, trained.af, naf_gp, None)?;
            for _ in 0..spec.budget {
                let step = Instant::now();
                let x = optimizer.suggest()?;
                let y = objective(function, &vec![x.clone()])?.remove(0);
                optimizer.observe(&x, y[0])?;
                observations.push(Observation {
                    evaluation: observations.len() + 1,
                    x,
                    y,
                    cumulative_seconds: started.elapsed().as_secs_f64(),
                    evaluation_seconds: step.elapsed().as_secs_f64(),
                });
            }
        }
    } else {
        let mut optimizer = SequentialOptimizer::new(config.clone())?;
        if workload == "taf_branin" {
            ensure!(
                optimizer.sources.len() == 3 && optimizer.warnings.is_empty(),
                "transfer fixture must load exactly three sources without warnings"
            );
            source_names = optimizer
                .sources
                .iter()
                .map(|source| source.name.clone())
                .collect();
        }
        let mut initial_points = initial.iter();
        for index in 0..spec.budget {
            let step = Instant::now();
            let x = if index < config.n_init {
                initial_points
                    .next()
                    .context("shared initial design too short")?
                    .clone()
            } else {
                optimizer.suggest()?
            };
            let y = objective(function, &vec![x.clone()])?.remove(0);
            optimizer.observe(&x, &y)?;
            observations.push(Observation {
                evaluation: index + 1,
                x,
                y,
                cumulative_seconds: started.elapsed().as_secs_f64(),
                evaluation_seconds: step.elapsed().as_secs_f64(),
            });
        }
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let configuration = match actual_naf_config {
        Some((naf_config, gp)) => {
            json!({"method":"naf_train_branin","naf_config":naf_config,"gp":gp,"training_tasks":spec.training_tasks.iter().map(|task|json!({"translation":task.translation,"scaling":task.scaling})).collect::<Vec<_>>()})
        }
        None => json!(config),
    };
    println!(
        "{}",
        json!({"backend":"rust","workload":workload,"seed":seed,"function":function,
        "total_seconds":total_seconds,"training_seconds":training_seconds,"training":training_details,
        "target_evaluations":observations.len(),"observations":observations,
        "configuration":configuration,"source_names":source_names,"numeric_backend":"nalgebra/native f64; native PPO f64"})
    );
    Ok(())
}
