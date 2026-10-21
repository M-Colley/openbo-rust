//! Scalar TAF source-only acquisition and observation-count warmup regression.
use openbo::{
    Matrix, acquisition,
    core::{self, GaussianProcess, GpConfig},
    optimizers::{OptimizerConfig, SequentialOptimizer, Source, argmax, meta_features},
};

fn sources() -> Vec<Source> {
    let x = vec![vec![0.0], vec![1.0 / 3.0], vec![2.0 / 3.0], vec![1.0]];
    // The broad source has the higher mean. The narrow source has a larger
    // improvement above its own 90th percentile (0.63, versus 1.0).
    [vec![0.9, 0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0, 1.0]]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            let mut gp = GaussianProcess::new(GpConfig {
                lengthscale: vec![0.1],
                kernel_type: "rbf".into(),
                noise: 1e-10,
                standardize_targets: false,
                ..GpConfig::default()
            })
            .unwrap();
            gp.fit(&x, &values).unwrap();
            let y: Matrix = values.into_iter().map(|v| vec![v]).collect();
            Source {
                name: format!("source_{i}"),
                gps: vec![gp],
                meta_features: meta_features(&x, &y),
                x: x.clone(),
                y,
            }
        })
        .collect()
}

fn config(method: &str) -> OptimizerConfig {
    OptimizerConfig {
        method: method.into(),
        bounds: vec![(0.0, 1.0)],
        n_init: 0,
        n_iter: 6,
        n_candidates: 128,
        n_starts: 1,
        seed: 7,
        refine: false,
        rho: 10.0,
        gp: GpConfig::default(),
        ..OptimizerConfig::default()
    }
}

#[test]
fn cold_start_maximizes_weighted_source_improvements_instead_of_source_means() {
    for method in ["bo_taf", "bo_taf_m", "bo_taf_r"] {
        let mut optimizer = SequentialOptimizer::new(config(method)).unwrap();
        optimizer.sources = sources();
        let candidates = core::low_discrepancy(1, 128, 7).unwrap();
        let predictions = optimizer
            .sources
            .iter()
            .map(|source| source.gps[0].predict(&candidates).unwrap().0)
            .collect::<Vec<_>>();
        let weights = if method == "bo_taf_m" {
            acquisition::meta_weights(
                &optimizer
                    .sources
                    .iter()
                    .map(|s| s.meta_features.clone())
                    .collect(),
                &[1.0, 0.0, 0.0],
                optimizer.config.rho,
            )
            .unwrap()
        } else {
            vec![1.0, 1.0]
        };
        // Independent, direct scalar formula for the fixed source references.
        // Delta / temperature is small enough that this expression is finite.
        let expected = (0..candidates.len())
            .map(|i| {
                predictions
                    .iter()
                    .zip([0.63, 1.0])
                    .zip(&weights)
                    .map(|((mu, reference), weight)| {
                        weight * 0.05 * ((mu[i] - reference) / 0.05).exp().ln_1p()
                    })
                    .sum::<f64>()
                    / weights.iter().sum::<f64>()
            })
            .collect::<Vec<_>>();
        let means = (0..candidates.len())
            .map(|i| 0.5 * (predictions[0][i] + predictions[1][i]))
            .collect::<Vec<_>>();
        let best = argmax(&expected).unwrap();
        assert_ne!(
            best,
            argmax(&means).unwrap(),
            "fixture must distinguish the acquisitions"
        );
        assert_eq!(optimizer.suggest().unwrap(), candidates[best], "{method}");
        assert_eq!(optimizer.last_source_weights, weights, "{method}");
    }
}

fn incompatible_target_scaling(config: &mut OptimizerConfig) {
    // These are valid finite GP parameters, but cannot normalize the supplied
    // zero observations. Source-only suggestions must not need a target fit.
    config.gp.fixed_target_mean = Some(f64::MAX);
    config.gp.fixed_target_std = Some(f64::MIN_POSITIVE);
}

#[test]
fn positive_source_warmup_does_not_fit_a_target_until_three_observations() {
    for method in ["bo_taf", "bo_taf_m", "bo_taf_r"] {
        let mut settings = config(method);
        incompatible_target_scaling(&mut settings);
        let mut optimizer = SequentialOptimizer::new(settings).unwrap();
        optimizer.sources = sources();
        for _ in 0..3 {
            let point = optimizer.suggest().unwrap();
            assert!(optimizer.last_source_weights.iter().any(|w| *w > 0.0));
            optimizer.observe(&point, &[0.0]).unwrap();
        }
        let error = optimizer.suggest().unwrap_err().to_string();
        assert!(
            error.contains("target normalization overflow"),
            "{method}: {error}"
        );
        assert!(optimizer.pending().is_none());
        optimizer.config.gp = GpConfig::default();
        assert!(optimizer.suggest().unwrap().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn preloaded_target_observations_consume_source_only_warmup() {
    let mut settings = config("bo_taf_r");
    incompatible_target_scaling(&mut settings);
    let mut optimizer = SequentialOptimizer::new(settings).unwrap();
    optimizer.sources = sources();
    for point in [0.1, 0.5, 0.9] {
        optimizer.observe(&[point], &[0.0]).unwrap();
    }
    assert!(
        optimizer
            .suggest()
            .unwrap_err()
            .to_string()
            .contains("target normalization overflow")
    );
}

#[test]
fn inactive_sources_retain_the_target_fallback_during_warmup() {
    let mut settings = config("bo_taf_r");
    settings.rho = 1.0;
    incompatible_target_scaling(&mut settings);
    let mut optimizer = SequentialOptimizer::new(settings).unwrap();
    optimizer.sources = vec![sources().remove(0)];
    optimizer.observe(&[0.0], &[0.0]).unwrap();
    optimizer.observe(&[1.0], &[1.0]).unwrap();
    // The source and target orderings are opposite, giving zero source weight.
    assert!(
        optimizer
            .suggest()
            .unwrap_err()
            .to_string()
            .contains("target normalization overflow")
    );
}
