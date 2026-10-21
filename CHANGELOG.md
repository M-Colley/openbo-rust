# Changelog

## 0.1.1 — 2026-10-21

- Scalar TAF cold starts now use weighted source improvements; source-only warmup
  counts observed target points and avoids fitting an unused target GP.
- Hosted API documentation and a student landing page, with documentation links
  in the package metadata and a portable docs.rs build configuration.
- Matched Python/Rust benchmark tooling, retained input and timing data, and
  comparison figures with reproducibility details.
- A multi-seed workflow benchmark harness for scalar, multi-objective, transfer
  and neural optimization, retaining timing and optimization quality per evaluation.
- Four-environment workflow results: 200 validated runs with timing tables,
  quality trajectories, shared inputs and source provenance.
- Rooted Cargo package allowlists exclude ignored results and Python bytecode;
  source and native archives retain documentation assets and relative image links.
- Updated package and citation metadata for registry distribution.
- Native archive packaging checks the executable version to prevent a stale
  `-SkipBuild` binary from being labeled as a newer release.

## 0.1.0 — 2026-10-21

Initial standalone Rust package, based on OpenBO Python revision
`3ceda2d34225d2f1601fc57815cd821fbc78f37d`.

- Native scalar and multi-objective Bayesian optimization, TAF/TAF+/TAF-EIPV,
  continual optimization, correlated GPs, BNN replay and neural acquisition training.
- Library, `openbo` CLI, WebSocket server, examples, native checkpoints and portable
  source/policy interchange.
- Independent Cargo package, documentation, license/citation, reproducible lockfile,
  cross-platform CI and distribution workflows.
- Numerical reference, optimizer, neural, CLI and server regression coverage.
