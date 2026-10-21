# Validation record

The standalone package is based on OpenBO revision
`3ceda2d34225d2f1601fc57815cd821fbc78f37d`. Validation date: **2026-10-21**.
Coverage and numerical backend differences are documented in [PORTING.md](PORTING.md).

## Reproduce from this repository

Only Rust and the platform's native linker/build tools are required:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo doc --locked --no-deps
cargo build --locked --release
cargo run --locked --release -- quickstart
cargo run --locked --release --example student
cargo package --locked
```

Before committing local changes, `cargo package --locked --allow-dirty` permits
package verification. The resulting source package is
`target/package/openbo-rust-0.1.0.crate`. Cargo unpacks and builds it during
verification, exercising the packaged source independently of the repository.

To check installation without changing your usual Cargo binaries:

```sh
cargo install --locked --path . --root results/install
./results/install/bin/openbo --version
./results/install/bin/openbo quickstart --n-evals 5
```

On Windows, the executable is named `openbo.exe`. The package-release helper also
builds a platform archive with documentation, attribution and a SHA-256 checksum:

```sh
pwsh -File tools/package-release.ps1
```

## Numerical and workflow coverage

The retained native suite contains **117 tests and one compiled documentation
example**. Tests use embedded fixtures and temporary directories, without Python
or external datasets. They cover:

- Scalar objectives and task families, scalar/correlated GPs, acquisition geometry
  and sampling, optimizer trajectories and population/source interchange.
- Finite-difference neural/likelihood gradients, actual PPO learning, MC-dropout
  replay, native checkpoints and exact continuation.
- Malformed input rejection, preserved pending experiments, bounded budgets,
  overflow handling and retrying a failed population save.
- Complete CLI workflows: quickstart, scalar/MO experiments, source creation,
  split/training/transfer, NAF training/resume/import, and successive ConBO users.
- Real WebSocket sessions, concurrent isolation, Python-style configuration and
  the TAF-MOBO adapter's total-budget semantics.

The complete native code previously passed stable Rust CI on Linux, Windows and
macOS and a Rust 1.85.1 minimum-version check. This repository has its own workflow
that repeats those checks, package verification, isolated installation and a
student quickstart. Release builders additionally prepare Linux musl, Windows and
macOS Intel/Apple Silicon command-line archives.

## Optional cross-language checks

Python is used only for comparison and policy conversion. In an environment with
the reference Python OpenBO dependencies, run:

```sh
python tools/validate_python_parity.py --python-checkout /path/to/python-openbo --build
```

Omit `--python-checkout` to use an installed Python reference package. Pass
`--binary PATH` to compare another native build. The default report is
`results/validation/python_parity.json`; it records the reference module path.
Both optional tools expose help without importing third-party dependencies.

Prior cross-language checks passed **78 numerical groups / 2,763 values** on
Python 3.14 and 3.15. They include seven scalar functions, three MetaBO formulas,
three multi-objective problems, 16 scalar GP configurations and joint covariance,
EI, two-through-four-dimensional hypervolume, transfer weights and TAF-EIPV.

The archived Python Branin policy also exported/imported successfully: all
**48 logits** passed the recorded float32/float64 tolerances, with maximum absolute
difference `0.0003397758346181945` (absolute tolerance `0.0005`, relative `0.00002`).
A real native TAF-MOBO WebSocket session with the Python project's archived
ten-user population completed successfully without altering its source files.

The reference Python project's **568 tests** passed separately on Python 3.14.6
and a clean Windows Python 3.15.0 installation. Those Python sources/tests are not
part of this Cargo package. Rust execution and installation do not depend on any
Python version or Python package index.

## Limits

Tests establish the behavior they cover, rather than correctness for every
numerical input. Native hyperparameter fitting, random streams, acquisition search
and neural training do not reproduce complete Python trajectories. BoTorch
selectors use the native backends described in [PORTING.md](PORTING.md).

Large paper-scale benchmark reproduction, GPU acceleration and resuming Python
training checkpoints are outside this validation. Native resume and exported
Python policy inference are tested. Keep `Cargo.lock` when reproducing results.
