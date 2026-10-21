# Complete workflow results

Each environment is measured independently. Ratios compare backends within that environment; they do not pool platforms.

## windows-local

windows-local · Windows AMD64; Intel(R) Core(TM) Ultra 7 155H · Python 3.14.6

| Workflow | Python seconds | Rust seconds | Python/Rust time | Final quality Python / Rust |
| --- | ---: | ---: | ---: | ---: |
| BO / Branin | 0.2192 | 0.004572 | 47.95× | 0.002532 / 0.004165 (regret ↓) |
| BO / Hartmann-6 | 0.6311 | 0.01076 | 58.65× | 1.965 / 1.737 (regret ↓) |
| MOBO / OKA2 | 0.2308 | 0.148 | 1.56× | 0.7101 / 0.6148 (HV fraction ↑) |
| TAF-R / Branin | 0.3004 | 0.009394 | 31.98× | 0.002532 / 0.0082 (regret ↓) |
| NAF training + BO | 0.1326 | 0.02126 | 6.24× | 0.2002 / 0.9022 (regret ↓) |

[Raw report](windows-local.json) · [Quality trajectories](windows-local-quality.svg)

## workflows-linux-x64

workflows-linux-x64 · Linux x86_64; AMD EPYC 9V45 96-Core Processor · Python 3.12.15

| Workflow | Python seconds | Rust seconds | Python/Rust time | Final quality Python / Rust |
| --- | ---: | ---: | ---: | ---: |
| BO / Branin | 0.1514 | 0.003498 | 43.28× | 0.002532 / 0.004165 (regret ↓) |
| BO / Hartmann-6 | 0.2898 | 0.008116 | 35.70× | 1.684 / 1.737 (regret ↓) |
| MOBO / OKA2 | 0.09272 | 0.09051 | 1.02× | 0.7101 / 0.6148 (HV fraction ↑) |
| TAF-R / Branin | 0.1772 | 0.005272 | 33.61× | 0.002532 / 0.0082 (regret ↓) |
| NAF training + BO | 0.04579 | 0.009654 | 4.74× | 0.2002 / 0.9022 (regret ↓) |

[Raw report](workflows-linux-x64.json) · [Quality trajectories](workflows-linux-x64-quality.svg)

## workflows-macos-arm64

workflows-macos-arm64 · macOS arm64; Apple M1 (Virtual) · Python 3.12.10

| Workflow | Python seconds | Rust seconds | Python/Rust time | Final quality Python / Rust |
| --- | ---: | ---: | ---: | ---: |
| BO / Branin | 0.2131 | 0.006018 | 35.41× | 0.002532 / 0.004165 (regret ↓) |
| BO / Hartmann-6 | 0.5928 | 0.01308 | 45.34× | 1.589 / 1.737 (regret ↓) |
| MOBO / OKA2 | 0.1702 | 0.1476 | 1.15× | 0.7101 / 0.6148 (HV fraction ↑) |
| TAF-R / Branin | 0.4627 | 0.01222 | 37.87× | 0.002532 / 0.0082 (regret ↓) |
| NAF training + BO | 0.05806 | 0.01828 | 3.18× | 0.2002 / 0.9022 (regret ↓) |

[Raw report](workflows-macos-arm64.json) · [Quality trajectories](workflows-macos-arm64-quality.svg)

## workflows-windows-x64

workflows-windows-x64 · Windows AMD64; INTEL(R) XEON(R) PLATINUM 8573C · Python 3.12.10

| Workflow | Python seconds | Rust seconds | Python/Rust time | Final quality Python / Rust |
| --- | ---: | ---: | ---: | ---: |
| BO / Branin | 0.289 | 0.007704 | 37.51× | 0.002532 / 0.004165 (regret ↓) |
| BO / Hartmann-6 | 0.5826 | 0.01473 | 39.55× | 1.965 / 1.737 (regret ↓) |
| MOBO / OKA2 | 0.1893 | 0.1653 | 1.15× | 0.7101 / 0.6148 (HV fraction ↑) |
| TAF-R / Branin | 0.4272 | 0.01278 | 33.42× | 0.002532 / 0.0082 (regret ↓) |
| NAF training + BO | 0.108 | 0.03171 | 3.41× | 0.2002 / 0.9022 (regret ↓) |

[Raw report](workflows-windows-x64.json) · [Quality trajectories](workflows-windows-x64-quality.svg)
