# SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery

Official code for the paper **"SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery"**
by Mansooreh Montazerin, Antonio Ortega and Ajitesh Srivastava, accepted to **NeurIPS 2026**.

SMILE (**S**ine, **M**ultiplication, **I**dentity, **L**ogarithm, **E**xponential) is a symbolic
regression framework that combines continuous gradient-based optimization with discrete symbolic
recovery, in three stages:

1. **Structural analysis** of the data: a log-linear scan decides whether a variable should be
   fixed so that the remaining expression can be learned separately.
2. **Continuous optimization** of a network built from interpretable activations
   (sin, multiplication, identity, log, exp).
3. **Symbolic recovery**: structured pruning, coefficient optimization, and rounding distill the
   network into a compact closed-form expression with exact constants.

## Repository layout

```
smile/                         SMILE package
  model.py                     network with interpretable activations
  activations.py               activation functions
  training.py                  training loop
  expression.py                network -> symbolic expression
  parametric.py                pruning, coefficient refit, rounding, simplification
  data.py, workflow.py         synthetic-data generator and a minimal end-to-end example
download_feynman.py            downloads the Feynman benchmark into data/
preprocess.py                  Feynman dataset loader
smile_pipeline.py              full pipeline on the Feynman benchmark
config.toml                    pipeline configuration
```

## Installation

Python 3.10 or newer, on **Linux or macOS** (the pipeline uses `signal.SIGALRM` for step
timeouts, which is not available on Windows).

```bash
pip install -r requirements.txt
```

A CUDA GPU is used automatically when available.

## Data

The pipeline runs on the 100 equations of the Feynman benchmark
([Udrescu & Tegmark, 2020](https://www.science.org/doi/10.1126/sciadv.aay2631)).
The original AI Feynman download page is no longer online, so `download_feynman.py` fetches
the data from [PMLB](https://github.com/EpistasisLab/pmlb), the copy used by
[SRBench](https://github.com/cavalab/srbench), and the equation metadata from the
[Internet Archive copy](https://web.archive.org/web/2025/https://space.mit.edu/home/tegmark/aifeynman.html)
of the AI Feynman page:

```bash
python download_feynman.py                    # all 100 equations (roughly 1 GB on disk)
python download_feynman.py --eqn I.12.1       # selected equations
```

This creates:

```
data/
  Feynman_with_units/          one space-separated file per equation (inputs, then target)
  FeynmanEquations.csv         equation metadata (variable names, ranges, formulas)
```

The PMLB files have 100,000 rows per equation, while the original AI Feynman files used in the
paper had 1,000,000. `config.toml` therefore sets `base_frac = 0.1`, which gives the same
10,000 training rows per equation as in the paper (`base_frac = 0.01` with the original files).
The fix-variable stage selects the rows inside a narrow window of the fixed variable, so with
PMLB data that window holds about 10 times fewer rows than with the original files. Results can
therefore differ slightly from the paper.

Paths are set in the `[feynman]` section of the config; relative paths are resolved next to the
config file.

## Running

```bash
python smile_pipeline.py                                  # uses config.toml
python smile_pipeline.py path/to/my_config.toml           # another config
```

In the config:

- `[feynman] eqn = "I.12.1"` runs one equation; `eqn = "all"` runs every equation in the data folder.
- `[data] noise_level` adds Gaussian noise `ε ~ N(0, σ² · mean(y²))` to the target (0 = no noise).
- `[pipeline] n_hidden_sequence = [1, 2]`: hidden layers tried in order for the direct fit.
- `[pipeline] enable_fix_var` turns the fix-variable stage on or off.

Each equation runs in its own subprocess. Results are written to `results/`
(one `<eqn>.csv` per equation with all trials, plus `_summary.csv`), and model checkpoints to
`checkpoints/`. Equations that already have a results file are skipped, so an
interrupted run can be resumed.

## Citation

```bibtex
@article{montazerin2026smile,
  title={SMILE: Bridging Continuous Optimization and Discrete Symbolic Recovery},
  author={Montazerin, Mansooreh and Ortega, Antonio and Srivastava, Ajitesh},
  journal={arXiv preprint arXiv:2609.04639},
  year={2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
