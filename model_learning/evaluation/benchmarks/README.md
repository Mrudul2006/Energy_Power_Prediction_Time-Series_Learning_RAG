# Accuracy Benchmarking

This directory contains a standalone rolling-origin benchmark for the hourly
energy-consumption series. It is intentionally separate from the exploratory
analysis scripts and model implementations.

## Run

From the repository root:

```bash
python benchmarks/accuracy_benchmark.py
```

Include the boosted-tree model when the local scikit-learn/SciPy stack is
working:

```bash
python benchmarks/accuracy_benchmark.py --include-boosted
```

Useful faster smoke run:

```bash
python benchmarks/accuracy_benchmark.py --max-folds 1
```

## Outputs

By default, results are written to `benchmark_outputs/`:

- `metrics.csv`: one row per model and fold.
- `forecasts.csv`: timestamp-level actuals and predictions.
- `summary.json`: average metrics by model.

## What It Compares

- `naive_last`: repeats the last observed value.
- `seasonal_24h`: repeats the same hour from the previous day.
- `seasonal_168h`: repeats the same hour from the previous week.
- `seasonal_blend`: averages daily and weekly seasonal forecasts.
- `boosted_trees`: optional existing repo forecaster with lag, rolling,
  calendar, and Fourier features. Enable with `--include-boosted`.
- `residual_boosted`: optional boosted-tree model that predicts the error left
  after a seasonal baseline, then adds that residual correction back to the
  baseline. Enable with `--include-boosted`.

The benchmark is designed to be extended with other model classes once the
baseline numbers are stable.
