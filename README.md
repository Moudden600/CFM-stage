# Synthetic Flow Matching Playground

This repository implements a lightweight experimental harness for the CFM internship
challenge: **build and stress-test generative models based on flow matching / stochastic
interpolants for synthetic time-series with control variables**.

## Features

- **Configurable synthetic data generator** with controllable noise level, non-linearity,
  and regime changes (`cfm.data`).
- **Conditional mean baselines** using an MLP and a Transformer encoder
  (`cfm.models`, `cfm.training`).
- **Conditional flow matching model** implemented as a velocity field MLP and Euler
  sampler (`cfm.flow_matching`).
- **Evaluation helpers** to analyse data efficiency and out-of-support performance
  (`cfm.evaluation`).
- **Interactive notebook** (`notebooks/flow_matching_experiments.ipynb`) reproducing the
  main experiments requested in the internship offer.

## Getting Started

Install the Python dependencies (PyTorch, NumPy, matplotlib, pandas):

```bash
pip install -r requirements.txt
```

Launch the notebook to explore the experiments:

```bash
jupyter notebook notebooks/flow_matching_experiments.ipynb
```

The notebook walks through:

1. Generating synthetic datasets with different noise, non-linearity, and regime-change
   intensities.
2. Training conditional mean estimators (MLP/Transformer) and measuring their data
   efficiency.
3. Training a conditional flow matching model and comparing its sampling quality to the
   baselines.
4. Stress-testing generalisation under out-of-support control perturbations.

## Repository Structure

```
cfm/                      # Python package with dataset, models, training, evaluation
notebooks/                # Interactive analysis notebook
requirements.txt          # Python dependencies
```

This codebase is intentionally compact so it can be extended easily for deeper research
or adaptation to real-world datasets.
