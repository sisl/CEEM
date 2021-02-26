# CE-EM

[![](https://img.shields.io/badge/docs-CE--EM-green)](https://sisl.github.io/CEEM/)

Official implementation of the the algorithm CE-EM and baseline Particle EM from "[Scalable Identification of Partially Observed Systems with Certainty-Equivalent EM](https://arxiv.org/abs/2006.11615)".

[Website](https://sites.google.com/stanford.edu/ceem)

## Usage

Ensure you are using at least Python 3.6

```
pip install CEEM
```

Run `python -m pytest` to ensure everything works.

A Jupyter notebook demonstrating usage can be found in the `examples` subfolder.

## Code overview

- `ceem/dynamics.py` defines the system API used by the CEEM algorithm.
- `ceem/systems/*.py` define various systems used in the experiments
- `ceem/ceem.py` contains the CEEM algorithm.
- `ceem/smoother.py` defines different smoothing routines used by the CEEM algorithm in the smoothing step.
- `ceem/learner.py` defines different learning routines used by the CEEM algorithm in the learning step.
- `ceem/opt_criteria.py` defines different optimization criteria used by the CEEM algorithm.
- `ceem/particleem.py` implements Particle EM

# Experiments

## Lorenz 

## Unbiased Estimation in Deterministic Settings

To regenerate the data in `data/lorenz/bias_experiment` run:
```
python experiments/lorenz/bias_experiment.py
```
To generate Table 1 run:
```
python experiments/lorenz/plotting/process_bias.py
```

## Comparison to Particle Based Methods

To regenerate the data in `data/lorenz/comp` run:
```
python experiments/lorenz/comp_pem.py
python experiments/lorenz/comp_ceem.py
```
To generate Figure 2 run:
```
python experiments/lorenz/plotting/process_comp.py
```

## Convergence of CE-EM on High Dimensional Problems

To regenerate data in `data/lorenz/convergence_experiment` run:
```
python experiments/lorenz/convergence_experiment_pem.py
python experiments/lorenz/convergence_experiment_ceem.py
```
To generate Figure 3 run:
```
python experiments/lorenz/plotting/process_convergence.py
```

## Helicopter

The following are scripts for training models in Section 4.2.
Pretrained models are provided in the `pretrained_models` folder.

### Data download
The dataset used in our experiments can be downloaded by running:
```
wget 'https://zenodo.org/record/3662987/files/datasets.zip?download=1' -O datasets.zip
unzip datasets.zip
```

### Baselines
#### Naive

Run the experiment with default parameters:
```
python experiments/heli/baselines.py --model naive
```
#### H25

Run the experiment with default parameters:
```
python experiments/heli/baselines.py --model H25
cp data/h25/best_net.th trained_models/h25.th
```

#### SID

Prepare the data first for residual training:
```
cp data/naive/best_net.th trained_models/naive_baseline.th
python experiments/heli/prepare_residual_dataset.py
```

Ensure you have MATLAB with the System Identification Toolbox installed then run from within MATLAB:
```
run_n4sid.m
```

#### LSTM
```
python experiments/heli/train_lstm.py
cp data/heli_lstm/ckpts/best_model.th trained_models/lstm.th
```

### NL (Ours)

Prepare the data first for residual training:
```
cp data/naive/best_net.th trained_models/naive_baseline.th
python experiments/heli/prepare_residual_dataset.py
```

Run the experiment with default parameters:
```
python experiments/heli/ceemnl.py 
```

Move the best model to trained_models
```
cp data/NLobsLdyn/ckpts/best_model.th trained_models/NL_model.th
```

### Evaluating and plotting test trajectories

First evaluate the models (uses pretrained by default) by running:

```
python experiments/heli/evaluate_models.py
```

```
python experiments/heli/plotting/plotbar.py
```

Then plot the n th trajectory in the test set by running:

```
python experiments/heli/plotting/plot_trajectories.py --trajectory 9
```

To plot the circular acceleration prediction (instead of horizontal) on the n th trajectory in the test set:

```
python experiments/heli/plotting/plot_trajectories.py --trajectory 9 --moments
```
