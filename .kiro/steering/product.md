# NERO Product Overview

NERO (NeuroEvolution vs. Gradient Research Orchestrator) is a CLI-based research tool for systematic comparison of optimization algorithms in neural networks.

## Core Purpose
- Compare gradient-based optimizers (SGD/Adam) vs neuroevolutionary optimizers (LEEA/SHADE-ILS)
- Ensure fair comparison on identical neural network architectures
- Generate publication-ready research results with statistical rigor

## Key Features
- **Systematic Benchmarking**: Based on established frameworks like DEEPOBS
- **Statistical Validity**: Minimum 30 runs per experiment for statistical significance
- **Reproducibility**: Deterministic seed management for consistent results
- **Comprehensive Analysis**: Training metrics, representation analysis, interpretability
- **Automated Reporting**: Publication-ready figures and statistical reports

## Target Users
- ML researchers comparing optimization algorithms
- Academic teams studying neuroevolution vs gradient methods
- Scientists requiring rigorous, reproducible ML benchmarks

## Supported Configurations
- **Optimizers**: adam, sgd, leea, shade-ils
- **Datasets**: mnist, cifar-10
- **Architectures**: CNN models
- **HPO Methods**: Optuna-based hyperparameter optimization