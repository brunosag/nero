# NERO: NeuroEvolution vs. Gradient Research Orchestrator

NERO is a CLI-based research tool designed to enable systematic comparison of gradient-based optimizers (SGD/Adam) versus neuroevolutionary optimizers (LEEA/SHADE-ILS) on identical neural network architectures.

## Project Structure

```
nero/
├── domain/          # Core business logic and value objects
├── orchestration/   # Experiment management and coordination
├── analysis/        # Statistical and representation analysis
└── cli/            # Command-line interface

tests/              # Unit tests
```

## Core Features

- **Systematic Optimizer Comparison**: Fair comparison between gradient-based and neuroevolutionary optimizers
- **Rigorous Benchmarking**: Based on established frameworks like DEEPOBS
- **Comprehensive Analysis**: Statistical, representation, interpretability, and generalization analysis
- **Reproducibility**: Full computational reproducibility with containerization
- **Publication-Ready**: Automated report and figure generation

## Installation

```bash
pip install -e .
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nero
```

## Requirements

This project addresses requirements 1.1 and 10.3 from the NERO specification:
- 1.1: Experiment configuration and execution
- 10.3: Experimental validity and fair comparison

## License

MIT License