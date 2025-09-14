# NERO Technical Stack

## Build System & Package Management
- **Build System**: PDM (Python Dependency Management)
- **Python Version**: >=3.12
- **Build Backend**: pdm-backend

## Core Dependencies
- **PyTorch**: >=2.8.0 (neural networks, model training)
- **Pandas**: >=2.3.2 (data manipulation, metrics storage)
- **NumPy**: Random number generation, numerical operations
- **Optuna**: Hyperparameter optimization

## Development Tools
- **Linting**: Ruff with comprehensive rule set (E, W, F, I, B, C4, UP, SIM, N, PT)
- **Type Checking**: Pyright >=1.1.405
- **Testing**: pytest >=8.4.2 with coverage (pytest-cov >=7.0.0)
- **Coverage Target**: Configured with `--cov=nero` flag

## Common Commands

### Development Setup
```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run tests with coverage
pytest

# Run specific test file
pytest tests/test_domain_models.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Run linting
ruff check

# Run code formatting
ruff format

# Run type checking
pyright

# Auto-fix linting issues
ruff check --fix
```

### Package Management
```bash
# Install dependencies
pdm install

# Add new dependency
pdm add package_name

# Add development dependency
pdm add -dG dev package_name
```

## Git Commit Guidelines

### Commit Message Format
- **Use one-line commit messages** following Semantic Commit Messages format
- Format: `<type>: <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Keep descriptions concise and descriptive
- Example: `feat: implement SeedManager for deterministic experiment seeding`

## CLI Entry Point

- Main command: `nero` (configured in pyproject.toml)
- Entry point: `nero.cli.main:main`
