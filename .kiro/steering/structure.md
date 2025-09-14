# NERO Project Structure

## Architecture Pattern
NERO follows **Domain-Driven Design (DDD)** with clean architecture principles:
- Domain layer contains core business logic and value objects
- Infrastructure concerns are separated from business logic
- Abstract interfaces define contracts between layers

## Directory Organization

### Core Package (`nero/`)
```
nero/
├── domain/          # Core business logic and value objects
├── orchestration/   # Experiment management and coordination  
├── analysis/        # Statistical and representation analysis
└── cli/             # Command-line interface
```

### Domain Layer (`nero/domain/`)
- **models.py**: Immutable value objects (ExperimentConfig, OptimizerConfig, TrainingMetrics)
- **interfaces.py**: Abstract base classes defining system contracts
- All domain models are frozen dataclasses with comprehensive validation
- Factory methods for complex object creation (e.g., `TrainingMetrics.from_log_data`)

### Orchestration Layer (`nero/orchestration/`)
- **seed_manager.py**: Deterministic seed generation for reproducible experiments
- Manages fair comparison between different optimizers
- Hash-based seed generation for consistency across runs

### Testing Structure (`tests/`)
- Mirror the main package structure
- Comprehensive unit tests for all domain models
- Integration tests for complex workflows
- Test naming convention: `test_<module_name>.py`

## Key Design Principles

### Immutability
- All domain models are frozen dataclasses
- Prevents accidental mutation of experiment configurations
- Ensures thread safety and reproducibility

### Validation
- Comprehensive validation in `__post_init__` methods
- Business rule enforcement (e.g., minimum 30 runs for statistical validity)
- Clear error messages for invalid configurations

### Deterministic Behavior
- Hash-based seed generation for reproducibility
- Fair comparison seeds ensure identical model initialization across experiments
- Separate seeds for model init, data shuffling, and optimizer randomness

### Interface Segregation
- Abstract base classes define clear contracts
- Optimizer, Dataset, ExperimentTracker, and ExperimentLogger interfaces
- Enables easy extension and testing with mocks

## File Naming Conventions
- Snake_case for all Python files
- Test files prefixed with `test_`
- Interface files contain abstract base classes
- Model files contain value objects and domain entities