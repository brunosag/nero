# Implementation Plan

## Priority Tiers

**Priority 0 (Core Training & Comparison Pipeline):** Minimum viable system for fair optimizer comparison
- Tasks: 1, 2, 3.1, 4, 5, 6, 7, 8, 14, 17.1, 18.1

**Priority 1 (Key Research Analyses):** Essential analyses to answer core research questions
- Tasks: 9, 10, 13

**Priority 2 (Advanced Analyses & Publication Readiness):** Publication-quality features and reproducibility
- Tasks: 3.2, 11, 12, 15, 16, 17.2, 18.2

## Parallelization Opportunities

- **After Task 1:** Tasks 7 (Gradient Optimizers) and 8 (Neuroevolution) can be developed in parallel
- **After Task 8:** Analysis modules (Tasks 9, 10, 11, 12, 13) are largely independent and can be parallelized
- **After Task 14:** Reporting (Task 15) and Reproducibility (Task 16) can be developed concurrently

## Vertical Slice Integration

**Early Integration Milestone (after Task 4):** Implement minimal end-to-end workflow
- Create basic GradientOptimizer with Adam support for 5 epochs
- Implement minimal ExperimentManager for single optimizer execution
- Add simple CLI command for dummy experiment execution
- **Purpose:** Validate core architecture and discover integration issues early

- [x] 1. Set up project structure and core domain model
  - Create directory structure for nero package with domain, orchestration, analysis, and CLI modules
  - Implement core value objects (ExperimentConfig, TrainingMetrics, OptimizerConfig) with validation
  - Create abstract base classes (Optimizer, Dataset, ExperimentTracker) defining system interfaces
  - Write unit tests for value object validation and immutability
  - _Requirements: 1.1, 10.3_

- [x] 2. Implement resource management and seed control
  - [x] 2.1 Create SeedManager for deterministic experiment seeding
    - Implement deterministic seed generation based on experiment IDs and run numbers
    - Create methods for model initialization, data shuffling, and optimizer-specific seeds
    - Write tests to verify seed consistency across runs and fair comparison enforcement
    - _Requirements: 10.2, 10.3_

  - [x] 2.2 Implement VRAMManager for dynamic batch size optimization
    - Create binary search algorithm for optimal batch size detection
    - Implement memory monitoring and safety margin enforcement
    - Add graceful fallback when VRAM is insufficient with clear error messages
    - Write tests for batch size optimization under different memory constraints
    - _Requirements: 9.4, 9.5_

- [ ] 3. Create dataset abstractions and implementations
  - [ ] 3.1 Implement Dataset abstract base class and MNIST dataset
    - Create Dataset interface with train/test loaders and sample batch methods
    - Implement MNISTDataset with automatic downloading and preprocessing
    - Add data validation and integrity checking with checksums
    - Write tests for dataset loading and sample extraction
    - _Requirements: 1.1, 9.3_

  - [ ] 3.2 Implement CIFAR-10 dataset and distribution shift testing
    - Create CIFAR10Dataset following same interface pattern
    - Implement texture vs shape bias test datasets for generalization analysis
    - Add adversarially filtered datasets for shortcut learning detection
    - Write tests for distribution shift dataset generation
    - _Requirements: 1.1, 11.1, 11.2_

- [ ] 4. Implement neural network models with activation extraction
  - Create CNNModel class with configurable architecture for MNIST and CIFAR-10
  - Implement get_activations method for layer-wise activation extraction with parameterized layer names
  - Add architecture hashing for fair comparison validation
  - Create model factory for consistent model instantiation across experiments
  - Write tests for model creation, activation extraction, and architecture consistency
  - _Requirements: 1.1, 4.1, 10.1_

- [ ] 4.5 **VERTICAL SLICE INTEGRATION** - Early end-to-end validation
  - Create minimal GradientOptimizer with basic Adam implementation for 5 epochs
  - Implement basic ExperimentManager that can execute single optimizer on MNIST
  - Add simple CLI command `run-dummy-experiment` for integration testing
  - Write integration test validating complete pipeline from CLI to results
  - **Purpose:** Validate core architecture and discover integration issues early
  - _Requirements: Integration validation_

- [ ] 5. Build hyperparameter optimization system
  - [ ] 5.1 Create DEEPOBS integration for established search spaces
    - Implement DEEPOBSIntegration class with predefined search spaces for Adam/SGD on MNIST/CIFAR-10
    - Add literature-based fallback search spaces when DEEPOBS spaces unavailable
    - Create search space validation and documentation methods
    - Write tests for search space retrieval and validation
    - _Requirements: 3.1, 3.4_

  - [ ] 5.2 Implement HyperparameterOptimizer with Optuna backend
    - Create HyperparameterOptimizer class with configurable budget types (trials, time, epochs)
    - Implement objective function for optimizer evaluation with proper cross-validation
    - Add optimization history tracking and best configuration selection
    - Write tests for HPO execution and result consistency
    - _Requirements: 3.2, 3.3_

- [ ] 6. Implement experiment logging and external tracking
  - [ ] 6.1 Create ExperimentLogger with callback-based metrics collection
    - Implement ExperimentLogger class with epoch-level metric logging
    - Create TrainingMetrics conversion from logged data
    - Add local CSV/JSON storage for all collected metrics
    - Write tests for logging functionality and data integrity
    - _Requirements: 2.1, 2.6_

  - [ ] 6.2 Implement Weights & Biases integration
    - Create ExperimentTracker abstract base class
    - Implement WandBTracker with metric and hyperparameter logging
    - Add configuration for project names and experiment organization
    - Write tests for external tracker integration (with mocking)
    - _Requirements: 2.2_

- [ ] 7. Build gradient-based optimizer implementations
  - [ ] 7.1 Implement GradientOptimizer base class with PyTorch integration
    - Create GradientOptimizer class using OptimizerConfig for hyperparameters
    - Implement standard training loop with gradient norm collection
    - Add support for Adam and SGD optimizers with configurable parameters
    - Write tests for training execution and gradient statistics collection
    - _Requirements: 1.2, 2.4_

  - [ ] 7.2 Add checkpoint saving and model state management
    - Implement checkpoint saving at start, mid-training, and final epochs
    - Create versioned checkpoint storage with metadata
    - Add checkpoint loading functionality for analysis
    - Write tests for checkpoint consistency and loading
    - _Requirements: 2.2_

- [ ] 8. Implement neuroevolutionary optimizers
  - [ ] 8.1 Create NeuroevolutionOptimizer base class
    - Implement population initialization and fitness evaluation
    - Create individual evaluation methods for train/test performance
    - Add population evolution logic with mutation and selection
    - Write tests for population management and evolution
    - _Requirements: 1.2, 2.3_

  - [ ] 8.2 Implement behavioral diversity tracking
    - Create BehavioralDiversityTracker with parameterized layer analysis
    - Implement phenotypic diversity calculation using activation pattern distances
    - Add diversity metrics logging through ExperimentLogger callbacks
    - Write tests for diversity calculation accuracy and consistency
    - _Requirements: 2.3, 2.4_

- [ ] 9. Build statistical analysis engine **[PRIORITY 1 - Can parallelize with Tasks 10, 13]**
  - [ ] 9.1 Implement StatisticalAnalyzer for performance comparison
    - Create statistical comparison methods with t-tests and effect size calculations
    - Implement confidence interval computation and multiple comparison corrections
    - Add power analysis and sample size recommendation functionality
    - Write tests for statistical correctness and edge cases
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 9.2 Create training curve analysis and visualization
    - Implement training dynamics comparison with mean and standard deviation plots
    - Add statistical significance annotations on performance curves
    - Create publication-ready figure generation with high-resolution export
    - Write tests for visualization consistency and data accuracy
    - _Requirements: 7.2, 7.3_

- [ ] 10. Implement representation analysis system **[PRIORITY 1 - Can parallelize with Tasks 9, 13]**
  - [ ] 10.1 Create linear probe analyzer as primary method
    - Implement LinearProbeAnalyzer for functional representation similarity assessment
    - Create probe training and evaluation with cross-validation
    - Add probe accuracy comparison between different optimizer representations
    - Write tests for probe training consistency and accuracy measurement
    - _Requirements: 4.2, 4.5_

  - [ ] 10.2 Implement CKA analysis with sensitivity testing
    - Create CKAAnalyzer for exploratory representation similarity measurement
    - Implement sensitivity analysis including outlier impact and class-subset translations
    - Add cross-validation with linear probe results for CKA findings
    - Write tests for CKA calculation accuracy and sensitivity analysis
    - _Requirements: 4.3, 4.4_

- [ ] 11. Build interpretability analysis system **[PRIORITY 2 - Can parallelize with Tasks 12]**
  - [ ] 11.1 Implement Layer-wise Relevance Propagation (LRP)
    - Create LRPAnalyzer using Captum library for relevance map generation
    - Implement relevance computation for specified sample sets
    - Add publication-ready relevance map visualization and export
    - Write tests for LRP computation consistency and visualization quality
    - _Requirements: 6.1, 6.4_

  - [ ] 11.2 Create attention pattern comparison
    - Implement InterpretabilityAnalyzer with LRP primary and gradient fallback methods
    - Create attention pattern similarity measurement between models
    - Add divergent sample identification where optimizers focus differently
    - Write tests for attention comparison accuracy and fallback functionality
    - _Requirements: 6.2, 6.3_

- [ ] 12. Implement generalization and robustness analysis **[PRIORITY 2 - Can parallelize with Task 11]**
  - Create GeneralizationAnalyzer for shortcut learning detection
  - Implement texture vs shape bias measurement using conflicted stimuli
  - Add robustness testing on adversarially filtered datasets
  - Create generalization gap computation and reporting
  - Write tests for robustness measurement accuracy and synthetic dataset generation
  - _Requirements: 11.1, 11.2, 11.3, 11.5_

- [ ] 13. Build error pattern and disagreement analysis **[PRIORITY 1 - Can parallelize with Tasks 9, 10]**
  - Implement ErrorPatternAnalyzer for sample-level disagreement identification
  - Create confusion matrix generation and comparison between optimizers
  - Add confidence-based disagreement analysis and systematic pattern detection
  - Create disagreement visualization with top example highlighting
  - Write tests for disagreement detection accuracy and pattern analysis
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 14. Create experiment orchestration layer
  - [ ] 14.1 Implement ExperimentManager for single experiment execution
    - Create ExperimentManager coordinating optimizer, model, and dataset factories
    - Implement fair comparison validation using ExperimentValidator
    - Add resource allocation through ResourceManager integration
    - Write tests for experiment execution and validation enforcement
    - _Requirements: 1.2, 10.1, 10.4_

  - [ ] 14.2 Implement BatchManager for multi-optimizer comparison
    - Create BatchManager orchestrating multiple experiments with identical architectures
    - Implement parallel execution when resources allow with proper resource management
    - Add result aggregation and statistical comparison coordination
    - Write tests for batch execution and result consistency
    - _Requirements: 1.3, 3.5_

- [ ] 15. Build automated report generation system **[PRIORITY 2 - Can parallelize with Task 16]**
  - [ ] 15.1 Create HTML report generator with embedded visualizations
    - Implement ReportGenerator creating structured HTML reports with performance comparisons
    - Add training curve visualizations, statistical summaries, and representation analysis sections
    - Create publication-ready figure export in PNG and PDF formats
    - Write tests for report completeness and figure quality
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 15.2 Implement comprehensive analysis integration
    - Integrate all analysis components (statistical, representation, interpretability, generalization)
    - Create unified reporting pipeline from raw results to final publication-ready output
    - Add error pattern analysis and disagreement visualization to reports
    - Write tests for end-to-end report generation and content accuracy
    - _Requirements: 7.3, 7.4_

- [ ] 16. Create computational reproducibility system **[PRIORITY 2 - Can parallelize with Task 15]**
  - [ ] 16.1 Implement dependency management and containerization
    - Create DependencyManager for exact package version tracking
    - Implement ContainerManager for Docker/Apptainer definition generation
    - Add system requirement detection and container base image selection
    - Write tests for dependency locking and container definition accuracy
    - _Requirements: 11.1, 11.2_

  - [ ] 16.2 Build reproducibility package generator
    - Implement ReproducibilityPackager creating complete reproduction bundles
    - Create automated workflow script generation for one-click result reproduction
    - Add artifact manifest creation with verification checksums
    - Write tests for package completeness and reproduction script functionality
    - _Requirements: 11.4, 11.5_

- [ ] 17. Implement CLI interface and command system
  - [ ] 17.1 Create core CLI commands and argument parsing
    - Implement CLI interface with run-experiment, run-batch, analyze, and generate-report commands
    - Add comprehensive argument validation and helpful error messages
    - Create progress indicators for long-running experiments with time estimation
    - Write tests for CLI argument parsing and command execution
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 17.2 Add experiment interruption and resumption
    - Implement graceful experiment interruption with partial result saving
    - Create experiment resumption functionality from saved checkpoints
    - Add comprehensive help system with usage examples and documentation
    - Write tests for interruption handling and resumption accuracy
    - _Requirements: 8.4, 8.5_

- [ ] 18. Integration testing and validation
  - [ ] 18.1 Create end-to-end experiment validation tests
    - Implement complete experiment pipeline tests from configuration to final report
    - Create known optimizer difference validation using synthetic datasets
    - Add statistical power verification tests for different effect sizes
    - Write reproducibility tests ensuring identical results with same seeds
    - _Requirements: 10.5, 11.3_

  - [ ] 18.2 Build research validation and benchmarking
    - Create benchmark experiments comparing tool results against known literature findings
    - Implement performance benchmarking for computational efficiency
    - Add memory usage validation under different resource constraints
    - Write comprehensive integration tests covering all analysis components
    - _Requirements: 9.6, 3.5_