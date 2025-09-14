# Requirements Document

## Introduction

NERO (NeuroEvolution vs. Gradient Research Orchestrator) is a CLI-based research tool designed to enable systematic comparison of gradient-based optimizers (SGD/Adam) versus neuroevolutionary optimizers (LEEA/SHADE-ILS) on identical neural network architectures. The primary research hypothesis is that different optimizers produce distinct internal representations in neural networks. The tool must generate sufficient experimental data and analysis to support a research paper by March 2026, with a foundational commitment to methodological soundness and computational reproducibility.

## Requirements

### Requirement 1: Experiment Configuration and Execution

**User Story:** As a researcher, I want to configure and execute training experiments with different optimizers, so that I can systematically compare their performance and behavior.

#### Acceptance Criteria

1. WHEN a researcher provides a YAML configuration file THEN the system SHALL parse and validate the configuration containing model architecture, optimizer type, dataset, and training parameters.
2. WHEN executing a single experiment THEN the system SHALL run training with the specified optimizer (Adam, SGD, LEEA, or SHADE-ILS) on the specified dataset (MNIST or CIFAR-10).
3. WHEN executing batch runs THEN the system SHALL perform 30+ statistical replications with different random seeds.
4. WHEN an experiment completes THEN the system SHALL create timestamped output directories containing `config.yaml`, all collected metrics, and a `models/` subdirectory.
5. IF an experiment fails THEN the system SHALL log the error and continue with remaining runs in a batch.

### Requirement 2: Comprehensive Data Collection

**User Story:** As a researcher, I want detailed metrics and model states collected during training, so that I can analyze optimizer behavior and internal representations.

#### Acceptance Criteria

1. WHEN training progresses THEN the system SHALL log per-epoch loss, accuracy, and training time.
2. WHEN using an experiment tracking platform (e.g., Weights & Biases) THEN the system SHALL log all metrics, hyperparameters, and system resource usage (GPU utilization, memory) for every run.
3. WHEN training reaches specific milestones THEN the system SHALL save model checkpoints at start, mid-training, and final states, with versioning.
4. WHEN using evolutionary optimizers THEN the system SHALL collect population fitness statistics and behavioral diversity metrics for each generation.
5. WHEN using gradient-based optimizers THEN the system SHALL collect gradient norm statistics for each epoch.
6. WHEN training completes THEN the system SHALL save all collected metrics in a structured, long-term format (e.g., CSV, Parquet).

### Requirement 3: Rigorous Benchmarking and Hyperparameter Management

**User Story:** As a researcher, I want to conduct fair and robust comparisons between optimizers, so that my conclusions are scientifically valid and transparent.

#### Acceptance Criteria

1. WHEN comparing optimizers THEN the system's benchmarking protocol SHALL be based on an established framework (e.g., DEEPOBS) to ensure rigor.
2. WHEN performing hyperparameter optimization THEN the system SHALL operate within a predefined computational budget (e.g., max evaluations, wall-clock time) and use early-stopping criteria to prevent resource waste.
3. WHEN defining a hyperparameter search THEN the system SHALL ensure the tuning budget and search space are allocated fairly and consistently across all optimizers being compared.
4. WHEN generating any comparative analysis or report THEN the system SHALL explicitly state the hyperparameter search space and the exact tuning budget allocated to each optimizer, as this is a critical confounding variable.
5. IF an optimizer comparison is performed THEN the system SHALL use a dedicated experiment tracking platform to log all results and configurations transparently.

### Requirement 4: Internal Representation Analysis

**User Story:** As a researcher, I want to explore and validate hypotheses about internal neural network representations, so that I can test the idea that different optimizers produce functionally distinct solutions.

#### Acceptance Criteria

1. WHEN analyzing trained models THEN the system SHALL extract layer activations for specified layers.
2. WHEN comparing representations THEN the system SHALL implement Linear Classifier Probes as the primary method for assessing the functional similarity of representations.
3. WHEN using Centered Kernel Alignment (CKA) THEN it SHALL be treated as an exploratory tool, and any conclusions suggested by CKA must be cross-validated with linear probes.
4. WHEN computing CKA THEN the system SHALL perform and report sensitivity analyses, including the impact of outliers and class-subset translations, to assess the stability of the score.
5. WHEN generating representation reports THEN the system SHALL include t-SNE/UMAP visualizations, weight histograms, and, most importantly, the results of the linear probe cross-validation.

### Requirement 5: Error Pattern and Disagreement Analysis

**User Story:** As a researcher, I want to understand where different optimizers disagree in their predictions, so that I can identify systematic differences in learned representations.

#### Acceptance Criteria

1. WHEN comparing model predictions THEN the system SHALL identify samples where optimizers disagree.
2. WHEN disagreements are found THEN the system SHALL generate confusion matrices for each optimizer.
3. WHEN analyzing error patterns THEN the system SHALL compute disagreement rates and sample-level analysis.
4. WHEN generating disagreement reports THEN the system SHALL highlight top disagreement examples with visualizations.
5. IF disagreement rates are low THEN the system SHALL report this finding and suggest potential causes.

### Requirement 6: Interpretability Integration

**User Story:** As a researcher, I want interpretability analysis applied to trained models, so that I can understand what different optimizers learn to focus on.

#### Acceptance Criteria

1. WHEN models are trained THEN the system SHALL apply Layer-wise Relevance Propagation (LRP) to saved model checkpoints.
2. WHEN LRP is applied THEN it SHALL use a composite strategy, with specific, predefined propagation rules (e.g., LRP-ϵ, αβ-LRP) for different layer types (input, convolutional, dense).
3. WHEN generating interpretability analysis THEN the system SHALL create relevance heatmaps for sample inputs.
4. WHEN comparing interpretability results THEN the system SHALL highlight differences in attention patterns between optimizers.
5. IF LRP analysis fails THEN the system SHALL fall back to gradient-based attribution methods and log the reason for the failure.

### Requirement 7: Automated Report and Artifact Generation

**User Story:** As a researcher, I want publication-ready reports, figures, and artifacts generated automatically, so that I can efficiently prepare reproducible research papers.

#### Acceptance Criteria

1. WHEN experiments complete THEN the system SHALL generate HTML reports with embedded visualizations.
2. WHEN creating figures THEN the system SHALL export high-resolution PNG and PDF formats suitable for publication.
3. WHEN generating reports THEN the system SHALL include statistical summaries, training curves, representation analysis, and error patterns.
4. WHEN a report is finalized for publication THEN it SHALL be accompanied by a "reproducibility package" (see Requirement 11).

### Requirement 8: CLI Interface and Usability

**User Story:** As a researcher, I want a straightforward command-line interface, so that I can efficiently run experiments and generate analyses.

#### Acceptance Criteria

1. WHEN using the CLI THEN the system SHALL provide commands for `run-experiment`, `run-batch`, `analyze`, and `generate-report`.
2. WHEN providing invalid arguments THEN the system SHALL display helpful error messages and usage examples.
3. WHEN running long experiments THEN the system SHALL display progress indicators and estimated completion times.
4. WHEN experiments are interrupted THEN the system SHALL save partial results and allow resumption where feasible.

### Requirement 9: Technical Constraints and Compatibility

**User Story:** As a researcher, I want the tool to work reliably on our lab infrastructure, so that I can conduct experiments without technical barriers.

#### Acceptance Criteria

1. WHEN running on Linux systems THEN the system SHALL execute all functionality without platform-specific issues.
2. WHEN using PyTorch THEN the system SHALL leverage GPU acceleration when available and fall back to CPU gracefully.
3. WHEN working with datasets THEN the system SHALL support MNIST and CIFAR-10 with automatic downloading and preprocessing.
4. WHEN managing GPU memory THEN the system SHALL dynamically adjust batch sizes to maximize VRAM utilization without causing out-of-memory errors.

### Requirement 10: Experimental Validity and Fair Comparison

**User Story:** As a researcher, I want to ensure scientific validity of comparisons, so that my results are methodologically sound and reproducible.

#### Acceptance Criteria

1. WHEN comparing optimizers THEN the system SHALL verify that identical neural network architectures are used across all runs by generating and comparing architecture checksums.
2. WHEN initializing models THEN the system SHALL use the same random seed for initial weights across optimizer comparisons within each replication.
3. WHEN detecting architecture mismatches THEN the system SHALL halt execution and report the specific differences.

### Requirement 11: Computational Reproducibility

**User Story:** As a researcher and member of the scientific community, I want all experiments to be fully and easily reproducible, so that findings can be verified, trusted, and built upon, meeting funder mandates.

#### Acceptance Criteria

1. WHEN any experiment is executed THEN it SHALL run within a containerized environment (e.g., Docker, Apptainer) to encapsulate the code, libraries, and OS dependencies.
2. WHEN managing dependencies THEN the project SHALL use a lock file (e.g., `poetry.lock`, `Pipfile.lock`) to ensure the exact versions of all libraries are used.
3. WHEN an experiment is run THEN the system SHALL version control the datasets and model artifacts used and produced.
4. WHEN analysis is performed THEN it SHALL be executed via an automated workflow script that runs the entire pipeline from data preprocessing to figure generation, eliminating manual steps.
5. WHEN preparing for publication THEN the system SHALL generate a "reproducibility package" containing the container image, versioned data, model weights, and the automated workflow script necessary for a third party to regenerate all key results.
