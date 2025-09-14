"""
NERO: NeuroEvolution vs. Gradient Research Orchestrator.

A comprehensive research framework for systematic comparison of gradient-based
optimizers (SGD, Adam) versus neuroevolutionary algorithms (LEEA, SHADE-ILS)
on identical neural network architectures. NERO provides rigorous benchmarking
capabilities with full computational reproducibility and publication-ready
analysis tools.
"""

from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"  # Fallback for local development
