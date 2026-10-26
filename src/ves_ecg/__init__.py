"""
VES-ECG: Variance-Evaluated Stability for Real-Time ECG Signal Quality Assessment and Artifact Vetting.

Official implementation for the manuscript submitted to:
Biomedical Signal Processing and Control (Elsevier).
"""

from .core import (
    compute_ves_metrics,
    extract_active_beat,
    compute_baseline_dispersion,
    VESResult
)
from .sqi import compute_standard_sqis

__version__ = "2.0.0"
__author__ = "Ahmed Sattar Jabbar"
__all__ = [
    "compute_ves_metrics",
    "extract_active_beat",
    "compute_baseline_dispersion",
    "compute_standard_sqis",
    "VESResult"
]
