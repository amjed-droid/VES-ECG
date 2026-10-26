"""
Core Mathematical and Algorithmic Implementation of VES-ECG.
Variance-Evaluated Stability for Electrocardiogram Time-Series Vetting.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import median_filter


@dataclass
class VESResult:
    """Dataclass holding all computed VES diagnostic statistics for a single ECG beat."""
    valid_len: int
    sigma_base: float
    sigma_res: float
    psi: float
    penalty: float
    tier: int
    snr_trad: float
    ves_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid_len": self.valid_len,
            "sigma_base": self.sigma_base,
            "sigma_res": self.sigma_res,
            "psi": self.psi,
            "penalty": self.penalty,
            "tier": self.tier,
            "snr_trad": self.snr_trad,
            "ves_score": self.ves_score
        }


def extract_active_beat(signal: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Extracts the active physiological support of a zero-padded ECG beat vector.
    
    Parameters:
        signal: 1D array representing the sampled ECG waveform.
        
    Returns:
        valid_signal: Sliced 1D array containing active samples.
        start_idx: Integer starting index.
        end_idx: Integer ending index.
    """
    nz = np.where(signal > 0)[0]
    if len(nz) < 10:
        return signal.copy(), 0, len(signal) - 1
    return signal[nz[0]:nz[-1] + 1], int(nz[0]), int(nz[-1])


def compute_baseline_dispersion(valid_signal: np.ndarray, fraction: float = 0.20) -> Tuple[float, np.ndarray]:
    """
    Estimates the quiescent baseline noise dispersion from the isoelectric segment.
    
    Parameters:
        valid_signal: Active ECG beat samples.
        fraction: Fraction of tail samples considered isoelectric.
        
    Returns:
        sigma_base: Estimated baseline standard deviation.
        tail: Array of isoelectric samples.
    """
    tail_len = max(8, int(fraction * len(valid_signal)))
    tail = valid_signal[-tail_len:]
    sigma_base = float(np.std(tail))
    if sigma_base < 1e-4:
        sigma_base = 1e-4
    return sigma_base, tail


def compute_ves_metrics(
    signal: np.ndarray,
    filter_size: int = 11,
    tau1: float = 0.15,
    tau2: float = 0.60
) -> Dict[str, Any]:
    """
    Computes the complete Variance-Evaluated Stability (VES) diagnostics for an ECG beat.
    
    Mathematical Pipeline:
      1. Non-parametric morphology detrending: r_t = x_t - \\hat{x}_t
      2. Residual dispersion: \\sigma_{res} = std(r_t)
      3. Dispersion contrast: \\psi = |\\sigma_{res} - \\sigma_{base}| / \\sigma_{base}
      4. Logarithmic surprisal penalty: \\mathcal{P} = \\ln(1 + \\psi)
      5. F-calibrated tier classification:
         - Tier 1: \\mathcal{P} <= tau1 (Clean / High-fidelity beat)
         - Tier 2: tau1 < \\mathcal{P} <= tau2 (Borderline / Review required)
         - Tier 3: \\mathcal{P} > tau2 (Severe variance burst / Artifact Veto)
      6. Attenuated Detection Score: VES = SNR_{trad} / (1 + \\mathcal{P})
    """
    valid_sig, _, _ = extract_active_beat(signal)
    sigma_base, tail = compute_baseline_dispersion(valid_sig)
    
    # Adaptive filter kernel size bounded by beat support
    k_size = min(filter_size, len(valid_sig) // 2 * 2 + 1)
    if k_size < 3:
        k_size = 3
        
    smooth_morphology = median_filter(valid_sig, size=k_size)
    residuals = valid_sig - smooth_morphology
    sigma_res = float(np.std(residuals))
    if sigma_res < 1e-4:
        sigma_res = 1e-4
        
    # Dispersion contrast and logarithmic surprisal penalty
    psi = abs(sigma_res - sigma_base) / sigma_base
    penalty = float(np.log(1.0 + psi))
    
    # Tier classification
    if penalty <= tau1:
        tier = 1
    elif penalty <= tau2:
        tier = 2
    else:
        tier = 3
        
    # Peak amplitude and Traditional SNR
    peak_amp = float(np.max(valid_sig) - np.min(tail))
    snr_trad = float(peak_amp / sigma_base)
    
    # Composite VES Score
    ves_score = float(snr_trad / (1.0 + penalty))
    
    res = VESResult(
        valid_len=len(valid_sig),
        sigma_base=sigma_base,
        sigma_res=sigma_res,
        psi=psi,
        penalty=penalty,
        tier=tier,
        snr_trad=snr_trad,
        ves_score=ves_score
    )
    return res.to_dict()
