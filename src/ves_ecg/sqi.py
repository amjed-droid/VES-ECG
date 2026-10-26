"""
Baseline Clinical Signal Quality Indices (SQIs) for ECG.
Implements established comparator metrics from PhysioNet and biomedical literature:
- Kurtosis SQI (kSQI)
- Skewness SQI (sSQI)
- Relative QRS Power SQI (pSQI)
- Traditional Matched-Filter SNR
- Robust Hampel MAD SNR
"""

from typing import Dict, Any
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

from .core import extract_active_beat, compute_baseline_dispersion


def compute_standard_sqis(signal: np.ndarray, fs: float = 125.0) -> Dict[str, float]:
    """
    Computes standard benchmark Signal Quality Indices (SQIs) on an active ECG beat.
    
    Parameters:
        signal: 1D array representing the sampled ECG waveform.
        fs: Sampling frequency in Hertz (default 125 Hz for PhysioNet benchmark).
        
    Returns:
        Dictionary of computed SQI values.
    """
    valid_sig, _, _ = extract_active_beat(signal)
    sigma_base, tail = compute_baseline_dispersion(valid_sig)
    
    # 1. Traditional SNR
    peak_amp = float(np.max(valid_sig) - np.min(tail))
    snr_trad = float(peak_amp / sigma_base)
    
    # 2. Robust MAD SNR (Hampel scale estimator)
    mad_base = float(1.4826 * np.median(np.abs(tail - np.median(tail))))
    if mad_base < 1e-4:
        mad_base = 1e-4
    snr_mad = float(peak_amp / mad_base)
    
    # 3. Kurtosis SQI (kSQI - 4th standardized moment)
    ksqi = float(kurtosis(valid_sig, fisher=False))
    
    # 4. Skewness SQI (sSQI - 3rd standardized moment)
    ssqi = float(abs(skew(valid_sig)))
    
    # 5. QRS Power SQI (pSQI) via Welch Periodogram
    # QRS band (5-15 Hz) vs Total ECG band (1-40 Hz)
    try:
        nperseg = min(len(valid_sig), 64)
        freqs, psd = welch(valid_sig, fs=fs, nperseg=nperseg)
        qrs_mask = (freqs >= 5.0) & (freqs <= 15.0)
        tot_mask = (freqs >= 1.0) & (freqs <= 40.0)
        p_qrs = float(np.sum(psd[qrs_mask]))
        p_tot = float(np.sum(psd[tot_mask]))
        psqi = float(p_qrs / (p_tot + 1e-8))
    except Exception:
        psqi = 0.0
        
    return {
        "snr_trad": snr_trad,
        "snr_mad": snr_mad,
        "ksqi": ksqi,
        "ssqi": ssqi,
        "psqi": psqi
    }
