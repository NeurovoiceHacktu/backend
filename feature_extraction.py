"""
Feature extraction module for tremor signal analysis.
Extract features from accelerometer data for tremor detection.

Required packages: numpy, scipy, pandas
"""

import numpy as np
from scipy import signal, stats
import pandas as pd


def extract_time_domain_features(data):
    """
    Extract time-domain features from tremor signals.
    
    Args:
        data: Input signal array (1D numpy array)
        
    Returns:
        Dictionary of time-domain features
    """
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'variance': np.var(data),
        'rms': np.sqrt(np.mean(data**2)),
        'max': np.max(data),
        'min': np.min(data),
        'range': np.ptp(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
    
    return features


def extract_frequency_domain_features(data, sampling_rate=100):
    """
    Extract frequency-domain features from tremor signals.
    
    Args:
        data: Input signal array (1D numpy array)
        sampling_rate: Sampling frequency in Hz (default: 100 Hz)
        
    Returns:
        Dictionary of frequency-domain features
    """
    # Compute FFT
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1/sampling_rate)
    
    # Get positive frequencies only
    positive_freq_idx = fft_freq > 0
    fft_vals = np.abs(fft_vals[positive_freq_idx])
    fft_freq = fft_freq[positive_freq_idx]
    
    features = {
        'dominant_frequency': fft_freq[np.argmax(fft_vals)],
        'spectral_entropy': stats.entropy(fft_vals + 1e-10),
        'spectral_energy': np.sum(fft_vals**2)
    }
    
    return features


def extract_all_features(data, sampling_rate=100):
    """
    Extract all features from signal.
    
    Args:
        data: Input signal array (1D numpy array)
        sampling_rate: Sampling frequency in Hz (default: 100 Hz)
        
    Returns:
        Dictionary of all extracted features (12 total features)
    """
    time_features = extract_time_domain_features(data)
    freq_features = extract_frequency_domain_features(data, sampling_rate)
    
    return {**time_features, **freq_features}


def preprocess_accelerometer_data(csv_path_or_df):
    """
    Load and preprocess accelerometer data from CSV file.
    
    Expected CSV format:
        time,acc_x,acc_y,acc_z
        0.0,0.12,-9.78,0.23
        0.01,0.15,-9.75,0.21
        ...
    
    Args:
        csv_path_or_df: Path to CSV file or pandas DataFrame
        
    Returns:
        tuple: (magnitude_signal, sampling_rate)
    """
    # Load data
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df
    
    # Check required columns
    required_cols = ['time', 'acc_x', 'acc_y', 'acc_z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Calculate magnitude
    magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # Calculate sampling rate
    dt = np.diff(df['time'].values)
    sampling_rate = 1.0 / np.mean(dt)
    
    return magnitude.values, sampling_rate


def extract_features_from_csv(csv_path):
    """
    Complete pipeline: Load CSV -> Preprocess -> Extract Features
    
    Args:
        csv_path: Path to accelerometer CSV file
        
    Returns:
        Dictionary of extracted features (12 features)
    """
    magnitude, sampling_rate = preprocess_accelerometer_data(csv_path)
    features = extract_all_features(magnitude, sampling_rate)
    
    return features


if __name__ == "__main__":
    # Example usage
    print("Feature Extraction Module")
    print("=" * 50)
    print("\nThis module extracts 12 features from accelerometer data:")
    print("\nTime-domain features (9):")
    print("  - mean, std, variance, rms, max, min, range")
    print("  - skewness, kurtosis")
    print("\nFrequency-domain features (3):")
    print("  - dominant_frequency, spectral_entropy, spectral_energy")
    print("\n" + "=" * 50)
