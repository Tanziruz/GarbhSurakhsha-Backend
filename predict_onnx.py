"""
Make predictions using ONNX model
Optimized for deployment and cross-platform compatibility
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile
from scipy.signal import butter, lfilter, hilbert, find_peaks
from scipy.stats import skew

EPS = 1E-8

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_audio(audio_path, filter=True):
    """Read and preprocess audio file"""
    data, sr = soundfile.read(audio_path)
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Apply high-pass filter to remove DC offset
    if filter:
        data = butter_highpass_filter(data, cutoff=20, fs=sr, order=5)
    
    return data, sr


def LogMelExtractor(data, sr, mel_bins=128, log=True, snv=False):
    """Extract log mel-spectrogram features"""
    MEL_ARGS = {
        'n_mels': mel_bins,
        'n_fft': 1024,
        'hop_length': 512,
        'win_length': 1024,
        'window': 'hann',
        'center': True,
        'pad_mode': 'reflect',
        'power': 2.0,
        'fmin': 20.0,
        'fmax': sr // 2,
    }
    
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, **MEL_ARGS)
    
    if log:
        mel_spectrogram = np.log(mel_spectrogram + EPS)
    
    if snv:
        mel_spectrogram = standard_normal_variate(mel_spectrogram)
    
    return mel_spectrogram


def standard_normal_variate(data):
    """Normalize data using standard normal variate"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + EPS)


# ============================================================================
# Heart Rate Feature Extraction Functions
# ============================================================================

def extract_envelope(signal_data):
    """Extract envelope using Hilbert transform"""
    analytic_signal = hilbert(signal_data)
    return np.abs(analytic_signal)


def detect_peaks(envelope, sr):
    """Detect peaks in envelope signal"""
    min_distance = int(0.3 * sr)  # ~300 ms minimum distance between peaks
    peaks, _ = find_peaks(
        envelope,
        distance=min_distance,
        height=np.mean(envelope)
    )
    return peaks


def compute_fhr(peaks, sr):
    """Compute Fetal Heart Rate from detected peaks"""
    times = peaks / sr
    ibi = np.diff(times)

    if len(ibi) < 3:
        return None, None

    fhr = float(60.0 / float(np.mean(ibi)))
    return ibi, fhr


def extract_heart_features(ibi):
    """Extract comprehensive heart rate features from inter-beat intervals"""
    if ibi is None or len(ibi) < 3:
        return None

    fhr_series = 60.0 / ibi

    mean_ibi = float(np.mean(ibi))
    std_ibi = float(np.std(ibi))
    rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    cv = float(std_ibi / mean_ibi) if mean_ibi != 0 else float('nan')

    features = {
        # Rhythm features
        "mean_ibi": mean_ibi,
        "sdnn": std_ibi,
        "rmssd": rmssd,
        "cv": cv,
        "ibi_skewness": float(skew(ibi)),
        "abnormal_beats": int(np.sum((ibi < 0.3) | (ibi > 1.2))),

        # FHR features
        "mean_fhr": float(np.mean(fhr_series)),
        "median_fhr": float(np.median(fhr_series)),
        "min_fhr": float(np.min(fhr_series)),
        "max_fhr": float(np.max(fhr_series)),
        "fhr_range": float(np.max(fhr_series) - np.min(fhr_series)),
        "short_term_variability": float(np.std(fhr_series))
    }

    return features


def get_beat_analysis(peaks, sr):
    """Get detailed beat timing analysis"""
    beat_times = peaks / sr
    beat_numbers = np.arange(1, len(beat_times) + 1)
    intervals = np.diff(beat_times)
    
    return {
        "beat_count": len(beat_times),
        "beat_times": beat_times.tolist(),
        "beat_numbers": beat_numbers.tolist(),
        "beat_intervals": intervals.tolist()
    }


def parse_gestation_weeks(gestation_period):
    """
    Parse gestation period string to extract weeks
    
    Args:
        gestation_period: String like "24 weeks", "32", "24w", etc.
        
    Returns:
        Integer weeks or None if parsing fails
    """
    import re
    if not gestation_period:
        return None
    
    # Extract numbers from string
    match = re.search(r'(\d+)', str(gestation_period))
    if match:
        return int(match.group(1))
    return None


def get_fhr_normal_range(gestation_weeks):
    """
    Get normal FHR range based on gestation period
    
    Args:
        gestation_weeks: Gestation period in weeks
        
    Returns:
        Tuple of (min_fhr, max_fhr) or None if unknown
    """
    if gestation_weeks is None:
        return None
    
    # FHR ranges by gestation period
    if 5 <= gestation_weeks <= 7:
        return (90, 130)
    elif 8 <= gestation_weeks <= 10:
        return (170, 180)
    elif 11 <= gestation_weeks <= 13:
        return (140, 160)
    elif 14 <= gestation_weeks <= 27:
        return (120, 160)
    elif 28 <= gestation_weeks <= 32:
        return (110, 160)
    elif 33 <= gestation_weeks <= 40:
        return (110, 150)
    else:
        # Outside typical range, use most common range
        return (110, 160)


def calculate_risk_probability(fhr, min_fhr, max_fhr):
    """
    Calculate risk probabilities for bradycardia and tachycardia
    
    Args:
        fhr: Measured fetal heart rate
        min_fhr: Minimum normal FHR
        max_fhr: Maximum normal FHR
        
    Returns:
        Dictionary with risk probabilities
    """
    range_width = max_fhr - min_fhr
    midpoint = (min_fhr + max_fhr) / 2
    
    # Calculate probabilities using sigmoid-like functions
    if fhr < min_fhr:
        # Bradycardia probability increases as FHR decreases
        deviation = min_fhr - fhr
        bradycardia_chance = min(100.0, (deviation / 30.0) * 100)
        bradycardia_chance = max(50.0, bradycardia_chance)  # At least 50% if below threshold
        tachycardia_chance = max(0.0, 5.0 - deviation * 0.5)  # Very low
        normal_chance = max(0.0, 100 - bradycardia_chance - tachycardia_chance)
    elif fhr > max_fhr:
        # Tachycardia probability increases as FHR increases
        deviation = fhr - max_fhr
        tachycardia_chance = min(100.0, (deviation / 30.0) * 100)
        tachycardia_chance = max(50.0, tachycardia_chance)  # At least 50% if above threshold
        bradycardia_chance = max(0.0, 5.0 - deviation * 0.5)  # Very low
        normal_chance = max(0.0, 100 - tachycardia_chance - bradycardia_chance)
    else:
        # Normal range - calculate proximity to boundaries
        distance_to_min = abs(fhr - min_fhr)
        distance_to_max = abs(fhr - max_fhr)
        distance_to_midpoint = abs(fhr - midpoint)
        
        # Normal chance is highest at midpoint
        normal_chance = 100.0 - (distance_to_midpoint / range_width * 40)
        normal_chance = max(60.0, min(100.0, normal_chance))
        
        # Small chances for abnormalities based on proximity to boundaries
        bradycardia_chance = max(0.0, (1 - distance_to_min / range_width) * 20)
        tachycardia_chance = max(0.0, (1 - distance_to_max / range_width) * 20)
        
        # Normalize to ensure they sum to 100
        total = normal_chance + bradycardia_chance + tachycardia_chance
        normal_chance = (normal_chance / total) * 100
        bradycardia_chance = (bradycardia_chance / total) * 100
        tachycardia_chance = (tachycardia_chance / total) * 100
    
    return {
        'normal_chance': round(normal_chance, 2),
        'bradycardia_chance': round(bradycardia_chance, 2),
        'tachycardia_chance': round(tachycardia_chance, 2)
    }


def get_severity_details(severity, deviation):
    """Get detailed severity information"""
    severity_info = {
        'mild': {
            'level': 1,
            'description': 'Slight deviation from normal range',
            'urgency': 'Monitor closely, consult if persists',
            'risk_level': 'Low to Moderate'
        },
        'moderate': {
            'level': 2,
            'description': 'Significant deviation from normal range',
            'urgency': 'Medical consultation recommended soon',
            'risk_level': 'Moderate to High'
        },
        'severe': {
            'level': 3,
            'description': 'Critical deviation from normal range',
            'urgency': 'Immediate medical attention required',
            'risk_level': 'High to Critical'
        },
        'none': {
            'level': 0,
            'description': 'Within normal range',
            'urgency': 'Continue routine monitoring',
            'risk_level': 'Low'
        }
    }
    
    return severity_info.get(severity, severity_info['none'])


def analyze_fhr_status(fhr, gestation_period):
    """
    Analyze FHR and determine if it's normal, bradycardia, or tachycardia
    with probability scores and detailed risk assessment
    
    Args:
        fhr: Fetal heart rate in bpm
        gestation_period: Gestation period string (e.g., "24 weeks")
        
    Returns:
        Dictionary with comprehensive FHR status analysis including probabilities
    """
    weeks = parse_gestation_weeks(gestation_period)
    normal_range = get_fhr_normal_range(weeks)
    
    if fhr is None or normal_range is None:
        return {
            'fhr_status': 'unknown',
            'fhr_classification': 'Unable to determine',
            'reason': 'Insufficient data for FHR analysis',
            'normal_chance': 0,
            'bradycardia_chance': 0,
            'tachycardia_chance': 0
        }
    
    min_fhr, max_fhr = normal_range
    
    # Calculate risk probabilities
    risk_probs = calculate_risk_probability(fhr, min_fhr, max_fhr)
    
    result = {
        'gestation_weeks': weeks,
        'normal_range_min': min_fhr,
        'normal_range_max': max_fhr,
        'measured_fhr': fhr,
        'normal_chance': risk_probs['normal_chance'],
        'bradycardia_chance': risk_probs['bradycardia_chance'],
        'tachycardia_chance': risk_probs['tachycardia_chance']
    }
    
    if fhr < min_fhr:
        severity = 'mild' if fhr >= min_fhr - 10 else 'moderate' if fhr >= min_fhr - 20 else 'severe'
        severity_details = get_severity_details(severity, fhr - min_fhr)
        
        result.update({
            'fhr_status': 'bradycardia',
            'fhr_classification': 'Bradycardia (Low Heart Rate)',
            'severity': severity,
            'severity_level': severity_details['level'],
            'severity_description': severity_details['description'],
            'urgency': severity_details['urgency'],
            'risk_level': severity_details['risk_level'],
            'deviation': fhr - min_fhr,
            'deviation_percentage': round(((min_fhr - fhr) / min_fhr) * 100, 2),
            'medical_concern': 'Yes - Heart rate below normal range',
            'recommendation': 'Immediate medical consultation recommended. Bradycardia may indicate fetal distress or conduction issues.'
        })
    elif fhr > max_fhr:
        severity = 'mild' if fhr <= max_fhr + 10 else 'moderate' if fhr <= max_fhr + 20 else 'severe'
        severity_details = get_severity_details(severity, fhr - max_fhr)
        
        result.update({
            'fhr_status': 'tachycardia',
            'fhr_classification': 'Tachycardia (High Heart Rate)',
            'severity': severity,
            'severity_level': severity_details['level'],
            'severity_description': severity_details['description'],
            'urgency': severity_details['urgency'],
            'risk_level': severity_details['risk_level'],
            'deviation': fhr - max_fhr,
            'deviation_percentage': round(((fhr - max_fhr) / max_fhr) * 100, 2),
            'medical_concern': 'Yes - Heart rate above normal range',
            'recommendation': 'Medical consultation recommended. Tachycardia may indicate fetal stress, maternal fever, or other concerns.'
        })
    else:
        severity_details = get_severity_details('none', 0)
        
        result.update({
            'fhr_status': 'normal',
            'fhr_classification': 'Normal Heart Rate',
            'severity': 'none',
            'severity_level': 0,
            'severity_description': severity_details['description'],
            'urgency': severity_details['urgency'],
            'risk_level': severity_details['risk_level'],
            'deviation': 0,
            'deviation_percentage': 0,
            'medical_concern': 'No',
            'recommendation': 'Heart rate is within normal range for gestational age. Continue routine prenatal care.'
        })
    
    return result


# ============================================================================
# ONNX Predictor Class
# ============================================================================

class ONNXPredictor:
    def __init__(self, onnx_model_path, config=None):
        """
        Initialize ONNX predictor
        
        Args:
            onnx_model_path: Path to ONNX model file
            config: Configuration dict for preprocessing
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(onnx_model_path))
        
        # Get input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Default config
        if config is None:
            config = {
                'in_channel': 3,
                'duration': 5,
                'delta': True,
                'norm': True,
                'mel_bins': 128
            }
        
        self.config = config
        self.class_labels = ['Normal', 'Abnormal']
        
        print(f"ONNX model loaded: {onnx_model_path}")
        print(f"Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"Output: {self.output_name}")
        print(f"Config: {self.config}")
    
    def extract_features(self, audio_path):
        """
        Extract features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed feature array
        """
        # Read audio with filtering
        audio, sr = read_audio(audio_path, filter=True)
        
        # Extract mel-spectrogram
        feature = LogMelExtractor(audio, sr, mel_bins=self.config['mel_bins'], log=True, snv=False)
        
        # Apply normalization if required
        if self.config['norm']:
            feature = standard_normal_variate(feature)
        
        # Add deltas if required
        if self.config['delta']:
            delta = librosa.feature.delta(feature)
            delta_2 = librosa.feature.delta(delta)
            feature = np.concatenate((feature, delta, delta_2), axis=0)
        
        # Pad or crop to expected time dimension
        # Expected shape from model: (mel_bins * channels, time_frames)
        expected_time = self.input_shape[2]  # 216
        current_time = feature.shape[1]
        
        if current_time < expected_time:
            # Pad with zeros
            pad_width = expected_time - current_time
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        elif current_time > expected_time:
            # Crop to expected length
            feature = feature[:, :expected_time]
        
        return feature
    
    def predict_file(self, audio_path, include_heart_analysis=True, gestation_period=None):
        """
        Make prediction on a single audio file
        
        Args:
            audio_path: Path to audio file
            include_heart_analysis: Whether to include heart rate analysis
            gestation_period: Gestation period string (e.g., "24 weeks") for FHR analysis
            
        Returns:
            Dictionary with prediction results and heart rate features
        """
        # Read audio for heart rate analysis
        audio, sr = read_audio(audio_path, filter=True)
        
        # Extract features for model prediction
        feature = self.extract_features(audio_path)
        
        # Prepare input (add batch dimension)
        feature_array = np.expand_dims(feature, axis=0).astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: feature_array})
        
        # Process output (log softmax output from model)
        log_probs = outputs[0][0]  # Shape: (num_classes,)
        probabilities = np.exp(log_probs)  # Convert log probabilities to probabilities
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        result = {
            'file': Path(audio_path).name,
            'predicted_class': int(predicted_class),
            'predicted_label': self.class_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_labels[i]: float(probabilities[i]) 
                for i in range(len(self.class_labels))
            }
        }
        
        # Add heart rate analysis if requested
        if include_heart_analysis:
            try:
                # Extract envelope and detect peaks
                envelope = extract_envelope(audio)
                peaks = detect_peaks(envelope, sr)
                
                # Compute FHR and extract features
                ibi, fhr = compute_fhr(peaks, sr)
                
                if fhr is not None:
                    result['heart_rate'] = {
                        'average_fhr': fhr,
                        'beat_count': len(peaks)
                    }
                    
                    # Extract detailed features
                    heart_features = extract_heart_features(ibi)
                    if heart_features:
                        result['heart_rate'].update(heart_features)
                    
                    # Analyze FHR status based on gestation period
                    if gestation_period:
                        fhr_status = analyze_fhr_status(fhr, gestation_period)
                        result['fhr_analysis'] = fhr_status
                    
                    # Add beat analysis (optional, can be large)
                    # beat_analysis = get_beat_analysis(peaks, sr)
                    # result['beat_analysis'] = beat_analysis
                else:
                    result['heart_rate'] = {
                        'error': 'Insufficient beats detected for heart rate analysis'
                    }
                    
            except Exception as e:
                result['heart_rate'] = {
                    'error': f'Heart rate analysis failed: {str(e)}'
                }
        
        return result
    
    def predict_batch(self, audio_paths, output_csv=None):
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            output_csv: Optional path to save results as CSV
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"\nPredicting on {len(audio_paths)} files...")
        for i, audio_path in enumerate(audio_paths, 1):
            try:
                result = self.predict_file(str(audio_path))
                results.append(result)
                print(f"[{i}/{len(audio_paths)}] {result['file']}: {result['predicted_label']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"[{i}/{len(audio_paths)}] Error processing {audio_path}: {e}")
                results.append({
                    'file': Path(audio_path).name,
                    'error': str(e)
                })
        
        # Save to CSV if requested
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict using ONNX model')
    parser.add_argument('--model', type=str, default='model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--in-channel', type=int, default=3,
                        help='Number of input channels (1=mel, 3=mel+delta+delta2)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Audio duration in seconds')
    parser.add_argument('--no-delta', action='store_true',
                        help='Disable delta features')
    parser.add_argument('--no-norm', action='store_true',
                        help='Disable normalization')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'in_channel': args.in_channel,
        'duration': args.duration,
        'delta': not args.no_delta,
        'norm': not args.no_norm,
        'mel_bins': 128
    }
    
    # Initialize predictor
    predictor = ONNXPredictor(args.model, config)
    
    # Handle input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        print(f"\nPredicting on single file: {input_path}")
        result = predictor.predict_file(str(input_path))
        
        print("\n" + "=" * 60)
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print("=" * 60)
        
    elif input_path.is_dir():
        # Directory of files
        audio_files = list(input_path.glob('*.wav'))
        if not audio_files:
            print(f"No .wav files found in {input_path}")
            return
        
        results = predictor.predict_batch(audio_files, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            print(f"Total predictions: {len(valid_results)}")
            
            # Count by class
            for label in predictor.class_labels:
                count = sum(1 for r in valid_results if r['predicted_label'] == label)
                pct = 100 * count / len(valid_results)
                print(f"{label}: {count} ({pct:.1f}%)")
            
            # Average confidence
            avg_conf = np.mean([r['confidence'] for r in valid_results])
            print(f"Average confidence: {avg_conf:.2%}")
        
        # Errors
        errors = [r for r in results if 'error' in r]
        if errors:
            print(f"\nErrors: {len(errors)} files failed")
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()
