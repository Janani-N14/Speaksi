import os
import shutil
import subprocess

def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    Convert an incoming audio file to standard PCM WAV format.
    Uses ffmpeg if available; falls back to librosa/soundfile.
    """
    # Try using ffmpeg if available
    if shutil.which("ffmpeg"):
        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(sample_rate),
                output_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception as e:
            print(f"[Warning] ffmpeg conversion failed: {e}. Falling back to librosa.")

    # Fallback to librosa & soundfile
    try:
        import librosa
        import soundfile as sf
        audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        sf.write(output_path, audio, sample_rate, subtype='PCM_16')
        return output_path
    except Exception as e:
        raise RuntimeError(f"Could not convert audio file: {e}")

def generate_tts_audio(text: str, output_path: str) -> str:
    """
    Generate reference TTS audio from text using gTTS.
    """
    from gtts import gTTS
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

def extract_mfcc_features(audio_path: str, n_mfcc: int = 13):
    """
    Extract normalized MFCC features from an audio file.
    """
    import numpy as np
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    std = np.std(mfcc)
    if std == 0:
        std = 1e-6
    return (mfcc - np.mean(mfcc)) / std

def pad_segment_to_length(seg1, seg2):
    """
    Pad two 2D numpy segments along axis 1 to matching maximum length.
    """
    import numpy as np
    max_len = max(seg1.shape[1], seg2.shape[1])
    if seg1.shape[1] < max_len:
        pad_width = max_len - seg1.shape[1]
        seg1 = np.pad(seg1, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    if seg2.shape[1] < max_len:
        pad_width = max_len - seg2.shape[1]
        seg2 = np.pad(seg2, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return seg1, seg2

def compute_overall_similarity(distances: list) -> float:
    """
    Compute percentage similarity from Euclidean distances across segments.
    """
    import numpy as np
    if not distances:
        return 0.0
    min_dist = min(distances)
    max_dist = max(distances)
    avg_dist = sum(distances) / len(distances)

    if max_dist == min_dist:
        normalized_score = 1.0
    else:
        shift_factor = 0.1
        scaling_factor = 2.0
        normalized_score = ((max_dist - avg_dist) / (max_dist - min_dist)) * scaling_factor + shift_factor

    percentage = float(np.clip(normalized_score * 100, 0.0, 100.0))
    return round(percentage, 2)

def calculate_acoustic_similarity(tts_audio_path: str, user_audio_path: str) -> float:
    """
    Generate similarity score by comparing MFCC features between TTS reference and user audio.
    """
    import numpy as np
    mfcc1 = extract_mfcc_features(tts_audio_path)
    mfcc2 = extract_mfcc_features(user_audio_path)

    # Normalize per coefficient
    mfcc1 = (mfcc1 - np.mean(mfcc1, axis=1, keepdims=True)) / (np.std(mfcc1, axis=1, keepdims=True) + 1e-6)
    mfcc2 = (mfcc2 - np.mean(mfcc2, axis=1, keepdims=True)) / (np.std(mfcc2, axis=1, keepdims=True) + 1e-6)

    segment_length = 20
    segments1 = [mfcc1[:, i:i+segment_length] for i in range(0, mfcc1.shape[1], segment_length)]
    segments2 = [mfcc2[:, i:i+segment_length] for i in range(0, mfcc2.shape[1], segment_length)]

    distances = []
    min_len = min(len(segments1), len(segments2))
    for i in range(min_len):
        s1, s2 = pad_segment_to_length(segments1[i], segments2[i])
        dist = float(np.linalg.norm(s1 - s2))
        distances.append(dist)

    if not distances:
        return 50.0

    return compute_overall_similarity(distances)
