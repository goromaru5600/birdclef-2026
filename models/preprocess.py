"""Shared audio preprocessing utilities."""
import numpy as np
import librosa

SR       = 32000
DURATION = 5
N_FFT    = 1024
HOP_LEN  = 320
N_MELS   = 128
FMIN     = 20
FMAX     = 16000


def audio_to_melspec(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)


def load_chunk(path: str, offset: float = 0.0, duration: float = DURATION,
               sr: int = SR) -> np.ndarray:
    """Load exactly `duration` seconds starting at `offset`."""
    audio, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    chunk_samples = int(duration * sr)
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    return audio[:chunk_samples]


def split_soundscape(path: str, chunk_duration: float = DURATION,
                     sr: int = SR) -> list[tuple[int, np.ndarray]]:
    """Split a long soundscape into (end_second, audio_chunk) pairs."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        end_sec = int((i + 1) * chunk_duration)
        chunks.append((end_sec, chunk))
    return chunks
