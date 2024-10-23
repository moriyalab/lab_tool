import numpy as np
from scipy import signal
from scipy import fftpack
from typing import Tuple


def highpass_filter(signal_data: np.ndarray, samplerate: float, fp: float, fs: float, gpass: float, gstop: float) -> np.ndarray:
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "high")

    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal


def overlap_frames(signal_data: np.ndarray, samplerate: float, frame_size: int, overlap: float) -> Tuple[np.ndarray, int]:
    total_duration = len(signal_data) / samplerate
    frame_duration = frame_size / samplerate
    step_size = frame_size * (1 - overlap / 100)

    num_frames = int((total_duration - (frame_duration * overlap / 100)) / (frame_duration * (1 - overlap / 100)))

    frames = []

    for i in range(num_frames):
        start_idx = int(step_size * i)
        frames.append(signal_data[start_idx:start_idx + frame_size])

    return np.array(frames), num_frames


def hanning(signal_data: np.ndarray, frame_size: int, num_frames: int) -> Tuple[np.ndarray, float]:
    han = signal.get_window('hann', frame_size)
    acf = 1 / (sum(han) / frame_size)

    for i in range(num_frames):
        signal_data[i] *= han

    return signal_data, acf


def fft_ave(signal_data: np.ndarray, samplerate: float, frame_size: int, num_frames: int, acf: float):
    fft_array = []
    for i in range(num_frames):
        fft_result = fftpack.fft(signal_data[i]) / frame_size
        fft_array.append(acf * np.abs(fft_result))

    fft_axis = np.linspace(0, samplerate / 2, frame_size // 2)
    fft_array = np.array(fft_array)[:, :frame_size // 2]
    fft_mean = np.mean(fft_array, axis=0)

    return fft_array, fft_mean, fft_axis


def linear_to_db(x: float, y: float) -> float:
    if y == 0:
        raise ValueError("y cannot be zero in logarithmic conversion")
    return 20 * np.log10(x / y)
