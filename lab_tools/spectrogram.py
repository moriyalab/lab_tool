import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import fftconvolve
from lab_tools import labutils
from lab_tools import filter
import math
import os


# モルレーウェーブレットの計算
def calculate_morlet_wavelet(time_array, frequency, wavelet_width):
    scale_factor = frequency / wavelet_width
    std_time = 1 / (2 * math.pi * scale_factor)
    amplitude = 1 / (std_time * np.sqrt(2 * math.pi))
    exp_component = -np.power(time_array, 2) / (2 * std_time**2)
    oscillatory_component = 1j * 2 * math.pi * frequency * time_array
    return amplitude * np.exp(oscillatory_component + exp_component)


# 連続ウェーブレット変換 (Continuous Wavelet Transform)
def perform_cwt(sample_rate, signal_data, max_frequency, wavelet_width=48, wavelet_range=0.5):
    time_step = 1 / sample_rate
    wavelet_time_array = np.arange(-wavelet_range, wavelet_range, time_step)
    signal_length = len(signal_data)
    cwt_matrix = np.zeros((max_frequency, signal_length))

    # モルレーウェーブレットを全て事前計算
    wavelets = [
        calculate_morlet_wavelet(wavelet_time_array, freq + 1, wavelet_width)
        for freq in range(max_frequency)
    ]

    for freq, wavelet in enumerate(wavelets):
        convolution_result = fftconvolve(signal_data, wavelet, mode='same')
        cwt_matrix[freq, :] = (2 * np.abs(convolution_result) / sample_rate) ** 2

    return cwt_matrix


# CWTの結果をプロットする関数
def plot_cwt_result(cwt_matrix, time_array, max_frequency):
    plt.subplots(figsize=(12, 6))
    plt.imshow(cwt_matrix, cmap='jet', aspect='auto',
               extent=[time_array[0], time_array[-1], max_frequency, 0],
               vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power")
    plt.clim(0, 5)
    plt.gca().invert_yaxis()


# 短時間フーリエ変換 (STFT) のスペクトログラムをプロットする関数
def plot_stft_spectrogram(signal_data, sample_rate, segment_length, overlap=0.5, max_frequency=None):
    """
    短時間フーリエ変換(STFT)スペクトログラムをプロットします。

    Parameters:
        signal_data (np.ndarray): 信号データ
        sample_rate (int): サンプリングレート
        segment_length (int): セグメント長(nperseg)
        overlap (float): セグメントのオーバーラップ率 (0.0~1.0)
        max_frequency (float, optional): 表示する最大周波数
    """
    # オーバーラップするサンプル数を計算
    noverlap = int(segment_length * overlap)

    # STFTの計算
    frequencies, times, stft_result = signal.stft(
        signal_data, fs=sample_rate, window='hamming', nperseg=segment_length, noverlap=noverlap
    )
    amplitude = np.abs(stft_result)
    amplitude[amplitude == 0] = np.finfo(float).eps  # 振幅がゼロの箇所を微小値に置き換え

    # プロット

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrogram = ax.pcolormesh(times, frequencies, amplitude, cmap='jet', shading="auto", vmin=0, vmax=5)
    fig.colorbar(spectrogram, ax=ax, orientation="vertical").set_label("Power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    # 最大周波数の設定
    if max_frequency:
        ax.set_ylim([0, max_frequency])


# UI処理: スペクトログラム生成・信号プロット
def generate_spectrogram_and_signal_plot(
        uploaded_file, analysis_method,
        sample_rate, max_frequency, signal_column_name, start_time, end_time,
        filter_type, highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation):
    file_path = uploaded_file.name
    file_basename = os.path.basename(file_path)

    signal_data = labutils.load_signal(file_path, signal_column_name)
    if len(signal_data) == 0:
        return None, None

    output_dir = "/tmp/spectrogram/"
    os.makedirs(output_dir, exist_ok=True)

    # フィルタ処理
    timestamps = labutils.load_signal(file_path, "Timestamp")
    delta_time = timestamps[1] - timestamps[0]
    actual_sample_rate = 1.0 / delta_time
    if filter_type == "High PASS":
        signal_data = filter.highpass(signal_data, actual_sample_rate, highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation)
    elif filter_type == "Low PASS":
        signal_data = filter.lowpass(signal_data, actual_sample_rate, highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation)

    # 時間配列を計算
    time_array = np.arange(0, len(signal_data) / sample_rate, 1 / sample_rate)

    # 開始時間と終了時間の範囲に基づきデータをトリミング
    start_index = int(start_time * sample_rate)
    end_index = int(end_time * sample_rate)
    signal_data = signal_data[start_index:end_index]
    time_array = time_array[start_index:end_index]

    # 信号プロットの保存
    window_size = 50
    smoothed_signal = np.convolve(signal_data, np.ones(window_size) / window_size, mode='valid')
    trimmed_time_array = time_array[:len(smoothed_signal)]

    title = f"Signal(File: {file_basename}, Convolve Window Size: {window_size})"

    plt.figure(dpi=200)
    plt.subplots(figsize=(12, 6))
    plt.title(title)
    plt.plot(trimmed_time_array, smoothed_signal)
    plt.xlim(start_time, end_time)
    plt.xlabel("Time [sec]")
    plt.ylabel("Voltage [uV]")
    signal_plot_path = os.path.join(output_dir, "signal_plot.png")
    plt.savefig(signal_plot_path)

    # スペクトログラムプロットの保存
    if analysis_method == "Short-Time Fourier Transform":
        plt.figure(dpi=200)
        plot_stft_spectrogram(signal_data, sample_rate=sample_rate, segment_length=4096, overlap=0.99, max_frequency=max_frequency)
        spectrogram_plot_path = os.path.join(output_dir, "stft_spectrogram_plot.png")
        title = f"Spectrogram (File: {file_basename}, Method: STFT, Fs: {sample_rate} Hz, Segment Length: {4096}, Overlap: {0.99*100:.0f}%)"
        plt.title(title)
        plt.savefig(spectrogram_plot_path)
    else:
        spectrogram_plot_path = os.path.join(output_dir, "wavelet_spectrogram_plot.png")
        cwt_matrix = perform_cwt(sample_rate, signal_data, max_frequency)
        plt.figure(dpi=200)
        plot_cwt_result(cwt_matrix, time_array, max_frequency)
        title = f"Spectrogram (File: {file_basename}, Method: Wavelet, Fs: {sample_rate} Hz, Wavelet Width: {48})"
        plt.title(title)
        plt.savefig(spectrogram_plot_path)

    return spectrogram_plot_path, signal_plot_path
