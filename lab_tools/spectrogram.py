import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from lab_tools import labutils
from lab_tools import filter
import math


def morlet_wavelet(x, f, width):
    sf = f / width
    st = 1 / (2 * math.pi * sf)
    A = 1 / (st * math.sqrt(2 * math.pi))
    h = -np.power(x, 2) / (2 * st**2)
    co1 = 1j * 2 * math.pi * f * x
    return A * np.exp(co1) * np.exp(h)


def continuous_wavelet_transform(Fs, data, fmax, width=48, wavelet_R=0.5):
    Ts = 1 / Fs
    wavelet_length = np.arange(-wavelet_R, wavelet_R, Ts)
    data_length = len(data)
    cwt_result = np.zeros([fmax, data_length])

    for i in range(fmax):
        conv_result = np.convolve(data, morlet_wavelet(wavelet_length, i + 1, width), mode='same')
        cwt_result[i, :] = (2 * np.abs(conv_result) / Fs) ** 2

    return cwt_result


def plot_cwt(cwt_result, time_data, fmax):
    plt.imshow(cwt_result, cmap='jet', aspect='auto',
               extent=[time_data[0], time_data[-1], fmax, 0],
               vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power")
    plt.clim(-5, 5)
    plt.gca().invert_yaxis()


def stft_plot_spectrogram(data, Fs, N, freq_limit=None):
    freqs, times, Zxx = signal.stft(data, fs=Fs, window='hann', nperseg=N, noverlap=None)
    amp = np.abs(Zxx)
    amp[amp == 0] = np.finfo(float).eps
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrogram = ax.pcolormesh(times, freqs, amp, shading="auto", vmin=0, vmax=5)
    fig.colorbar(spectrogram, ax=ax, orientation="vertical").set_label("Amplitude")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    if freq_limit:
        ax.set_ylim([0, freq_limit])
    plt.show()


# グラフ描画とスペクトログラムの処理を行う関数
def spectrogram_ui(
        uploaded_file, analysis_method,
        Fs, fmax, column_name, start_time, end_time,
        filter_setting, fp_hp, fs_hp, gpass, gstop):
    filepath = uploaded_file.name
    signal = labutils.load_signal(filepath, column_name)
    if len(signal) == 0:
        return None, None

    # Filter
    timestamps = labutils.load_signal(filepath, "Timestamp")
    dt = (timestamps[1] - timestamps[0])
    samplerate = 1.0 / dt
    if filter_setting == "High PASS":
        signal = filter.highpass(signal, samplerate, fp_hp, fs_hp, gpass, gstop)
    elif filter_setting == "Low PASS":
        signal = filter.lowpass(signal, samplerate, fp_hp, fs_hp, gpass, gstop)

    # 時間データを計算
    t_data = np.arange(0, len(signal) / Fs, 1 / Fs)

    # スライダーの範囲に基づいてデータをフィルタリング
    start_idx = int(start_time * Fs)
    end_idx = int(end_time * Fs)
    signal = signal[start_idx:end_idx]
    t_data = t_data[start_idx:end_idx]

    # 信号をプロットして保存
    plt.figure(dpi=200)
    plt.title("Signal")
    plt.plot(t_data, signal)
    plt.xlim(start_time, end_time)
    plt.xlabel("Time [sec]")
    plt.ylabel("Voltage [uV]")
    signal_filename = "signal_plot.png"
    plt.savefig(signal_filename)

    # スペクトログラムをプロットして保存
    if analysis_method == "Short-Time Fourier Transform":
        plt.figure(dpi=200)
        stft_plot_spectrogram(data=signal, Fs=Fs, N=256, freq_limit=fmax)
        spectrogram_filename = "stft_spectrogram_plot.png"
        plt.savefig(spectrogram_filename)
    else:
        spectrogram_filename = "wavelet_spectrogram_plot.png"
        cwt_signal = continuous_wavelet_transform(Fs=Fs, data=signal, fmax=fmax)
        plt.figure(dpi=200)
        plot_cwt(cwt_signal, t_data, fmax)
        plt.savefig(spectrogram_filename)

    return spectrogram_filename, signal_filename