import numpy as np
import matplotlib.pyplot as plt
import math
import tempfile

from lab_tools import labutils

# モルレーウェーブレット関数
def morlet(x, f, width):
    sf = f / width
    st = 1 / (2 * math.pi * sf)
    A = 1 / (st * math.sqrt(2 * math.pi))
    h = -np.power(x, 2) / (2 * st**2)
    co1 = 1j * 2 * math.pi * f * x
    return A * np.exp(co1) * np.exp(h)


# 連続ウェーブレット変換
def continuous_wavelet_transform(Fs, data, fmax, width=48, wavelet_R=0.5):
    Ts = 1 / Fs
    wavelet_length = np.arange(-wavelet_R, wavelet_R, Ts)
    data_length = len(data)
    cwt_result = np.zeros([fmax, data_length])

    for i in range(fmax):
        conv_result = np.convolve(data, morlet(wavelet_length, i + 1, width), mode='same')
        cwt_result[i, :] = (2 * np.abs(conv_result) / Fs) ** 2

    return cwt_result


# 連続ウェーブレット変換結果をカラーマップとしてプロット
def plot_cwt(cwt_result, time_data, fmax):
    plt.imshow(cwt_result, cmap='jet', aspect='auto',
               extent=[time_data[0], time_data[-1], 0, fmax],
               vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power")
    plt.clim(-5, 5)


# グラフ描画とCWTの処理を行う関数
def wavelet_ui(uploaded_file, Fs, fmax, column_name, start_time, end_time):
    filepath = uploaded_file.name
    signal = labutils.load_signal(filepath, column_name)

    if len(signal) == 0:
        return None, None

    # 時間データを計算
    t_data = np.arange(0, len(signal) / Fs, 1 / Fs)

    # スライダーの範囲に基づいてデータをフィルタリング
    start_idx = int(start_time * Fs)
    end_idx = int(end_time * Fs)
    signal = signal[start_idx:end_idx]
    t_data = t_data[start_idx:end_idx]

    signal_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    plt.figure(dpi=200)
    plt.title("Signal")
    plt.plot(t_data, signal)
    plt.xlim(start_time, end_time)
    plt.xlabel("Time [sec]")
    plt.ylabel("Voltage [uV]")
    plt.savefig(signal_filename)

    cwt_signal_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    cwt_signal = continuous_wavelet_transform(Fs=Fs, data=signal, fmax=fmax)
    plt.figure(dpi=200)
    plot_cwt(cwt_signal, t_data, fmax)
    plt.savefig(cwt_signal_filename)

    return cwt_signal_filename, signal_filename
