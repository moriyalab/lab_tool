import sys
import numpy as np
import matplotlib.pyplot as plt
import math


# ファイルから信号データを読み込む
def load_signal(file_path):
    signal = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                signal.append(float(line))
            except ValueError as e:
                print(e, file=sys.stderr)
    return signal


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
        conv_result = np.convolve(
            data,
            morlet(wavelet_length, i + 1, width),
            mode='same')
        cwt_result[i, :] = (2 * np.abs(conv_result) / Fs) ** 2

    return cwt_result


# 連続ウェーブレット変換結果をカラーマップとしてプロット
def plot_cwt(cwt_result, time_data, fmax):
    plt.imshow(
        cwt_result,
        cmap='jet',
        aspect='auto',
        vmax=abs(cwt_result).max(),
        vmin=-abs(cwt_result).max())
    plt.xlabel("Time [min]")
    plt.ylabel("Frequency [Hz]")
    plt.axis([0, len(time_data) / 1000, 0, fmax - 1])
    plt.xticks(np.arange(0, 1200001, step=60000))
    plt.ticklabel_format(style='plain', axis='x')
    plt.colorbar()
    plt.clim(-5, 5)


if __name__ == "__main__":
    # サンプリング設定
    Fs = 1000  # サンプリング周波数
    Ts = 1 / Fs  # 時間ステップ
    time_S = 1200  # 信号長さ（秒）
    t_data = np.arange(0, time_S, Ts)  # 時間データ

    # 信号データを読み込み
    signal = load_signal('fp2.txt')

    # 連続ウェーブレット変換
    fmax = 60
    cwt_signal = continuous_wavelet_transform(Fs=Fs, data=signal, fmax=fmax)

    # 信号のプロット
    plt.figure(dpi=200)
    plt.title("Signal")
    plt.plot(t_data, signal)
    plt.xlim(0, 1200)
    plt.xticks(np.arange(0, 1201, step=100))
    plt.xlabel("Time [s]")
    plt.savefig("signal_fp2.png")

    # 連続ウェーブレット変換のプロット
    plt.figure(dpi=200)
    plot_cwt(cwt_signal, t_data, fmax)
    plt.savefig("cwt_fp2.png")
    plt.show()
