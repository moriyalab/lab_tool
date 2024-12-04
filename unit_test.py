# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
from lab_tools import spectrogram


def spectrogram_test():
    # パラメータ設定
    Fs = 1000          # サンプリング周波数 (Hz)
    duration = 5       # 信号の長さ（秒）
    f0, f1 = 0, 50     # チャープ信号の開始・終了周波数
    mod_freq = 1.0     # 振幅変調の周波数（Hz）
    fmax = 50          # CWTの最大周波数

    # 時間ベクトル
    t = np.arange(0, duration, 1 / Fs)

    # チャープ信号（0から50 Hzに周波数が変化）
    chirp_signal = np.sin(2 * np.pi * ((f1 - f0) / duration * t**2 / 2 + f0 * t))

    # 振幅変調（1 Hzの正弦波で振幅を変化させる）
    modulation = (np.sin(2 * np.pi * mod_freq * t) + 1) / 0.2  # 0〜1の範囲にスケーリング
    modulated_signal = chirp_signal * modulation

    plt.figure(dpi=200)
    plt.subplot(2, 1, 1)
    spectrogram.plot_stft_spectrogram(modulated_signal, sample_rate=Fs, segment_length=512+256, overlap=0.99, max_frequency=fmax)
    spectrogram_filename = "cwt_result.png"

    plt.savefig(spectrogram_filename)

    plt.figure(figsize=(12, 6))
    plt.plot(t, modulated_signal, label="Amplitude Modulated Signal", color="orange")
    plt.title("Amplitude Modulated Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    spectrogram_filename = "signal.png"
    plt.savefig(spectrogram_filename)


if __name__ == "__main__":
    spectrogram_test()
