import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fftpack import fft
import yt_dlp
import os

url = "https://youtu.be/Ci_zad39Uhw?si=AhB9ArgrWUvbPiv5"

ydl_opts = {
    'postprocessors': [
        {
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }
    ],
    'outtmpl': '%(title)s.%(ext)s'  # ファイル名のテンプレート
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=True)
    file_path = ydl.prepare_filename(info_dict)
    filename, ext = os.path.splitext(file_path)
    filename += ".mp3"
    print(f"Downloaded file path: {filename}")

# MP3ファイルの読み込みとWAV形式への変換
audio = AudioSegment.from_mp3(filename)
os.remove(filename)
data = np.array(audio.get_array_of_samples())
sample_rate = audio.frame_rate

# モノラル変換（ステレオの場合）
if audio.channels > 1:
    data = data.reshape((-1, audio.channels)).mean(axis=1)

# フーリエ変換の実行
N = len(data)
T = 1.0 / sample_rate
yf = fft(data)
xf = np.fft.fftfreq(N, T)[:N//2]

# パワースペクトルの計算
power_spectrum = 2.0/N * np.abs(yf[:N//2])

# プロット用に周波数とパワーを制限
xf_log = xf[1:]  # 0Hzを除去 (ログスケールでは0が扱えないため)
power_spectrum_log = power_spectrum[1:]

# 縦軸の範囲を指定（例: 0から100まで）
y_min = 0
y_max = 1000

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(xf_log, power_spectrum_log)
plt.xscale('log')
plt.yscale('log')

plt.title('Power Spectrum (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True, which="both", ls="--")
plt.xlim([1, sample_rate // 2])  # 1Hz から Nyquist周波数 (sample_rate/2) まで

# 縦軸の範囲を指定
# plt.ylim([y_min, y_max])


plt.show()
