import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fftpack import fft
import os
import tempfile
from sklearn.linear_model import LinearRegression
from pydub.exceptions import CouldntDecodeError
import subprocess

from lab_tools import ytutil


# YouTubeから音声をダウンロードし、指定範囲の音声データを返す
def download_and_process_audio(youtube_url: str, start_time: float, end_time: float = None):
    filename = ytutil.download_youtube(youtube_url)
    audio = AudioSegment.from_mp3(filename)
    os.remove(filename)

    # 時間範囲をミリ秒単位に変換
    if end_time is None:
        end_time = len(audio) / 1000.0  # 音声の長さを秒単位で取得
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    audio = audio[start_ms:end_ms]

    return audio, filename


def convert_to_mp3(input_file: str, output_file: str):
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-q:a", "0", "-map", "a", output_file],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpegによるMP3変換に失敗しました: {e.stderr.decode()}")


def process_audio(filename: str, start_time: float, end_time: float = None):
    mp3_filename = filename.rsplit('.', 1)[0] + ".mp3"

    if not filename.endswith(".mp3"):
        convert_to_mp3(filename, mp3_filename)
        filename = mp3_filename

    try:
        audio = AudioSegment.from_mp3(filename)
    except CouldntDecodeError:
        raise ValueError(f"MP3ファイル {filename} のデコードに失敗しました。ファイルが破損している可能性があります。")

    os.remove(filename)

    # 時間範囲をミリ秒単位に変換
    if end_time is None:
        end_time = len(audio) / 1000.0  # 音声の長さを秒単位で取得
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    audio = audio[start_ms:end_ms]

    return audio, mp3_filename  # 変換後のファイル名を返す


# 音声データをモノラルに変換して、NumPy配列とサンプリングレートを返す
def convert_audio_to_numpy(audio: AudioSegment):
    data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # ステレオ音声の場合はモノラルに変換
    if audio.channels > 1:
        data = data.reshape((-1, audio.channels)).mean(axis=1)

    return data, sample_rate


# パワースペクトルを計算
def compute_power_spectrum(data: np.ndarray, sample_rate: int):
    N = len(data)
    T = 1.0 / sample_rate
    yf = fft(data)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    power_spectrum = 2.0 / N * np.abs(yf[:N // 2])

    # 対数スケールに変換
    xf_log = np.log10(xf[1:])
    power_spectrum_log = np.log10(power_spectrum[1:])

    return xf_log, power_spectrum_log


# 線形回帰によるパワースペクトルの近似
def perform_linear_regression(xf_log: np.ndarray, power_spectrum_log: np.ndarray):
    xf_log_reshaped = xf_log.reshape(-1, 1)
    reg = LinearRegression().fit(xf_log_reshaped, power_spectrum_log)
    power_spectrum_pred = reg.predict(xf_log_reshaped)

    return power_spectrum_pred, reg


# パワースペクトルと線形近似のグラフを保存
def plot_and_save_spectrum(xf_log: np.ndarray, power_spectrum_log: np.ndarray, power_spectrum_pred: np.ndarray, sample_rate: int):
    graphfile_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(10**xf_log, 10**power_spectrum_log, label='Power Spectrum')
    plt.plot(10**xf_log, 10**power_spectrum_pred, color='red', linestyle='--', label='Linear Approximation')
    plt.xscale('log')
    plt.yscale('log')

    plt.title('Power Spectrum (Log Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, which="both", ls="--")
    plt.xlim([1, sample_rate // 2])
    plt.legend()
    plt.savefig(graphfile_path)

    return graphfile_path


# メインの解析関数 Use YouTube
def analyze_1f_noise(mode: str, youtube_url: str, file_path: str, start_time: float = 0, end_time: float = None):
    # 音声データの取得
    if mode == "YouTube":
        audio, filename = download_and_process_audio(youtube_url, start_time, end_time)
    else:
        audio, filename = process_audio(file_path, start_time, end_time)

    # 音声データをNumPy配列に変換
    data, sample_rate = convert_audio_to_numpy(audio)

    # パワースペクトルを計算
    xf_log, power_spectrum_log = compute_power_spectrum(data, sample_rate)

    # 線形回帰を実行
    power_spectrum_pred, _ = perform_linear_regression(xf_log, power_spectrum_log)

    # グラフを描画して保存
    graphfile_path = plot_and_save_spectrum(xf_log, power_spectrum_log, power_spectrum_pred, sample_rate)

    # ファイル名を返す
    filename, _ = os.path.splitext(filename)
    return filename, graphfile_path
