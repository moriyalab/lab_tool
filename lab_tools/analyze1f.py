import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fftpack import fft
import yt_dlp
import os
import tempfile


def download_youtube(youtube_url: str) -> str:
    ydl_opts = {
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }
        ],
        'outtmpl': '%(title)s.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        file_path = ydl.prepare_filename(info_dict)
        filename, _ = os.path.splitext(file_path)
        filename += ".mp3"
        print(f"Downloaded file path: {filename}")

    return filename


def analyze_1f_noise(youtube_url: str):
    filename = download_youtube(youtube_url)
    audio = AudioSegment.from_mp3(filename)
    os.remove(filename)
    data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    if audio.channels > 1:
        data = data.reshape((-1, audio.channels)).mean(axis=1)

    N = len(data)
    T = 1.0 / sample_rate
    yf = fft(data)
    xf = np.fft.fftfreq(N, T)[:N//2]

    power_spectrum = 2.0/N * np.abs(yf[:N//2])

    xf_log = xf[1:]
    power_spectrum_log = power_spectrum[1:]

    graphfile_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(xf_log, power_spectrum_log)
    plt.xscale('log')
    plt.yscale('log')

    plt.title('Power Spectrum (Log Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, which="both", ls="--")
    plt.xlim([1, sample_rate // 2])
    plt.savefig(graphfile_path)

    filename, _ = os.path.splitext(filename)

    return filename, graphfile_path
