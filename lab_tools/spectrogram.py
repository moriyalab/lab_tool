import math
import os
import yaml
import subprocess
import uuid
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import fftconvolve
from scipy.integrate import simpson, trapezoid
import pandas as pd
from lab_tools import labutils
from lab_tools import filter


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
def plot_cwt_result(cwt_matrix, time_array, max_frequency, fontsize=12):
    plt.subplots(figsize=(12, 6))
    plt.imshow(cwt_matrix, cmap='jet', aspect='auto',
               extent=[time_array[0], time_array[-1], max_frequency, 0],
               vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    plt.xlabel("Time [sec]", fontsize=fontsize)
    plt.ylabel("Frequency [Hz]", fontsize=fontsize)
    plt.colorbar().set_label("Power", fontsize=fontsize)
    plt.clim(0, 5)
    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


def perform_stft(signal_data, sample_rate: int, segment_length: int, overlap=0.5):
    """
    Parameters:
        signal_data (np.ndarray): 信号データ
        sample_rate (int): サンプリングレート
        segment_length (int): セグメント長(nperseg)
        overlap (float): セグメントのオーバーラップ率 (0.0~1.0)
    """
    noverlap = int(segment_length * overlap)
    frequencies, times, stft_result = signal.stft(
        signal_data, fs=sample_rate, window='hamming', nperseg=segment_length, noverlap=noverlap
    )
    amplitude = np.abs(stft_result)
    amplitude[amplitude == 0] = np.finfo(float).eps

    return frequencies, times, amplitude


# 短時間フーリエ変換 (STFT) のスペクトログラムをプロットする関数
def plot_stft_spectrogram(amplitude, frequencies, times, max_frequency=None, fontsize=12):
    """
    Parameters:
        amplitude (np.ndarray): 信号強度データ
        frequencies (np.ndarray): 周波数データ
        times (np.ndarray): 時間データ(nperseg)
        max_frequency (float, optional): 表示する最大周波数
    """
    # プロット

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrogram = ax.pcolormesh(times, frequencies, amplitude, cmap='jet', shading="gourand", vmin=0, vmax=5)
    fig.colorbar(spectrogram, ax=ax, orientation="vertical").set_label("Power", fontsize=fontsize)
    ax.set_xlabel("Time [s]", fontsize=fontsize)
    ax.set_ylabel("Frequency [Hz]", fontsize=fontsize)

    # 最大周波数の設定
    if max_frequency:
        ax.set_ylim([0, max_frequency])

    ax.tick_params(axis='both', which='major', labelsize=fontsize)


def plot_signal(signal_data, time_data, start_time, end_time, fontsize, window_size=50):
    smoothed_signal = np.convolve(signal_data, np.ones(window_size) / window_size, mode='valid')
    trimmed_time_array = time_data[:len(smoothed_signal)]
    plt.subplots(figsize=(12, 6))
    plt.plot(trimmed_time_array, smoothed_signal)
    plt.xlim(start_time, end_time)
    plt.xlabel("Time [sec]", fontsize=fontsize)
    plt.ylabel("Voltage [uV]", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


def calculate_frequency_band_intensity(
        frequency_array, analysis_matrix, freq_band, window_size=10, method="STFT", integration_method="simps"):
    """
    特定の周波数帯域の積分値を0〜36Hzの積分値で正規化した比率を計算する関数

    Parameters:
        frequency_array (np.ndarray): 周波数配列 (STFT の場合は frequency、CWT の場合は range(max_frequency))
        analysis_matrix (np.ndarray): CWT または STFT の解析結果
        freq_band (tuple): 計算する周波数帯 (例: (10, 20))
        method (str): 解析手法 ("STFT" または "CWT")
        integration_method (str): 数値積分の手法 ("trapz" または "simps")

    Returns:
        band_intensity_ratio (np.ndarray): 指定周波数帯域の強度の時間変化 (0〜36Hz の積分値で正規化)
    """
    # 全帯域 (0〜36Hz) のインデックス範囲を取得
    total_freq_band = (0, 36)
    total_start, total_end = total_freq_band
    if method == "STFT":
        total_indices = np.where((frequency_array >= total_start) & (frequency_array <= total_end))[0]
        total_values = frequency_array[total_indices]
    elif method == "CWT":
        total_indices = range(total_start, total_end + 1)
        total_values = np.arange(total_start, total_end + 1)  # 仮の周波数配列
    else:
        raise ValueError("Invalid method. Use 'STFT' or 'CWT'.")

    # 指定した周波数帯域のインデックス範囲を取得
    freq_start, freq_end = freq_band
    if method == "STFT":
        freq_indices = np.where((frequency_array >= freq_start) & (frequency_array <= freq_end))[0]
        freq_values = frequency_array[freq_indices]
    elif method == "CWT":
        freq_indices = range(freq_start, freq_end + 1)
        freq_values = np.arange(freq_start, freq_end + 1)  # 仮の周波数配列
    else:
        raise ValueError("Invalid method. Use 'STFT' or 'CWT'.")

    # 全帯域 (0〜36Hz) の解析データと指定帯域の解析データを抽出
    total_matrix = analysis_matrix[total_indices, :]
    selected_matrix = analysis_matrix[freq_indices, :]

    # 積分の実行
    if integration_method == "trapz":
        total_intensity = trapezoid(total_matrix, x=total_values, axis=0)
        band_intensity = trapezoid(selected_matrix, x=freq_values, axis=0)
    elif integration_method == "simps":
        total_intensity = simpson(total_matrix, x=total_values, axis=0)
        band_intensity = simpson(selected_matrix, x=freq_values, axis=0)
    else:
        raise ValueError("Invalid integration method. Use 'trapz' or 'simps'.")

    # 比率を計算
    band_intensity_ratio = band_intensity / total_intensity
    smoothed_band_intensity_ratio = np.convolve(band_intensity_ratio, np.ones(window_size) / window_size, mode='valid')
    smoothed_band_intensity = np.convolve(band_intensity_ratio, np.ones(window_size) / window_size, mode='valid')

    return smoothed_band_intensity_ratio, band_intensity_ratio, band_intensity, smoothed_band_intensity


def plot_frequency_band_intensity(
        time_array, band_intensity, fontsize=12):
    """
    特定の周波数帯の強度変化をプロットする関数

    Parameters:
        time_array (np.ndarray): 時間配列
        band_intensity (np.ndarray): 指定周波数帯域の強度の時間変化
        freq_band (tuple): 表示する周波数帯 (例: (10, 20))
        fontsize (int): プロットのフォントサイズ
    """
    trimmed_time_array = time_array[:len(band_intensity)]
    plt.figure(figsize=(12, 6))
    plt.plot(trimmed_time_array, band_intensity)
    plt.xlabel("Time [sec]", fontsize=fontsize)
    plt.ylabel("Integrated Power", fontsize=fontsize)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, trimmed_time_array[len(trimmed_time_array) - 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


def save_all_data_to_csv(
        time_array, frequency_array, analysis_matrix, output_path_normalized, output_path_raw, method="STFT", integration_method="simps"):
    """
    時間・周波数帯の解析データをCSVファイルに保存する関数 (pandasを使用)

    Parameters:
        time_array (np.ndarray): 時間配列
        frequency_array (np.ndarray): 周波数配列
        analysis_matrix (np.ndarray): STFTまたはCWTの解析結果
        output_path_normalized (str): 正規化されたデータの保存先CSVファイルパス
        output_path_raw (str): RAWデータの保存先CSVファイルパス
        method (str): 解析手法 ("STFT" または "CWT")
        integration_method (str): 数値積分の手法 ("trapz" または "simps")
    """
    # 各帯域の強度を計算
    _, _, gamma_raw, gamma_smothed = calculate_frequency_band_intensity(frequency_array, analysis_matrix, (30, 36), method=method, integration_method=integration_method)
    _, _, beta_raw, beta_smothed = calculate_frequency_band_intensity(frequency_array, analysis_matrix, (15, 30), method=method, integration_method=integration_method)
    _, _, alpha_raw, alpha_smothed = calculate_frequency_band_intensity(frequency_array, analysis_matrix, (8, 12), method=method, integration_method=integration_method)
    _, _, theta_raw, theta_smothed = calculate_frequency_band_intensity(frequency_array, analysis_matrix, (4, 8), method=method, integration_method=integration_method)
    _, _, delta_raw, delta_smothed = calculate_frequency_band_intensity(frequency_array, analysis_matrix, (0, 4), method=method, integration_method=integration_method)

    # 全体の積分値を計算
    total_intensity = gamma_smothed + beta_smothed + alpha_smothed + theta_smothed + delta_smothed

    # 各帯域を正規化
    gamma_normalized = gamma_smothed / total_intensity
    beta_normalized = beta_smothed / total_intensity
    alpha_normalized = alpha_smothed / total_intensity
    theta_normalized = theta_smothed / total_intensity
    delta_normalized = delta_smothed / total_intensity

    # pandas DataFrameにまとめる
    trimmed_time_array = time_array[:len(gamma_smothed)]
    data1 = {
        "Time [sec]": trimmed_time_array,
        "Gamma Normalized [30-36 Hz]": gamma_normalized,
        "Beta Normalized [15-30 Hz]": beta_normalized,
        "Alpha Normalized [8-12 Hz]": alpha_normalized,
        "Theta Normalized [4-8 Hz]": theta_normalized,
        "Delta Normalized [0-4 Hz]": delta_normalized,
    }
    df1 = pd.DataFrame(data1)

    # CSVファイルに書き込み
    df1.to_csv(output_path_normalized, index=False)

    # pandas DataFrameにまとめる
    data2 = {
        "Time [sec]": time_array,
        "Gamma Raw [30-36 Hz]": gamma_raw,
        "Beta Raw [15-30 Hz]": beta_raw,
        "Alpha Raw 8-12 Hz]": alpha_raw,
        "Theta Raw [4-8 Hz]": theta_raw,
        "Delta Raw [0-4 Hz]": delta_raw
    }
    df2 = pd.DataFrame(data2)

    # CSVファイルに書き込み
    df2.to_csv(output_path_raw, index=False)


def save_stft_to_csv(frequencies, times, amplitude, output_path):
    """
    STFTの結果(frequencies, times, amplitude)をCSVに書き出す関数

    Parameters:
        frequencies (np.ndarray): 周波数データ
        times (np.ndarray): 時間データ
        amplitude (np.ndarray): 振幅データ
        output_path (str): 保存先のCSVファイルパス
    """
    # 周波数データを列名として使用
    column_names = [f"Freq_{freq:.2f}Hz" for freq in frequencies]

    # 振幅データをDataFrameに変換
    df_amplitude = pd.DataFrame(amplitude.T, columns=column_names)
    df_amplitude.insert(0, "Time [s]", times)  # 時間データを先頭列に挿入

    # CSVファイルに書き出し
    df_amplitude.to_csv(output_path, index=False)
    print(f"STFT結果をCSVファイルとして保存しました: {output_path}")


def save_cwt_to_csv(time_array, frequency_array, cwt_matrix, output_path):
    """
    CWTの結果(time_array, frequency_array, cwt_matrix)をCSVに書き出す関数

    Parameters:
        time_array (np.ndarray): 時間配列
        frequency_array (np.ndarray): 周波数配列
        cwt_matrix (np.ndarray): CWTの振幅データ (時間×周波数)
        output_path (str): 保存先のCSVファイルパス
    """
    # 周波数データを列名として使用
    column_names = [f"Freq_{freq:.2f}Hz" for freq in frequency_array]

    # 振幅データをDataFrameに変換
    df_cwt = pd.DataFrame(cwt_matrix.T, columns=column_names)
    df_cwt.insert(0, "Time [s]", time_array)  # 時間データを先頭列に挿入

    # CSVファイルに書き出し
    df_cwt.to_csv(output_path, index=False)
    print(f"CWT結果をCSVファイルとして保存しました: {output_path}")


def export_arguments_to_yaml(
        uploaded_file, analysis_method, sample_rate, max_frequency,
        signal_column_name, start_time, end_time, filter_type,
        highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation,
        band_intensity_setting, integration_method, segment_length, overlap,
        output_path="arguments.yaml"):
    """
    generate_spectrogram_and_signal_plotの引数をわかりやすい形式でYAMLに書き出す関数

    Parameters:
        各引数: generate_spectrogram_and_signal_plotの引数
        output_path (str): 書き出すYAMLファイルのパス
    """
    arguments = {
        "input_file": uploaded_file,  # アップロードされたファイル
        "analysis": {
            "method": analysis_method,  # 使用する解析手法
            "sample_rate_hz": sample_rate,  # サンプリングレート
            "max_frequency_hz": max_frequency,  # 最大周波数
            "time_range_sec": {
                "start": start_time,  # 開始時間
                "end": end_time       # 終了時間
            }
        },
        "signal_processing": {
            "signal_column": signal_column_name,  # 信号データが格納された列名
            "filter": {
                "type": filter_type,  # フィルタタイプ (High Pass, Low Pass など)
                "highpass_cutoff_hz": highpass_cutoff,  # ハイパスフィルタのカットオフ周波数
                "stopband_cutoff_hz": stopband_cutoff,  # ストップバンドのカットオフ周波数
                "passband_ripple_db": passband_ripple,  # パスバンドのリップル
                "stopband_attenuation_db": stopband_attenuation  # ストップバンドの減衰量
            }
        },
        "frequency_analysis": {
            "band_of_interest": band_intensity_setting,  # 強度を解析する周波数帯 (例: Gamma, Beta など)
            "integration_method": integration_method,  # 数値積分の手法 (Simpson, Trapezoidal など)
        },
        "stft_settings": {
            "segment_length_samples": segment_length,  # セグメント長 (サンプル数)
            "overlap_ratio": overlap  # セグメントのオーバーラップ率
        },
        "all_band_intensity_analysis": {
            "smoothed_window_size": 10
        }
    }

    # YAMLファイルに書き出し
    with open(output_path, "w") as yaml_file:
        yaml.dump(arguments, yaml_file, default_flow_style=False, allow_unicode=True)


def zip_directory_with_command(directory_path, output_zip_path):
    """
    zipコマンドを使ってディレクトリを圧縮する関数。

    Args:
        directory_path (str): 圧縮するディレクトリのパス。
        output_zip_path (str): 出力するzipファイルのパス。
    """

    if os.name == 'nt':
        return None
    try:
        # zipコマンドを実行
        subprocess.run(['zip', '-j', '-r', output_zip_path, directory_path], check=True)
        print(f"{output_zip_path} に圧縮しました。")
    except FileNotFoundError:
        print("zipコマンドが見つかりません。インストールされているか確認してください。")
    except subprocess.CalledProcessError as e:
        print(f"zipコマンドの実行に失敗しました: {e}")

    return output_zip_path


# UI処理: スペクトログラム生成・信号プロット
def generate_spectrogram_and_signal_plot(
        uploaded_file, analysis_method,
        sample_rate, max_frequency, signal_column_name, start_time, end_time,
        filter_type, highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation,
        band_intensity_setting, integration_method, segment_length, overlap, fontsize):
    """
    入力パラメータに基づいてスペクトログラムと信号プロットを生成し、結果をファイルに保存する関数。

    Parameters:
        uploaded_file (File): 信号データを含むアップロードされたファイルオブジェクト。
        analysis_method (str): 使用する解析方法 ("Short-Time Fourier Transform" または "Continuous Wavelet Transform")。
        sample_rate (int): 信号のサンプリングレート (Hz)。
        max_frequency (int): 解析する最大周波数 (Hz)。
        signal_column_name (str): アップロードされたファイル内の信号データが含まれる列の名前。
        start_time (float): 解析を開始する時刻 (秒)。
        end_time (float): 解析を終了する時刻 (秒)。
        filter_type (str): 適用するフィルターの種類 ("High PASS" または "Low PASS")。
        highpass_cutoff (float): ハイパスフィルターのカットオフ周波数 (Hz)。
        stopband_cutoff (float): フィルターのストップバンドカットオフ周波数 (Hz)。
        passband_ripple (float): パスバンドで許容される最大リップル (dB)。
        stopband_attenuation (float): ストップバンドでの最小減衰量 (dB)。
        band_intensity_setting (str): 解析対象の周波数帯域 (例: "Gamma", "Beta", "Alpha")。
        integration_method (str): 使用する数値積分法 ("simps" または "trapz")。
        segment_length (int): STFT用のセグメント長 (サンプル数)。
        overlap (float): STFTセグメント間のオーバーラップ率 (0.0 ～ 1.0)。
        fontsize (int): 生成するプロットのフォントサイズ。

    Returns:
        tuple: 以下のパスを含むタプル:
            - str: スペクトログラムプロットの保存先パス。
            - str: 周波数帯域強度プロットの保存先パス。
            - str: 信号プロットの保存先パス。
            - str: すべての結果を含むZIPファイルの保存先パス。

    Raises:
        ValueError: アップロードされたファイルが有効な信号データを含んでいない場合。
        FileNotFoundError: 必要なファイルまたはディレクトリにアクセスできない場合。
        Exception: 処理やプロットの保存中に問題が発生した場合。

    Notes:
        - フィルタリング、CWTまたはSTFT解析を実行し、ユーザー入力に基づいてプロットを生成します。
        - 中間および最終的な出力は一時ディレクトリに保存され、ダウンロード用にZIP化されます。
        - 信号処理には `labutils` および `filter` モジュールを使用します。

    Example:
        ```python
        # 使用例
        spectrogram_path, band_intensity_path, signal_plot_path, zip_file_path = generate_spectrogram_and_signal_plot(
            uploaded_file=my_file,
            analysis_method="Short-Time Fourier Transform",
            sample_rate=1000,
            max_frequency=50,
            signal_column_name="EEG Signal",
            start_time=0,
            end_time=10,
            filter_type="High PASS",
            highpass_cutoff=0.5,
            stopband_cutoff=0.1,
            passband_ripple=0.5,
            stopband_attenuation=40,
            band_intensity_setting="Alpha",
            integration_method="simps",
            segment_length=256,
            overlap=0.5,
            fontsize=14
        )
        print(f"スペクトログラムの保存先: {spectrogram_path}")
        print(f"周波数帯域強度プロットの保存先: {band_intensity_path}")
        print(f"信号プロットの保存先: {signal_plot_path}")
        print(f"すべての結果が保存されたZIPファイル: {zip_file_path}")
        ```
    """

    file_path = uploaded_file.name
    file_basename = os.path.basename(file_path)

    signal_data = labutils.load_signal(file_path, signal_column_name)
    if len(signal_data) == 0:
        return None, None

    output_dir = os.path.join("/tmp/spectrogram/", str(uuid.uuid1())[0:20].replace("-", ""))
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
    plt.figure(dpi=200)
    plot_signal(signal_data, time_array, start_time, end_time, fontsize, window_size)
    signal_plot_path = os.path.join(output_dir, "signal_plot.png")
    plt.savefig(signal_plot_path)

    freq_start, freq_end = labutils.band_intensity_setting_to_band(band_intensity_setting)
    integration_method = labutils.integration_method_to_method(integration_method)
    overlap = float(overlap) / 100.0

    # スペクトログラムプロットの保存
    if analysis_method == "Short-Time Fourier Transform":
        frequencies, times, amplitude = perform_stft(signal_data, sample_rate, segment_length, overlap)

        plt.figure(dpi=200)
        plot_stft_spectrogram(amplitude, frequencies, times, max_frequency, fontsize)
        spectrogram_plot_path = os.path.join(output_dir, "stft_spectrogram_plot.png")
        plt.savefig(spectrogram_plot_path)

        plt.figure(dpi=200)
        band_intensity, _, _, _ = calculate_frequency_band_intensity(frequencies, amplitude, (freq_start, freq_end), method="STFT", integration_method=integration_method)
        plot_frequency_band_intensity(times, band_intensity, fontsize=fontsize)
        plot_frequency_band_intensity_path = os.path.join(output_dir, "band_intensity.png")
        plt.savefig(plot_frequency_band_intensity_path)

        csv_path_normalized = os.path.join(output_dir, "band_intensity_data_normalized.csv")
        csv_path_raw = os.path.join(output_dir, "band_intensity_data_raw.csv")
        save_all_data_to_csv(times, frequencies, amplitude, csv_path_normalized, csv_path_raw, method="STFT", integration_method=integration_method)
        spectrogram_raw_data_path = os.path.join(output_dir, "spectrogram_raw_data.csv")
        save_stft_to_csv(frequencies, times, amplitude, spectrogram_raw_data_path)

    else:
        spectrogram_plot_path = os.path.join(output_dir, "wavelet_spectrogram_plot.png")
        cwt_matrix = perform_cwt(sample_rate, signal_data, max_frequency)
        plt.figure(dpi=200)
        plot_cwt_result(cwt_matrix, time_array, max_frequency, fontsize)
        plt.savefig(spectrogram_plot_path)

        plt.figure(dpi=200)
        band_intensity, _, _, _ = calculate_frequency_band_intensity(np.arange(max_frequency), cwt_matrix, (freq_start, freq_end), method="CWT", integration_method=integration_method)
        plot_frequency_band_intensity(time_array, band_intensity, fontsize=fontsize)
        plot_frequency_band_intensity_path = os.path.join(output_dir, "band_intensity.png")
        plt.savefig(plot_frequency_band_intensity_path)

        csv_path_normalized = os.path.join(output_dir, "band_intensity_data_normalized.csv")
        csv_path_raw = os.path.join(output_dir, "band_intensity_data_raw.csv")
        save_all_data_to_csv(time_array, np.arange(max_frequency), cwt_matrix, csv_path_normalized, csv_path_raw, method="CWT", integration_method=integration_method)

        spectrogram_raw_data_path = os.path.join(output_dir, "spectrogram_raw_data.csv")
        save_cwt_to_csv(time_array, np.arange(max_frequency), cwt_matrix, spectrogram_raw_data_path)

    config_yaml_path = os.path.join(output_dir, "lab_tool_spectrogram.yaml")

    export_arguments_to_yaml(
        file_basename, analysis_method,
        sample_rate, max_frequency, signal_column_name, start_time, end_time,
        filter_type, highpass_cutoff, stopband_cutoff, passband_ripple, stopband_attenuation,
        band_intensity_setting, integration_method, segment_length, overlap,
        config_yaml_path
    )

    if os.path.exists('/tmp/all_analyze_file.zip'):
        os.remove('/tmp/all_analyze_file.zip')
    ziped_file = zip_directory_with_command(output_dir, "/tmp/all_analyze_file.zip")

    return spectrogram_plot_path, plot_frequency_band_intensity_path, signal_plot_path, ziped_file
