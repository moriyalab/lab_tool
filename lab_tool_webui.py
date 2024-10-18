import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import gradio as gr
import tempfile


# CSVファイルから信号データを読み込む
def load_signal(file_path, column_name):
    try:
        df = pd.read_csv(file_path)
        signal = df[column_name].values
        return signal
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return []
    except KeyError as e:
        print(f"Column '{column_name}' not found in the file. ({e})", file=sys.stderr)
        return []


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
    signal = load_signal(filepath, column_name)

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


def update_slider_range(filepath):
    timestamp = load_signal(filepath, "Timestamp")
    max_value = float(timestamp[len(timestamp) - 1])
    min_value = float(timestamp[0])

    return gr.update(minimum=min_value, maximum=max_value), gr.update(minimum=min_value, maximum=max_value, value=max_value)


with gr.Blocks() as main_ui:
    with gr.Tab("Wavelet"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="CSVファイルをアップロードしてください。", file_count="single", file_types=["csv"])
                fs_slider = gr.Slider(minimum=0, maximum=10000, value=1000, label="サンプリング周波数", step=10, info="単位はHz。")
                fmax_slider = gr.Slider(minimum=0, maximum=200, value=60, label="wavelet 最大周波数", step=10, info="単位はHz。")
                column_dropdown = gr.Dropdown(["Fp1", "Fp2", "T7", "T8", "O1", "O2"], value="Fp2", label="使用する信号データ", allow_custom_value=True, info="使用する信号データを選んでください。デフォルトはFp2です。")
                start_time = gr.Slider(minimum=0, maximum=60, value=0.0, step=0.5, label="Start Time (sec)")
                end_time = gr.Slider(minimum=0, maximum=60, value=60.0, step=0.5, label="End Time (sec)")
                submit_button = gr.Button("計算開始")

                file_input.change(
                    update_slider_range,
                    inputs=file_input,
                    outputs=[start_time, end_time]
                )

            with gr.Column():
                wavelet_image = gr.Image(type="filepath", label="Wavelet")
                signal_image = gr.Image(type="filepath", label="Signal")

        submit_button.click(wavelet_ui, inputs=[file_input, fs_slider, fmax_slider, column_dropdown, start_time, end_time], outputs=[wavelet_image, signal_image])
if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0")
