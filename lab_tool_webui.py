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
    plt.imshow(cwt_result, cmap='jet', aspect='auto', vmax=abs(cwt_result).max(), vmin=-abs(cwt_result).max())
    plt.xlabel("Time [min]")
    plt.ylabel("Frequency [Hz]")
    plt.axis([0, len(time_data) / 1000, 0, fmax - 1])
    plt.xticks(np.arange(0, len(time_data), step=60000))
    plt.ticklabel_format(style='plain', axis='x')
    plt.colorbar()
    plt.clim(-5, 5)

# グラフ描画とCWTの処理を行う関数
def wavelet_ui(uploaded_file, Fs, fmax, column_name):
    filepath = uploaded_file.name
    signal = load_signal(filepath, column_name)
    if len(signal) == 0:
        return None, None
    
    t_data = np.arange(0, len(signal) / Fs, 1 / Fs)
    signal_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    plt.figure(dpi=200)
    plt.title("Signal")
    plt.plot(t_data, signal)
    plt.xlim(0, t_data[-1])
    plt.xticks(np.arange(0, t_data[-1] + 1, step=100))
    plt.xlabel("Time [s]")
    plt.savefig(signal_filename)

    cwt_signal_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    cwt_signal = continuous_wavelet_transform(Fs=Fs, data=signal, fmax=fmax)
    plt.figure(dpi=200)
    plot_cwt(cwt_signal, t_data, fmax)
    plt.savefig(cwt_signal_filename)

    return cwt_signal_filename, signal_filename

# Gradio UIの設定
with gr.Blocks() as main_ui:
    with gr.Tab("Wavelet"):
        with gr.Row():
            # 左側に入力要素を配置
            with gr.Column():
                file_input = gr.File(label="CSVファイルをアップロードしてください。", file_count="single", file_types=["csv"])
                fs_slider = gr.Slider(minimum=0, maximum=10000, value=1000, label="サンプリング周波数", step=10, info="単位はHz。")
                fmax_slider = gr.Slider(minimum=0, maximum=200, value=60, label="wavelet 最大周波数", step=10, info="単位はHz。")
                column_dropdown = gr.Dropdown(["Fp1", "Fp2", "T7", "T8", "O1", "O2"], value="Fp2", label="使用する信号データ", allow_custom_value=True, info="使用する信号データを選んでください。デフォルトはFp2です。")
                submit_button = gr.Button("計算開始")

            # 右側に出力要素を配置
            with gr.Column():
                wavelet_image = gr.Image(type="filepath", label="Wavelet")
                signal_image = gr.Image(type="filepath", label="Signal")

        # ボタンのクリックで処理を実行
        submit_button.click(wavelet_ui, inputs=[file_input, fs_slider, fmax_slider, column_dropdown], outputs=[wavelet_image, signal_image])

if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0")