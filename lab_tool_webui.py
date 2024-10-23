import gradio as gr
from lab_tools import wavelet
from lab_tools import labutils
from lab_tools import analyze1f


def update_slider_range(filepath):
    timestamp = labutils.load_signal(filepath, "Timestamp")
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

        submit_button.click(wavelet.wavelet_ui, inputs=[file_input, fs_slider, fmax_slider, column_dropdown, start_time, end_time], outputs=[wavelet_image, signal_image])

    with gr.Tab("1f Noise Search"):
        with gr.Row():
            with gr.Column():
                file_input = gr.Text(label="YouTubeのリンクを貼り付けてください。")
                submit_button = gr.Button("計算開始")

            with gr.Column():
                caption = gr.Text(label="動画タイトル")
                result = gr.Image(type="filepath", label="Wavelet")

        submit_button.click(analyze1f.analyze_1f_noise, inputs=[file_input], outputs=[caption, result])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0")
