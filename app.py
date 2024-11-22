import os
if os.name == 'nt':
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from moviepy.editor import VideoFileClip
from lab_tools import spectrogram
from lab_tools import labutils
from lab_tools import analyze1f
from lab_tools import ytutil


def update_slider_range(filepath):
    timestamp = labutils.load_signal(filepath, "Timestamp")
    max_value = float(timestamp[len(timestamp) - 1])
    min_value = float(timestamp[0])

    return gr.update(minimum=min_value, maximum=max_value), gr.update(minimum=min_value, maximum=max_value, value=max_value)


def update_anlyze_setting(mode):
    if mode == "YouTube":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None), gr.update(visible=True), gr.update(value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None), gr.update(visible=False), gr.update(value=None)


def update_slidar_range_youtube(video_url):
    if video_url == "":
        return gr.update(maximum=0, value=0), gr.update(maximum=0, value=0), gr.update(value=None), gr.update(value=None), gr.update(value=None)

    video_info = ytutil.get_video_info(video_url)
    duration = int(video_info["duration"]) - 1
    title = video_info["title"]

    ytutil.remove_mp4_file()

    return gr.update(maximum=duration), gr.update(maximum=duration, value=duration), gr.update(value=title), gr.update(value=None), gr.update(value=None)


def update_slidar_range_video_file(file_path):
    if file_path == "":
        return gr.update(maximum=0), gr.update(maximum=0, value=0), gr.update(value=None), gr.update(value=None)

    with VideoFileClip(file_path) as video:
        duration = video.duration

    return gr.update(maximum=duration), gr.update(maximum=duration, value=duration), gr.update(value=None), gr.update(value=None)


with gr.Blocks() as main_ui:
    with gr.Tab("Spectrogram analyze"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="CSVファイルをアップロードしてください。", file_count="single", file_types=["csv"])
                analysis_method = gr.Radio(
                    ["Short-Time Fourier Transform", "Wavelet"],
                    label="Analysis method",
                    value="Short-Time Fourier Transform",
                )
                fs_slider = gr.Slider(minimum=0, maximum=10000, value=1000, label="サンプリング周波数", step=10, info="単位はHz。")
                fmax_slider = gr.Slider(minimum=0, maximum=200, value=60, label="wavelet 最大周波数", step=10, info="単位はHz。")
                column_dropdown = gr.Dropdown(["Fp1", "Fp2", "T7", "T8", "O1", "O2"], value="Fp2", label="使用する信号データ", allow_custom_value=True, info="使用する信号データを選んでください。デフォルトはFp2です。")
                start_time = gr.Slider(minimum=0, maximum=60, value=0.0, step=0.5, label="Start Time (sec)")
                end_time = gr.Slider(minimum=0, maximum=60, value=60.0, step=0.5, label="End Time (sec)")
                filter_setting = gr.Radio(
                    ["No Filter", "High PASS", "Low PASS"],
                    label="フィルター設定",
                    value="High PASS",
                )
                fp_hp = gr.Slider(minimum=0, maximum=20, value=3, step=0.1, label="通過域端周波数 [Hz]")
                fs_hp = gr.Slider(minimum=0, maximum=20, value=1, step=0.1, label="阻止域端周波数 [Hz]")
                gpass = gr.Slider(minimum=0, maximum=100, value=3, step=1, label="通過域端最大損失 [dB]")
                gstop = gr.Slider(minimum=0, maximum=100, value=40, step=1, label="阻止域端最小損失 [dB]")

                submit_button = gr.Button("計算開始")

                file_input.change(
                    update_slider_range,
                    inputs=file_input,
                    outputs=[start_time, end_time]
                )

            with gr.Column():
                wavelet_image = gr.Image(type="filepath", label="Spectrogram")
                signal_image = gr.Image(type="filepath", label="Signal")

        submit_button.click(spectrogram.generate_spectrogram_and_signal_plot, inputs=[
            file_input, analysis_method,
            fs_slider, fmax_slider, column_dropdown, start_time, end_time,
            filter_setting, fp_hp, fs_hp, gpass, gstop],
            outputs=[wavelet_image, signal_image])

    with gr.Tab("1f noise analyze"):
        with gr.Row():
            with gr.Column():
                mode_setting = gr.Radio(
                    ["YouTube", "動画ファイル"],
                    label="解析データ",
                    value="YouTube",
                )
                youtube_url_input = gr.Text(label="YouTubeのリンクを貼り付けてください。")
                file_input = gr.File(label="動画をアップロードしてください", visible=False, file_count="single", file_types=["mp4"])
                start_time = gr.Slider(minimum=0, maximum=3600, value=0, step=1, label="Start Time (sec)")
                end_time = gr.Slider(minimum=0, maximum=3600, value=10, step=1, label="End Time (sec)")
                submit_button = gr.Button("計算開始")
                download_button = gr.Button("ダウンロード")

            with gr.Column():
                caption = gr.Text(label="動画タイトル")
                result = gr.Image(type="filepath", label="Wavelet")
                file_result = gr.File(label="Downloaded Video")

        submit_button.click(analyze1f.analyze_1f_noise, inputs=[mode_setting, youtube_url_input, file_input, start_time, end_time], outputs=[caption, result])
        download_button.click(ytutil.download_youtube_video, inputs=[youtube_url_input], outputs=[file_result])

        mode_setting.change(
            update_anlyze_setting,
            inputs=mode_setting,
            outputs=[youtube_url_input, file_input, download_button, result]
        )
        youtube_url_input.change(
            update_slidar_range_youtube,
            inputs=youtube_url_input,
            outputs=[start_time, end_time, caption, file_result, result]
            )
        file_input.change(
            update_slidar_range_video_file,
            inputs=file_input,
            outputs=[start_time, end_time, file_result, result]
        )

if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7860)
    ytutil.remove_mp4_file()
