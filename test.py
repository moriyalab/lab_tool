import pandas as pd
import sys
import matplotlib.pyplot as plt
from lab_tools import highpass


def load_signal(file_path, column_name):
    try:
        with open(file_path, 'r') as file:
            # データ部分が始まる行を見つける
            for i, line in enumerate(file):
                if 'Timestamp' in line:
                    header_line = i
                    break

        # 見つけたヘッダー行からデータを読み込む
        df = pd.read_csv(file_path, skiprows=header_line)
        signal = df[column_name].values
        return signal
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return []
    except KeyError as e:
        print(f"Column '{column_name}' not found in the file. ({e})", file=sys.stderr)
        return []


samplerate = 1000
time_data = load_signal("./test2_143809.csv", "Timestamp")
signal_data = load_signal("./test2_143809.csv", "Fp1")
fp = 22                            # 通過域端周波数[Hz]※ベクトル
fs = 10                              # 阻止域端周波数[Hz]※ベクトル
gpass = 5                                               # 通過域端最大損失[dB]
gstop = 40                                              # 阻止域端最小損失[dB]
Fs = 4096                                               # フレームサイズ
overlap = 90

data_filt = highpass.highpass_filter(signal_data, samplerate, fp, fs, gpass, gstop)

t_array_org, N_ave_org = highpass.overlap_frames(signal_data, samplerate, Fs, overlap)
t_array_filt, N_ave_filt = highpass.overlap_frames(signal_data, samplerate, Fs, overlap)

t_array_org, acf_org = highpass.hanning(t_array_org, Fs, N_ave_org)
t_array_filt, acf_filt = highpass.hanning(t_array_filt, Fs, N_ave_filt)

fft_array_org, fft_mean_org, fft_axis_org = highpass.fft_ave(t_array_org, samplerate, Fs, N_ave_org, acf_org)
fft_array_filt, fft_mean_filt, fft_axis_filt = highpass.fft_ave(t_array_filt, samplerate, Fs, N_ave_filt, acf_filt)

fft_mean_org = highpass.linear_to_db(fft_mean_org, 2e-5)
fft_mean_filt = highpass.linear_to_db(fft_mean_filt, 2e-5)

# フォントの種類とサイズを設定する。
# plt.rcParams['font.size'] = 14
# plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(211)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax2 = fig.add_subplot(212)
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('V[μV]')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Amp[dB]')

# データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
ax1.plot(time_data, signal_data, label='original', lw=1)
ax1.plot(time_data, data_filt, label='filtered', lw=1)
ax2.plot(fft_axis_org, fft_mean_org, label='original', lw=1)
ax2.plot(fft_axis_filt, fft_mean_filt, label='filtered', lw=1)
plt.legend()

# 軸のリミットを設定する。
# ax1.set_xlim(0,1200)
# ax1.set_xticks(np.arange(0,1201,100))
# ax2.set_xlim(0, max(fft_axis_org)/2)
# ax2.set_xticks(np.arange(0,501,10))
# ax2.set_ylim(-50, 150)

# レイアウト設定
fig.tight_layout()

# グラフを表示する。
plt.savefig("./out.png")
