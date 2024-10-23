import scipy.signal as signal


def apply_filter(signal_data, sample_rate, pass_freq, stop_freq, pass_gain, stop_gain, filter_type):
    nyquist_freq = sample_rate / 2                       # ナイキスト周波数
    normalized_pass_freq = pass_freq / nyquist_freq      # 通過域端周波数を正規化
    normalized_stop_freq = stop_freq / nyquist_freq      # 阻止域端周波数を正規化
    filter_order, cutoff_freq = signal.buttord(
        normalized_pass_freq, normalized_stop_freq, pass_gain, stop_gain
    )  # フィルタのオーダーと正規化周波数を計算
    b, a = signal.butter(filter_order, cutoff_freq, filter_type)  # フィルタの伝達関数を計算
    filtered_signal = signal.filtfilt(b, a, signal_data)          # 信号にフィルタを適用
    return filtered_signal


def lowpass(signal_data, sample_rate, pass_freq, stop_freq, pass_gain, stop_gain):
    return apply_filter(signal_data, sample_rate, pass_freq, stop_freq, pass_gain, stop_gain, "low")


def highpass(signal_data, sample_rate, pass_freq, stop_freq, pass_gain, stop_gain):
    return apply_filter(signal_data, sample_rate, pass_freq, stop_freq, pass_gain, stop_gain, "high")
