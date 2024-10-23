import sys
import pandas as pd


# CSVファイルから信号データを読み込む
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
