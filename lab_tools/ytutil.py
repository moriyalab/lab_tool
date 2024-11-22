import yt_dlp
import os
import glob


def sanitize_filename(filename: str, max_length: int = 20) -> str:
    """
    ファイル名を安全な形式に変換し、長さを制限する
    """
    # 日本語をASCIIに置き換える（必要に応じて削除）
    sanitized = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    return sanitized[:max_length]


def download_youtube(youtube_url: str) -> str:
    video_info = get_video_info(youtube_url)
    title = video_info["title"]
    title = sanitize_filename(title)

    ydl_opts = {
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }
        ],
        'format': 'bestaudio+worstvideo',
        'outtmpl': title
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        file_path = ydl.prepare_filename(info_dict)
        filename, _ = os.path.splitext(file_path)
        filename += ".mp3"
        print(f"Downloaded file path: {filename}")

    return filename


def download_youtube_video(youtube_url: str) -> str:
    video_info = get_video_info(youtube_url)
    title = video_info["title"]
    title = sanitize_filename(title)

    ydl_opts = {
        'format': 'bestaudio+worstvideo',
        'outtmpl': title
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        file_path = ydl.prepare_filename(info_dict)
        print(f"Downloaded file path: {file_path}")

    return file_path


def remove_mp4_file():
    print("remove_mp4_file")
    mp4_files = glob.glob(os.path.join("./", '*.mp4'))

    for file in mp4_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"削除失敗: {file} - {e}")


def get_video_info(url):
    ydl_opts = {
        'dump_single_json': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info
