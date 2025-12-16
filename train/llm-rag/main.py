import ffmpeg
import os

def extract_audio_with_ffmpeg_python(input_video_path, output_audio_path):
    # Ensure directory exists for audio output
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # Check for audio streams
    try:
        probe = ffmpeg.probe(input_video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        if not audio_streams:
            print(f"Warning: No audio stream found in {input_video_path}. Skipping audio extraction.")
            return

    except ffmpeg.Error as e:
        print(f"Error probing file: {e}")
        return

    try:
        (
            ffmpeg
            .input(input_video_path)
            .output(
                output_audio_path,
                **{'vn': None, 'acodec': 'pcm_s16le', 'ar': '16000', 'ac': '1'}
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("สำเร็จ! ไฟล์เสียงถูกสร้างที่:", output_audio_path)

    except ffmpeg.Error as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        if e.stderr:
            print("FFmpeg Error Output:\n", e.stderr.decode('utf8'))


def extract_frames_with_ffmpeg_python(input_video_path, output_pattern):
    output_dir = os.path.dirname(output_pattern)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        (
            ffmpeg
            .input(input_video_path)
            .filter('fps', fps=1/5)
            .output(output_pattern)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("สำเร็จ! ภาพถูกสร้างในโฟลเดอร์:", output_pattern.split('/')[0])

    except ffmpeg.Error as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        if e.stderr:
            print("FFmpeg Error Output:\n", e.stderr.decode('utf8'))


idx = 1
file_input = f"downloads/{idx}.mp4"
audio_input = f"downloads/{idx}.m4a"
file_output = f"output/{idx}-out.mp4"
# extract_frames_with_ffmpeg_python(file_input, f"output/{idx}/frames/frame_%03d.jpg")
# extract_audio_with_ffmpeg_python(audio_input, f"output/{idx}/voice-{idx}.wav")

