import os
import argparse
import soundfile as sf
import json
import numpy as np
from tqdm import tqdm

def cut_audio_by_vad(input_dir, output_dir, segment_length=60, sample_rate=16000):
    # 根据VAD信息将长音频切割成指定长度的短片段
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    target_frames_per_segment = segment_length * sample_rate
    # 遍历 input_dir 下的所有子目录
    for speaker_id in tqdm(os.listdir(input_dir), desc=f"Processing speakers in {input_dir}"):
        speaker_path = os.path.join(input_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for book_name in os.listdir(speaker_path):
            book_path = os.path.join(speaker_path, book_name)
            if not os.path.isdir(book_path):
                continue
            output_book_path = os.path.join(output_dir, speaker_id, book_name)
            os.makedirs(output_book_path, exist_ok=True)

            audio_file = None
            vad_json_file = None

            # 寻找 .flac 和 .json 文件
            for f in os.listdir(book_path):
                if f.endswith(".flac"):
                    audio_file = os.path.join(book_path, f)
                elif f.endswith(".json"):
                    vad_json_file = os.path.join(book_path, f)
            
            if not audio_file or not vad_json_file:
                tqdm.write(f"Warning: Missing audio (.flac) or VAD (.json) file in {book_path}. Skipping.")
                continue

            try:
                with open(vad_json_file, 'r') as f:
                    vad_data = json.load(f)
                
                # VAD信息格式: {"voice_activity": [[start_sec, end_sec], ...]}
                # "snr": ...
                # "speaker": ...
                # "book_meta": ...
                voice_activities = vad_data.get("voice_activity", [])
                if not voice_activities:
                    tqdm.write(f"Warning: No voice activity found in {vad_json_file}. Skipping.")
                    continue

                # 读取整个音频文件
                audio_waveform, current_sr = sf.read(audio_file)
                if current_sr != sample_rate:
                    # 如果采样率不匹配，需要重新采样
                    tqdm.write(f"Warning: Audio file {audio_file} has sample rate {current_sr}, but expected {sample_rate}. Attempting to resample...")
                    tqdm.write("Resampling is not implemented in this script. Skipping file.")
                    continue

                all_voice_frames = []
                for start_sec, end_sec in voice_activities:
                    start_frame = int(start_sec * sample_rate)
                    end_frame = int(end_sec * sample_rate)
                    all_voice_frames.append(audio_waveform[start_frame:end_frame])
                
                if not all_voice_frames:
                    tqdm.write(f"Warning: No valid voice frames extracted from {audio_file}. Skipping.")
                    continue

                # 将所有有声片段拼接起来
                concatenated_voice_waveform = np.concatenate(all_voice_frames)
                total_voice_frames = len(concatenated_voice_waveform)
                # 切割为目标长度的片段
                segment_count = 0
                current_start_frame = 0

                while current_start_frame < total_voice_frames:
                    segment_end_frame = min(current_start_frame + target_frames_per_segment, total_voice_frames)
                    if (segment_end_frame - current_start_frame) < (target_frames_per_segment / 2) and segment_count > 0:
                        break # 太短，舍弃
                    segment_waveform = concatenated_voice_waveform[current_start_frame:segment_end_frame]
                    output_filename = os.path.join(output_book_path, 
                                                 f"{os.path.splitext(os.path.basename(audio_file))[0]}_{segment_count:04d}.flac")
                    sf.write(output_filename, segment_waveform, sample_rate)
                    current_start_frame = segment_end_frame
                    segment_count += 1

            except Exception as e:
                tqdm.write(f"Error processing {audio_file}: {e}. Skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut long audio files into segments based on VAD.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing raw FLAC audio and JSON VAD files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save segmented FLAC files.")
    parser.add_argument("--segment_length", type=int, default=60,
                        help="Target length of each audio segment in seconds.")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of the audio files (Hz).")

    args = parser.parse_args()
    cut_audio_by_vad(args.input_dir, args.output_dir, args.segment_length, args.sample_rate)