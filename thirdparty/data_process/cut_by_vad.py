import os
import argparse
import soundfile as sf
import json
import numpy as np
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_audio_file(audio_file, input_dir, output_dir, segment_length, sample_rate):
    try:
        # 自动推断 VAD json 文件的路径
        vad_json_file = os.path.splitext(audio_file)[0] + '.json'

        if not os.path.exists(vad_json_file):
            tqdm.write(f"Warning: Corresponding VAD file not found for {audio_file}. Skipping.")
            return

        with open(vad_json_file, 'r') as f:
            vad_data = json.load(f)

        voice_activities = vad_data.get("voice_activity", [])
        if not voice_activities:
            tqdm.write(f"Warning: No voice activity found in {vad_json_file}. Skipping.")
            return

        audio_waveform, current_sr = sf.read(audio_file)
        if current_sr != sample_rate:
            tqdm.write(f"Warning: Resampling needed for {audio_file} (SR: {current_sr}), but not implemented. Skipping.")
            return

        all_voice_frames = [
            audio_waveform[int(start_sec * sample_rate):int(end_sec * sample_rate)]
            for start_sec, end_sec in voice_activities
        ]

        if not all_voice_frames:
            tqdm.write(f"Warning: No valid voice frames extracted from {audio_file}. Skipping.")
            return
        
        # 将所有有声片段拼接起来
        concatenated_voice_waveform = np.concatenate(all_voice_frames)
        total_voice_frames = len(concatenated_voice_waveform)

        # 切割为目标长度的片段
        target_frames_per_segment = segment_length * sample_rate
        segment_count = 0
        current_start_frame = 0
        relative_path = os.path.relpath(os.path.dirname(audio_file), input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        base_filename = os.path.splitext(os.path.basename(audio_file))[0]

        while current_start_frame < total_voice_frames:
            segment_end_frame = min(current_start_frame + target_frames_per_segment, total_voice_frames)
            # 如果最后一段太短，就丢弃它
            if (segment_end_frame - current_start_frame) < (0.5 * sample_rate): 
                break
            
            segment_waveform = concatenated_voice_waveform[current_start_frame:segment_end_frame]
            output_filename = os.path.join(current_output_dir, f"{base_filename}_{segment_count:04d}.flac")
            sf.write(output_filename, segment_waveform, sample_rate)
            
            current_start_frame = segment_end_frame
            segment_count += 1

    except Exception as e:
        tqdm.write(f"Error processing {audio_file}: {e}. Skipping.")

def cut_audio_by_vad_recursive(input_dir, output_dir, segment_length=60, sample_rate=16000, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 使用 glob 递归查找所有的 .flac 文件
    search_pattern = os.path.join(input_dir, '**', '*.flac')
    audio_files = glob.glob(search_pattern, recursive=True)

    if not audio_files:
        print(f"Warning: No .flac files found recursively in {input_dir}. Exiting.")
        return

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for audio_file in audio_files:
            futures.append(executor.submit(process_audio_file, audio_file, input_dir, output_dir, segment_length, sample_rate))

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files", unit="file"):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut long audio files into segments based on VAD. This version supports recursive search with multi-threading.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing raw FLAC audio and JSON VAD files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save segmented FLAC files.")
    parser.add_argument("--segment_length", type=int, default=60,
                        help="Target length of each audio segment in seconds.")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of the audio files (Hz).")
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Maximum number of threads to use for processing.")
    
    args = parser.parse_args()
    cut_audio_by_vad_recursive(args.input_dir, args.output_dir, args.segment_length, args.sample_rate, args.max_workers)
