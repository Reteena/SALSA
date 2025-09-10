from collections import defaultdict
from pydub import AudioSegment
from ffmpeg_normalize import FFmpegNormalize
import os

def vad_merge(audio_paths):
    merged_audio = AudioSegment.from_file(audio_paths[0])
    try:
        for i in range(1, len(audio_paths)-1):
            audio = AudioSegment.from_file(audio_paths[i])
            merged_audio += audio

        return merged_audio

    except Exception as e:
        print(f"Loading error: {e}")

def conv_16khz_mono_normalize(audio_path, output_audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_audio_path, format="wav")
        print(f"converted {audio_path} to {output_audio_path} to 16kyhz mono")

        normalizer = FFmpegNormalize(target_level=-16.0)
        normalizer.add_media_file(output_audio_path, output_audio_path)
        normalizer.run_normalization()
        print(f"normalized {output_audio_path} to -16 dB")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")

def segment_audio(audio_path, output_dir, window_size_sec=10, overlap_frac=0.5):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)

        window_size_ms = window_size_sec * 1000
        step_size_ms = int(window_size_ms * (1-overlap_frac))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        segments = []
        for i, start_ms in enumerate(range(0, duration_ms, step_size_ms)):
            segment = audio[start_ms:start_ms + window_size_ms]
            segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
            segment.export(segment_path, format="wav")
            segments.append(segment_path)
            print(f"exported segment: {segment_path}")

        return segments

    except Exception as e:
        print(f"Loading error: {e}")

train_dir = "./dataset/ADReSS-IS2020-data/test/Normalised_audio-chunks"
grouped_files = defaultdict(list)

for fname in os.listdir(train_dir):
    prefix = fname[0:4]
    full_path = os.path.join(train_dir, fname)
    grouped_files[prefix].append(full_path)

output_root_dir = "./dataset/ADReSS-IS2020-data/test/full_preprocessing/preprocessed_chunks"
os.makedirs(output_root_dir, exist_ok=True)

for prefix, files in grouped_files.items():
    print(f"Processing group: {prefix} with {len(files)} files")
    output_dir = os.path.join(output_root_dir, f"{prefix}_processed")
    os.makedirs(output_dir, exist_ok=True)

    merged = vad_merge(files)
    if merged is None:
        print(f"Skipping group {prefix} due to merge error")
        continue
    
    merged_path = os.path.join(output_dir, f"{prefix}_merged.wav")
    merged.export(merged_path, format="wav")
    print(f"Merged audio saved to {merged_path}")

    norm_path = os.path.join(output_dir, f"{prefix}_normalized.wav")
    conv_16khz_mono_normalize(merged_path, norm_path)

    segments = segment_audio(norm_path, output_dir)
    print(f"Segmented audio into {len(segments)} segments in {output_dir}")

    # if beginning part of filename are same, append to array, otherwise run functions above