import argparse
import os
from pathlib import Path
from transformers import pipeline

parser = argparse.ArgumentParser(description='Script to transcribe a custom audio file of any length using Whisper Models of various sizes.')
parser.add_argument(
    "--is_public_repo",
    required=False,
    default=True, 
    type=lambda x: (str(x).lower() == 'true'),
    help="If the model is available for download on huggingface.",
)
parser.add_argument(
    "--hf_model",
    type=str,
    required=False,
    default="openai/whisper-tiny",
    help="Huggingface model name. Example: openai/whisper-tiny",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=False,
    default=".",
    help="Folder with the pytorch_model.bin file",
)
parser.add_argument(
    "--temp_ckpt_folder",
    type=str,
    required=False,
    default="temp_dir",
    help="Path to create a temporary folder containing the model and related files needed for inference",
)
parser.add_argument(
    "--audio_dir",
    type=str,
    required=False,
    default="./t5/audio_v",
    help="Path to the directory containing audio files to be transcribed.",
)
parser.add_argument(
    "--language",
    type=str,
    required=False,
    default="en",
    help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
)
parser.add_argument(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
)

args = parser.parse_args()


model_id="./temp_dir_1"
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=5,
    device=args.device,
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language=args.language, task="transcribe")

# Get list of audio files in the directory
audio_files = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

# Transcribe each audio file
for audio_file in audio_files:
    print(f'Transcribing {audio_file}:')
    result = transcribe(audio_file)
    print(result["text"])
    print('-' * 40)

if args.is_public_repo == False:
    os.system(f"rm -r {args.temp_ckpt_folder}")