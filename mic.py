import argparse
from transformers import pipeline
import sounddevice as sd
import numpy as np

parser = argparse.ArgumentParser(description="Live transcription using Whisper models.")

parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Two-letter language code for transcription, e.g., 'en' for English.",
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Device to run the pipeline on. Use -1 for CPU, 0 for GPU.",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=16000,
    help="Sample rate for audio recording.",
)
parser.add_argument(
    "--chunk_duration",
    type=int,
    default=5,
    help="Duration of each audio chunk to transcribe in seconds.",
)
args = parser.parse_args()
model_id = "./temp_dir_1"
print("Loading model...")
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=args.chunk_duration,
    device=args.device,
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
    language=args.language, task="transcribe"
)

def process_audio(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_data = np.squeeze(indata)
    transcription = transcribe(audio_data)["text"]
    print(f"Transcription: {transcription}")

print("Listening... (Press Ctrl+C to stop)")
with sd.InputStream(
    channels=1,
    samplerate=args.sample_rate,
    callback=process_audio,
    blocksize=int(args.sample_rate * args.chunk_duration),
):
    try:
        sd.sleep(int(1e10))  # Keep the stream open indefinitely
    except KeyboardInterrupt:
        print("\nStopped listening.")

        
