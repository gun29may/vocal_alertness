from transformers import pipeline
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import numpy as np
import librosa  # For resampling audio if needed

# Load the fine-tuned Whisper model using pipeline
model_id = "./temp_dir_1"  # Replace with your model path
device = 0  # Replace with your device (e.g., 0 for GPU, -1 for CPU)
batch_size = 8  # Adjust based on GPU memory

transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,  # Replace with your chunk duration
    device=device,
    batch_size=batch_size,  # Enable batch processing
)

# Load the LJ Speech dataset
dataset = load_dataset("lj_speech", split="train[:10%]", trust_remote_code=True)  # Use a smaller subset for testing

# Load the WER metric
wer_metric = load("wer")

def evaluate_model(transcribe, dataset, wer_metric, batch_size):
    predictions = []
    references = []

    # Iterate over the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        
        # Extract audio arrays and reference texts
        audios = []
        batch_references = []
        for sample in batch:
            try:
                # Ensure the sample is a dictionary
                if not isinstance(sample, dict):
                    raise ValueError(f"Unexpected sample type: {type(sample)}")

                # Ensure the audio key exists and is a dictionary
                if "audio" not in sample or not isinstance(sample["audio"], dict):
                    raise ValueError(f"Missing or invalid 'audio' key in sample: {sample}")

                # Ensure the audio array exists
                if "array" not in sample["audio"]:
                    raise ValueError(f"Missing 'array' key in audio: {sample['audio']}")

                audio = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]
                
                # Resample to 16kHz if necessary (Whisper expects 16kHz)
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
                
                audios.append(audio)
                batch_references.append(sample["text"])
            except Exception as e:
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                audios.append(np.zeros(0))  # Add empty audio for failed sample
                batch_references.append("")  # Add empty reference for failed sample

        # Transcribe the batch of audios
        try:
            batch_predictions = transcribe(audios)
            predictions.extend([pred["text"] for pred in batch_predictions])
            references.extend(batch_references)
        except Exception as e:
            print(f"Error transcribing batch {i}: {e}")
            predictions.extend([""] * len(batch))  # Add empty predictions for failed batch
            references.extend(batch_references)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

# Run evaluation
evaluate_model(transcribe, dataset, wer_metric, batch_size)