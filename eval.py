from transformers import pipeline
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

# Load the fine-tuned Whisper model using pipeline
model_id = "./temp_dir_1"  # Replace with your model path
transcribe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    chunk_length_s=30,  # Replace with your chunk duration
    device=0,  # Replace with your device (e.g., 0 for GPU, -1 for CPU)
)

# Load a standard dataset (e.g., LibriSpeech)
dataset = load_dataset("librispeech_asr", "clean", split="test")  # Replace with your dataset

# Load the WER metric
wer_metric = load("wer")

def evaluate_model(transcribe, dataset, wer_metric):
    predictions = []
    references = []

    for sample in tqdm(dataset):
        # Get the audio and reference text
        audio = sample["audio"]["array"]
        reference = sample["text"]

        # Generate transcription using the pipeline
        prediction = transcribe(audio)["text"]

        # Append predictions and references
        predictions.append(prediction)
        references.append(reference)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

# Run evaluation
evaluate_model(transcribe, dataset, wer_metric)