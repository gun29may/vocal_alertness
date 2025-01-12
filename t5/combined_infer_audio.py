import argparse
import os
from transformers import pipeline
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sys
from scipy.io import wavfile

transcriptions = []

def process_audio_file(file_path, transcribe):
    """
    Process a single audio file and return its transcription.
    
    Args:
        file_path (str): Path to the audio file.
        transcribe (pipeline): Whisper ASR pipeline.
        
    Returns:
        str: Transcription of the audio file.
    """
    try:
        # Load the audio file
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Ensure the audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Transcribe the audio
        transcription = transcribe(audio_data)["text"]
        print(f"Transcription for {file_path}: {transcription}")
        
        return transcription
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

class SentimentClassifier:
    def __init__(self, model_path):
        """
        Initialize the sentiment classifier with a trained model.
        
        Args:
            model_path (str): Path to the directory containing the trained model and tokenizer
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully and running on {self.device}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def classify_text(self, text, max_length=128):
        """
        Classify a single text input.
        
        Args:
            text (str): Input text to classify
            max_length (int): Maximum length for input tokenization
            
        Returns:
            str: Predicted sentiment label
        """
        try:
            # Prepare input
            input_text = f"classify: {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=32,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode prediction
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return predicted_text.strip()
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None

    def classify_batch(self, texts, batch_size=32, max_length=128):
        """
        Classify a batch of texts.
        
        Args:
            texts (list): List of input texts to classify
            batch_size (int): Size of batches for processing
            max_length (int): Maximum length for input tokenization
            
        Returns:
            list: List of predicted sentiment labels
        """
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                input_texts = [f"classify: {text}" for text in batch_texts]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    input_texts,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=32,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # Decode predictions
                batch_predictions = [
                    self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    for output in outputs
                ]
                results.extend(batch_predictions)
            
            return results
            
        except Exception as e:
            print(f"Error during batch classification: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Transcription and sentiment analysis of audio files in a directory.")

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
            default=44100,
            help="Sample rate for audio recording.",
        )
        parser.add_argument(
            "--chunk_duration",
            type=int,
            default=4,
            help="Duration of each audio chunk to transcribe in seconds.",
        )
        parser.add_argument(
            "--audio_dir", 
            type=str, 
            required=False,
            default="audio_v",
            help='Directory containing audio files to process.'
        )
        parser.add_argument(
            "--output_file", 
            type=str, 
            required=False, 
            help='Output file to save transcriptions and sentiment analysis.'
        )

        args = parser.parse_args()

        # Load Whisper model
        model_id = "openai/whisper-small"

        print("Loading Whisper model...")
        transcribe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            chunk_length_s=args.chunk_duration,
            device=args.device,
        )

        transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )

        # Process each audio file in the directory
        for filename in os.listdir(args.audio_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(args.audio_dir, filename)
                transcription = process_audio_file(file_path, transcribe)
                if transcription:
                    transcriptions.append(transcription)

        # Perform sentiment analysis on transcriptions
        classifier = SentimentClassifier("./results1/checkpoint-4630")
        results = classifier.classify_batch(transcriptions)
        
        # Save results to output file if provided
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for text, sentiment in zip(transcriptions, results):
                    f.write(f"Text: {text}\nSentiment: {sentiment}\n\n")
        
        # Print results to console
        print("Batch classification results:")
        for text, sentiment in zip(transcriptions, results):
            print(f"Text: {text}\nSentiment: {sentiment}\n")
            
    except Exception as e:
        print(f"Error in example usage: {str(e)}")