import argparse
from transformers import pipeline
import sounddevice as sd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sys


transcriptions = []
def process_audio(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_data = np.squeeze(indata)
    transcription = transcribe(audio_data)["text"]
    print(f"Transcription: {transcription}")

    transcriptions.append(transcription)


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
        parser.add_argument(
            "--output_file", 
            type=str, 
            required=False, 
            help='Transcriptions reqd for T5')


        args = parser.parse_args()

        #WHISPER STARTS HERE
        model_id = "/home/gunmay/whisper-finetune/temp_dir_1"

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

        print("Listening... (Press Ctrl+C to stop)")
        with sd.InputStream(
            channels=1,
            samplerate=args.sample_rate,
            callback=process_audio,
            blocksize=int(args.sample_rate * args.chunk_duration),
        ):
            try:
                sd.sleep(int(5000))  # Keep the stream open indefinitely
            except KeyboardInterrupt:
                print("\nStopped listening.")
        #WHISPER ENDS HERE\
        
        #T5 INFERENCE ON TRANSCRIPTION FILE STARTS HERE
        classifier = SentimentClassifier("./results1/checkpoint-4630")
        texts_1 = transcriptions
        print(f"Transcriptions: {texts_1}")
        # results = classifier.classify_text(texts)
        # print("Batch classification results:")
        # with open(args.output_file, 'w') as f:
        #   for text, sentiment in zip(texts, results):
        #     print(f"Text: {text}\TSentiment: {sentiment}\n")
        #     f.write(f"Text: {text}\TSentiment: {sentiment}\n")
        # sys.exit()
        #T5 INFERENCE ON TRANSCRIPTION FILE ENDS HERE

        
        # Initialize classifier
        # classifier = SentimentClassifier("./results/checkpoint-1360")
        
        
        # Single text classification
        text = "This product is amazing and works perfectly!"
        result = classifier.classify_text(text)
        print(f"Single text classification result: {result}")
        
        results = classifier.classify_batch(texts_1)
        print("Batch classification results:")
        for text, sentiment in zip(texts_1, results):
            print(f"Text: {text}\nSentiment: {sentiment}\n")
            
    except Exception as e:
        print(f"Error in example usage: {str(e)}")