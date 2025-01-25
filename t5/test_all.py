import argparse
import random
import sounddevice as sd
import numpy as np
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time
import os
from scipy.io.wavfile import write

# Predefined list of sentences
additional_messages = [
    "I'm just scratched up, don't worry about me.",
    "It’s just a headache, I’ll be fine.",
    "I’m a little bruised, but nothing serious.",
    "Don’t worry, I’m okay, just shaken up.",
    "I’m fine, just some cuts and scrapes.",
    "It’s just a small cut, I’ll be alright.",
    "I’m good, where are the first responders?",
    "I feel a bit dizzy, but I’ll manage.",
    "No need to hurry for me, I’m fine.",
    "I’ve got a bruise, but it’s nothing to worry about.",
    "Just a scratch, I’m doing okay.",
    "It’s only a small headache, nothing serious.",
    "I’m good, no major injuries here.",
    "It’s just a bit of pain, nothing urgent.",
    "I’m okay, I think I just twisted my ankle.",
    "I’ll be alright, just a few scratches.",
    "It’s just a minor bruise, I’m fine.",
    "I’m alright, just a little shaken up.",
    "No major injuries, just a small cut.",
    "I’m okay, where are the paramedics?",
    "It’s just my arm, it’s not bad.",
    "I feel a bit sore, but I’m okay.",
    "Don’t worry about me, it’s just a bump.",
    "I’m good, just waiting for the medics.",
    "It’s just my head, a little dizzy but fine.",
    "I’m not in pain, just need some rest.",
    "I’m just tired, but I’m alright.",
    "Nothing major, just a small bruise.",
    "I’ll be fine, just a minor ache in my leg.",
    "It’s just my shoulder, but I can move.",
    "I’m okay, no serious injuries.",
    "It’s just a headache, I’ll be good soon.",
    "I’m just a little sore, no big deal.",
    "I’ll be fine, just need to sit down.",
    "Don’t worry, it’s only a small cut.",
    "I’m feeling alright, just a little bruised.",
    "It’s just my knee, but I’m good to go.",
    "I’m fine, no need to rush for me.",
    "Just a little dizziness, I’ll be okay.",
    "It’s just a scratch on my arm, nothing serious.",
    "I’m good, just feeling a bit off.",
    "Nothing to worry about, just a bump.",
    "I’m alright, just need some air.",
    "It’s only my leg, but I can still walk.",
    "Just a little headache, I’m okay.",
    "I’m fine, just a bit sore from the impact.",
    "I’m okay, where’s the rescue team?",
    "It’s just a bruise, nothing to worry about.",
    "I’ll be fine, just a bit of soreness.",
    "It’s only my hand, I’m okay otherwise.",
    "I’m doing alright, just waiting for help."
]
vocally_abnormal_messages = [
    "My sweet Caroline... where are you?",
    "Get out of my head! Get out! Stop it!",
    "The sky... why is the sky so purple today?",
    "I told them not to touch my things! I told them!",
    "I can hear the birds singing, can you hear them?",
    "Is it Tuesday? I can’t remember if it’s Tuesday.",
    "I need to find my keys, they’re lost somewhere.",
    "Why is everyone running? I just want to sleep.",
    "My dog’s coming to pick me up, I have to go!",
    "I have to go back... I forgot my umbrella!",
    "The stars are falling... I can see them!",
    "Where’s my homework? I need to turn it in!",
    "This train is taking me to the moon, I’m sure of it.",
    "The music is too loud, turn it off!",
    "Don’t touch my shoes! They’re mine!",
    "The colors are all wrong... everything’s wrong.",
    "I can’t find the fish... where did the fish go?",
    "It’s raining inside... why is it raining inside?",
    "I need to water my plants, they’re dying.",
    "Why is the floor moving? It won’t stop!",
    "I’m flying, look at me, I’m flying!",
    "The cake’s in the oven, it’s going to burn!",
    "I need to call my mom, she’s waiting for me.",
    "There’s sand in my shoes, I need to get it out.",
    "The walls are closing in, they’re too close!",
    "It’s so cold here, why is it snowing?",
    "The clocks won’t stop ticking, it’s too loud!",
    "I need to finish the painting, it’s almost done!",
    "They took my hat, I need it back!",
    "Why can’t I find the door? It’s always been here!",
    "My cat... where is my cat? She’s hiding from me.",
    "I’m late for work! I need to get to the office!",
    "The ice cream truck is coming, I need to get money!",
    "Get these spiders off me! They’re crawling everywhere!",
    "I have to bake the cookies before she gets home!",
    "Where’s the music coming from? It won’t stop!",
    "The floor is lava, don’t step on it!",
    "I need to feed the baby, she’s crying!",
    "The flowers are talking... can you hear them too?",
    "I’m swimming, but I can’t find the shore.",
    "Stop staring at me! Everyone’s staring!",
    "Where’s my boat? I was just on it a second ago.",
    "I have to catch the train! It’s leaving without me!",
    "The bees are everywhere, get them away!",
    "I’m late for my wedding! I need to get dressed!",
    "I told them not to build the tower so high!",
    "Why is the grass blue? It’s supposed to be green.",
    "I need to find my notebook, the answers are in there!",
    "I’m chasing the butterfly, it’s almost in my hand!",
    "The floor is upside down, how do I stand up?"
]
sentiment_real="abnormal"
PREDEFINED_SENTENCES=vocally_abnormal_messages
# Function to record audio
def record_audio(duration, sample_rate=16000):
    """
    Records audio for a specified duration.
    
    Args:
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sample rate for the recording.
    
    Returns:
        np.ndarray: Recorded audio data.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio_data.flatten()

# Function to save audio to a file
def save_audio(audio_data, sample_rate, file_name):
    """
    Saves audio data to a WAV file.
    
    Args:
        audio_data (np.ndarray): Audio data to save.
        sample_rate (int): Sample rate of the audio.
        file_name (str): Base name of the output file.
    
    Returns:
        str: Full path to the saved audio file.
    """
    # Create the audio_files directory if it doesn't exist
    if not os.path.exists("audio_files"):
        os.makedirs("audio_files")
    
    # Replace spaces with underscores in the file name
    base_name = file_name.replace(" ", "_")
    file_path = os.path.join("audio_files", f"{base_name}.wav")
    
    # Handle duplicate file names
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join("audio_files", f"{base_name}_{i}.wav")
        i += 1
    
    # Save the audio file
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved to {file_path}")
    return file_path

# Function to transcribe audio
def transcribe_audio(audio_data, transcribe_pipeline):
    """
    Transcribes audio using the Whisper model.
    
    Args:
        audio_data (np.ndarray): Audio data to transcribe.
        transcribe_pipeline: Whisper pipeline for transcription.
    
    Returns:
        str: Transcribed text.
    """
    transcription = transcribe_pipeline(audio_data)["text"]
    return transcription

# SentimentClassifier class
class SentimentClassifier:
    def __init__(self, model_path):
        """
        Initializes the sentiment classifier with a trained T5 model.
        
        Args:
            model_path (str): Path to the T5 model.
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
        Classifies a single text input.
        
        Args:
            text (str): Input text to classify.
            max_length (int): Maximum length for input tokenization.
        
        Returns:
            str: Predicted sentiment label.
        """
        try:
            input_text = f"given a person is asked describe your condtion? answer whether the person is normal or abnormal with reason : {text}"
            inputs = self.tokenizer(
                input_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=32,
                    num_beams=4,
                    early_stopping=True
                )
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return predicted_text.strip()
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return None

# Main function
def main(args):
    """
    Main function to run the entire pipeline in a loop:
    1. Selects a random sentence.
    2. Records audio for the sentence.
    3. Saves the audio to a uniquely named file.
    4. Transcribes the audio using Whisper.
    5. Analyzes sentiment using T5.
    6. Saves the outputs to a file.
    7. Repeats until the user chooses to exit.
    """
    # Load Whisper model
    print("Loading Whisper model...")
    transcribe_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=args.whisper_model_path,
        chunk_length_s=args.chunk_duration,
        device=args.device,
    )

    # Load T5 sentiment classifier
    print("Loading T5 sentiment classifier...")
    sentiment_classifier = SentimentClassifier(args.t5_model_path)

    # Loop to repeatedly test the pipeline
    while True:
        # Randomly sample a sentence from the predefined list
        selected_sentence = random.choice(PREDEFINED_SENTENCES)
        print(f"\nSelected sentence: {selected_sentence}")

        # Record audio for the selected sentence
        print("Please say the following sentence:")
        print(selected_sentence)
        a=input("start with keypress")
        audio_data = record_audio(duration=args.chunk_duration, sample_rate=args.sample_rate)

        # Save the recorded audio to a uniquely named file
        audio_file_path = save_audio(audio_data, args.sample_rate, selected_sentence)

        # Transcribe the recorded audio
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_data, transcribe_pipeline)
        print(f"Transcription: {transcription}")

        # Perform sentiment analysis on the transcription
        print("Analyzing sentiment...")
        # sentiment = sentiment_classifier.classify_text(transcription)
        sentiment = sentiment_classifier.classify_text(transcribe_audio)

        print(f"Sentiment: {sentiment}")

        # Save outputs to a file
        with open(args.output_file, "a") as f:
            f.write(f"Selected Sentence: {selected_sentence}\n")
            f.write(f"Audio File: {audio_file_path}\n")
            f.write(f"Transcription: {transcription}\n")
            f.write(f"Sentiment: {sentiment}\n")
            f.write(f"actual sentiment :normal\n")
            f.write("-" * 40 + "\n")  # Separator for readability
        print(f"Outputs appended to {args.output_file}")

        # Ask the user if they want to continue
        user_input = input("\nDo you want to test another sentence? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("Exiting the loop. Goodbye!")
            break

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Whisper and T5 models with predefined sentences in a loop.")
    parser.add_argument(
        "--whisper_model_path",
        type=str,
        default="/home/gunmay/audio_classifiaction/temp_dir_1",
        help="Path to the Whisper model."
    )
    parser.add_argument(
        "--t5_model_path",
        type=str,
        default="./results1/checkpoint-4610",
        help="Path to the T5 sentiment model."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio recording."
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=5,
        help="Duration of each audio chunk to transcribe in seconds."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device to run the pipeline on. Use -1 for CPU, 0 for GPU."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.txt",
        help="File to save the outputs."
    )
    args = parser.parse_args()

    # Run the main function
    main(args)