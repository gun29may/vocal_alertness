# Initialize empty lists to store the data

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TextGenerator:
    def __init__(self, model_path):
        """
        Initialize the text generator with a trained model.
        
        Args:
            model_path (str): Path to the directory containing the trained model and tokenizer
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print(f"Model loaded successfully and running on {self.device}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def generate_response(self, text, max_length=256):
        """
        Generate a response for a single text input.
        
        Args:
            text (str): Input text to generate a response for
            max_length (int): Maximum length for input tokenization
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare input
            input_text = f"given a person is asked describe your condtion? answer whether the person is normal or abnormal with reason : {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=64,  # Adjust based on your output length
                    num_beams=4,    # Beam search for better results
                    early_stopping=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return None

    def generate_batch_responses(self, texts, batch_size=32, max_length=128):
        """
        Generate responses for a batch of texts.
        
        Args:
            texts (list): List of input texts to generate responses for
            batch_size (int): Size of batches for processing
            max_length (int): Maximum length for input tokenization
            
        Returns:
            list: List of generated responses
        """
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                input_texts = [f"given a person is asked describe your condtion? answer whether the person is normal or abnormal with reason : {text}" for text in batch_texts]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    input_texts,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate responses
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=64,  # Adjust based on your output length
                        num_beams=4,    # Beam search for better results
                        early_stopping=True
                    )
                
                # Decode responses
                batch_responses = [
                    self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    for output in outputs
                ]
                results.extend(batch_responses)
            
            return results
            
        except Exception as e:
            print(f"Error during batch text generation: {str(e)}")
            return None
def evaluate_accuracy(responses, gt):
    correct_predictions = 0
    total_predictions = len(responses)
    false_cases = []  # To store false cases
    
    for i, response in enumerate(responses):
        first_word = response.split(",")[0]  # Extract the first word
        # print(first_word,response,gt[i])
        if first_word.lower() == gt[i].lower():
            correct_predictions += 1
        else:
            false_cases.append((i, response))  # Store the index and the false response
    
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy
# Read the file
def get_data(filename):
    selected_sentences = []
    audio_files = []
    transcriptions = []
    sentiments = []
    actual_sentiments = []
    with open( filename, 'r') as file:
        lines = file.readlines()

    # Process the file line by line
    for line in lines:
        if line.startswith("Selected Sentence:"):
            selected_sentences.append(line.split("Selected Sentence: ")[1].strip())
        elif line.startswith("Audio File:"):
            audio_files.append(line.split("Audio File: ")[1].strip())
        elif line.startswith("Transcription:"):
            transcriptions.append(line.split("Transcription: ")[1].strip())
        elif line.startswith("Sentiment:"):
            sentiments.append(line.split("Sentiment: ")[1].strip())
        elif line.startswith("actual sentiment :"):
            actual_sentiments.append(line.split("actual sentiment :")[1].strip())
    # print(selected_sentences)
    return selected_sentences,audio_files,transcriptions,sentiments,actual_sentiments
# Print the lists to verify the data
if __name__ == "__main__":
    selected_sentences,audio_files,transcriptions,sentiments,actual_sentiments=get_data('output_noisy_inside.txt')
    # print("Selected Sentences:", selected_sentences)
    # print("Audio Files:", audio_files)
    # print("Transcriptions:", transcriptions)
    # print("Sentiments:", sentiments)
    # print("Actual Sentiments:", actual_sentiments)
    generator = TextGenerator("./results3/checkpoint-11360")
    responses = generator.generate_batch_responses(selected_sentences)
    normal_accuracy = evaluate_accuracy(responses, actual_sentiments)
    print(f"Accuracy ==: {normal_accuracy:.2f}%")