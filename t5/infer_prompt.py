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
def evaluate_accuracy(responses, expected_class,x):
    correct_predictions = 0
    total_predictions = len(responses)
    false_cases = []  # To store false cases
    
    for i, response in enumerate(responses):
        first_word = response.split()[0]  # Extract the first word
        first_word=first_word[0:x]
        if first_word.lower() == expected_class.lower():
            correct_predictions += 1
        else:
            false_cases.append((i, response))  # Store the index and the false response
    
    accuracy = (correct_predictions / total_predictions) * 100
    
    # Print all false cases
    if false_cases:
        print(f"\nFalse Cases for '{expected_class}' class:")
        for idx, false_response in false_cases:
            print(f"Index: {idx}, Generated Response: {false_response[0:8]}, text {ABNORMAL[idx]}")
    else:
        print(f"\nNo false cases for '{expected_class}' class.")
    
    return accuracy

# Example usage
if __name__ == "__main__":
    try:
        # Initialize text generator
        generator = TextGenerator("./results1/checkpoint-4610")  # Path to your trained model
        
        # Single text generation
        text = "Help help help! The train flipped, and I can’t breathe!"
        response = generator.generate_response(text)
        print(f"Input: {text}\nGenerated Response: {response}\n")
        
        # Batch text generation
        texts = [
           "my sweet caroline where are you",
    "get out of my head get out stop it",
    "the sky why is the sky so purple today",
    "i told them not to touch my things i told them",
    "i can hear the birds singing can you hear them",
    "is it tuesday i can’t remember if it’s tuesday",
    "i need to find my keys they’re lost somewhere",
    "why is everyone running i just want to sleep",
    "my dog’s coming to pick me up i have to go",
    "i have to go back i forgot my umbrella",
    "the stars are falling i can see them",
    "where’s my homework i need to turn it in",
    "this train is taking me to the moon i’m sure of it",
    "the music is too loud turn it off",
    "don’t touch my shoes they’re mine",
    "the colors are all wrong everything’s wrong",
    "i can’t find the fish where did the fish go",
    "it’s raining inside why is it raining inside",
    "i need to water my plants they’re dying",
    "why is the floor moving it won’t stop",
    "i’m flying look at me i’m flying",
    "the cake’s in the oven it’s going to burn",
    "i need to call my mom she’s waiting for me",
    "there’s sand in my shoes i need to get it out",
    "the walls are closing in they’re too close",
    "it’s so cold here why is it snowing",
    "the clocks won’t stop ticking it’s too loud",
    "i need to finish the painting it’s almost done",
    "they took my hat i need it back",
    "why can’t i find the door it’s always been here",
    "my cat where is my cat she’s hiding from me",
    "i’m late for work i need to get to the office",
    "the ice cream truck is coming i need to get money"
        ]
        NORMAL= [
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
        ABNORMAL= [
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
  ]
        responses = generator.generate_batch_responses(NORMAL)
        normal_accuracy = evaluate_accuracy(responses, "normal",6)
        print(f"Accuracy for NORMAL class: {normal_accuracy:.2f}%")
    
        responses = generator.generate_batch_responses(ABNORMAL)
        # print(responses)
        normal_accuracy = evaluate_accuracy(responses, "abnormal",8)
        print(f"Accuracy for ABNORMAL class: {normal_accuracy:.2f}%")    
        
        
        print("Batch text generation results:")
        for text, response in zip(texts, responses):
            print(f"Input: {text}\nGenerated Response: {response}\n")
            
    except Exception as e:
        print(f"Error in example usage: {str(e)}")