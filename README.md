# Verbal Alertness Detection Project

This project focuses on detecting verbal alertness in audio recordings. It leverages two powerful models: Whisper for accurate audio transcription and T5 for nuanced sentiment classification, which is used as a proxy for verbal alertness.

## Project Overview

The core idea is that alert individuals tend to express more positive or active sentiment, while less alert individuals might exhibit more neutral or negative sentiment in their speech. By transcribing audio and then analyzing the sentiment of the text, we can derive insights into the speaker's verbal alertness.

## Key Features

*   **Audio Transcription:** Utilizes the Whisper model to convert audio recordings into accurate text transcripts.
*   **Sentiment Classification:** Employs the T5 model to classify the sentiment of the transcribed text.
*   **Data Preparation:** Includes scripts to process and prepare audio and text data for model training and inference.
*   **Training and Fine-tuning:** Supports training and fine-tuning of the T5 model on custom datasets.
* **Output:** output of the files are stored in output.txt file in t5 folder.

## Project Structure

*   **`data_prep.py`:** Scripts for processing and preparing data, including audio and text files.
*   **`eval.py`:** Contains functions for evaluating the performance of trained models.
*   **`infer_batch.py`:** Used for performing batch inference on a set of audio files.
*   **`mic.py`:** Enables real-time transcription and analysis using a microphone.
*   **`save.py`:** contains functions for saving models
*   **`transcribe_audio.py`:**  Transcribes audio files using the Whisper model.
*   **`.vscode/settings.json`:** configurations files for vs code.
*   **`t5/`:** Directory containing T5-related code.
    *   **`combined_infer_audio.py`:** performs inference on audio files using whisper and then classify the sentiment using T5 model.
    *   **`infer_prompt.py`:** performs inference on a given prompt using T5 model.
    *   **`infert5.py`:** performs inference on a text file using T5 model.
    *   **`infert5_combined.py`:** Combines the transcript and classifies the sentiment using T5 model.
    *   **`output.txt`:** Stores the output of the T5 model's inference results.
    *   **`t5n.py`:** Contains the implementation for T5 model.
    *   **`test_all.py`:**  Tests the functions in t5 folder.
    * **`test_txt.py`** Tests the T5 sentiment classification.
    *   **`train_t5.py`:** Trains or fine-tunes the T5 model.
    *   **`logs/`:** Contains tensorboard logs.
*   **`train/`:** Directory for training-related scripts.
    *   **`fine-tune_on_custom_dataset.py`:** fine-tunes a pre-trained model on custom data.
    *   **`newtrain.py`:** script to train model with custom data.
    *   **`train.py`:** Main training script.
    * **`trainnew.py`:** New training script for fine tuning the model.
    *   **`whisper-env.yml`:**  Conda environment file for whisper.

## Setup Instructions

1.  **Clone the Repository:**
```
bash
    git clone <repository-url>
    cd <repository-directory>
    
```
2.  **Create and Activate Conda Environments:**
```
bash
   conda env create -f environment.yml
   conda activate whisper
   
```
3.  **Download Models:**
    *   Ensure the Whisper and T5 models are downloaded by the respective scripts. The scripts will handle downloading the models on the first run.

## Data Preparation

1.  **Audio Data:** Collect audio recordings of individuals.
2.  **Transcription (Optional):** If you have manual transcripts, prepare them in a text file.
3. **Text classification dataset**: Prepare a text file with the label for each of the input sentences.
4.  **Data Processing:** Use `data_prep.py` to process the audio and text data. This might involve:
    *   Resampling audio to the desired sample rate.
    *   Splitting audio files into segments.
    *   Converting transcripts to the correct format.

## Training and Fine-tuning

1.  **T5 Fine-tuning:** Use `train/train.py`, `train/trainnew.py` or  `train/fine-tune_on_custom_dataset.py` scripts to train or fine-tune the T5 model with your dataset.
2.  **Training Configuration:** Adjust hyperparameters (e.g., learning rate, batch size) in the training script.
3.  **Monitor Training:** Check tensorboard logs for model performance.
4. **Save model**: Once training is done run save.py to save the trained model.

## Usage Examples

1.  **Transcribe an Audio File:**
```
bash
    python transcribe_audio.py --audio_file audio.wav
    
```
2.  **Infer Sentiment:**
```
bash
    python t5/infert5.py --text_file input.txt
    
```
3. **Infer from Audio File**:
```
bash
    python t5/combined_infer_audio.py --audio_file audio.wav
    
```
4. **Infer from Prompt**:
```
bash
    python t5/infer_prompt.py --prompt "hello"
    
```
5.  **Real-time Transcription and Analysis:**
```
bash
    python mic.py
    
```
## How the Scripts Work Together

1.  **Audio Input:** The project begins with an audio file or real-time audio from a microphone.
2.  **Transcription (Whisper):** The `transcribe_audio.py` or `mic.py` script uses the Whisper model to transcribe the audio into text.
3.  **Sentiment Classification (T5):** The transcribed text is then passed to the `t5/infert5.py` or `t5/infert5_combined.py` script, which uses the T5 model to classify the sentiment.
4.  **Alertness Inference:** The sentiment score is used as a proxy for verbal alertness. For instance, a positive sentiment might indicate higher alertness, while a neutral or negative sentiment might suggest lower alertness.
5. **Output**: output of the files are stored in output.txt file in t5 folder.

## Future Improvements

*   Implement more sophisticated metrics for verbal alertness.
*   Explore other models for sentiment classification.
*   Improve the real-time analysis capabilities.
*   Add a user interface for easier interaction.