import torch
import argparse
from torch.utils.data import DataLoader
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from dataclasses import dataclass
import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torchaudio.transforms as T

# Define audio transformations
volume_transform = T.Vol(gain=0.5)  # Adjust volume
time_masking = T.TimeMasking(time_mask_param=35)  # Time masking
frequency_masking = T.FrequencyMasking(freq_mask_param=15) 

####################### ARGUMENT PARSING #########################
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument('--model_name', type=str, default='openai/whisper-small')
parser.add_argument('--language', type=str, default='English')
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument('--num_proc', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1.75e-5)
parser.add_argument('--train_batchsize', type=int, default=7)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--train_datasets', type=str, nargs='+', required=False,default="../data/train_data/")
parser.add_argument('--eval_datasets', type=str, nargs='+', required=False,default="../data/eval_data/")

args = parser.parse_args()

print(f"Arguments: {vars(args)}")

######################## MODEL LOADING ###########################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
model.train()
max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


######################## DATASET LOADING #########################

def load_custom_dataset(split):
    datasets = []
    if split == 'train':
        
        datasets.append(load_from_disk(args.train_datasets))
        return concatenate_datasets(datasets).shuffle(seed=42)
    if split == 'eval':
        print("eval")
        datasets.append(load_from_disk(args.eval_datasets))
        return concatenate_datasets(datasets).shuffle(seed=42)


def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]
    audio_tensor = torch.tensor(audio["array"], dtype=torch.float32)
    audio_tensor = volume_transform(audio_tensor)
    audio_tensor = time_masking(audio_tensor)
    audio_tensor = frequency_masking(audio_tensor)
    audio["array"] = audio_tensor

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

raw_dataset = DatasetDict()
raw_dataset["train"] = load_custom_dataset('train')
raw_dataset["eval"] = load_custom_dataset('eval')
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)
raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print(data_collator)
print('DATASET PREPARATION COMPLETED')
training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=20000,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100000000000,
        save_total_limit=10,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=1,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=None,
    )
# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=raw_dataset["train"],
#     eval_dataset=raw_dataset["train"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,  # Replace tokenizer with feature_extractor
)
processor.save_pretrained(training_args.output_dir)

trainer.train()

# train_loader = DataLoader(raw_dataset["train"], batch_size=args.train_batchsize, shuffle=True, collate_fn=lambda x: processor.pad(x, return_tensors="pt"))

####################### TRAINING LOOP ############################

# for epoch in range(args.num_epochs):
#     model.train()
#     train_loss = 0
#     for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
#         optimizer.zero_grad()
#         input_features = batch["input_features"].to(torch.float32).to(model.device)
#         labels = batch["labels"].to(model.device)
#         outputs = model(input_features, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     print(f"Epoch {epoch+1}: Training Loss = {train_loss / len(train_loader)}")

#     # Save model
#     model.save_pretrained(args.output_dir)
#     processor.save_pretrained(args.output_dir)

# print("Training Completed.")

