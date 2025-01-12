import torch
import argparse
from torch.utils.data import DataLoader
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

####################### ARGUMENT PARSING #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument('--model_name', type=str, default='openai/whisper-small')
parser.add_argument('--language', type=str, default='English')
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument('--num_proc', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1.75e-5)
parser.add_argument('--train_batchsize', type=int, default=48)
parser.add_argument('--eval_batchsize', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--output_dir', type=str, default='output_model_dir')
parser.add_argument('--train_datasets', type=str, nargs='+', required=False,default="/home/gunmay/whisper-finetune/custom_data")
parser.add_argument('--eval_datasets', type=str, nargs='+', required=False,default="/home/gunmay/whisper-finetune/custom_data")
args = parser.parse_args()

print(f"Arguments: {vars(args)}")

######################## MODEL LOADING ###########################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

######################## DATASET LOADING #########################

def load_custom_dataset(split):
    datasets = []
    if split == 'train':
        for dset in args.train_datasets:
            datasets.append(load_from_disk(dset))
    elif split == 'eval':
        for dset in args.eval_datasets:
            datasets.append(load_from_disk(dset))
    return concatenate_datasets(datasets).shuffle(seed=42)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    transcription = batch["sentence"]
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

raw_dataset = DatasetDict({
    "train": load_custom_dataset('train'),
    "eval": load_custom_dataset('eval')
})

raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

train_loader = DataLoader(raw_dataset["train"], batch_size=args.train_batchsize, shuffle=True, collate_fn=lambda x: processor.pad(x, return_tensors="pt"))
eval_loader = DataLoader(raw_dataset["eval"], batch_size=args.eval_batchsize, collate_fn=lambda x: processor.pad(x, return_tensors="pt"))

####################### TRAINING LOOP ############################

def compute_wer(pred_ids, label_ids):
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    normalizer = BasicTextNormalizer()
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]
    return evaluate.load("wer").compute(predictions=pred_str, references=label_str)

for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
        optimizer.zero_grad()
        input_features = batch["input_features"].to(torch.float32).to(model.device)
        labels = batch["labels"].to(model.device)
        outputs = model(input_features, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}: Training Loss = {train_loss / len(train_loader)}")

    model.eval()
    eval_loss = 0
    wer_score = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_features = batch["input_features"].to(torch.float32).to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_features, labels=labels)
            eval_loss += outputs.loss.item()

            # Compute WER
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            wer_score += compute_wer(pred_ids.cpu(), labels.cpu())

    eval_loss /= len(eval_loader)
    wer_score /= len(eval_loader)

    print(f"Epoch {epoch+1}: Evaluation Loss = {eval_loss}, WER = {wer_score}")

    # Save model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

print("Training Completed.")
