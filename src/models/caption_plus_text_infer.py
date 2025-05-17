import time
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import os
import re


text_model_name = "sentence-transformers/LaBSE" # sentence-transformers/LaBSE, xlm-roberta-base

class StandaloneClassifier(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_emb):
        t = self.text_proj(text_emb)
        return self.classifier(t)

# --- Dataset Class for Inference ---
class CrisisTestDataset(Dataset):
    def __init__(self, df, text_tokenizer, text_model, max_length=128, device='cpu'):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model.to(device)
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df)

    def clean_tweet(self, text):
        # Remove RT at the start
        text = re.sub(r'^RT\s+', '', text)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_caption = row['caption']
        text = row[f'tweet_text_{args.lang}']
        if getattr(self, 'preprocess_text', False):
            text = self.clean_tweet(text)
        label = row['label']
        combined_text = f"Image is of - {image_caption}. The tweet attached is - {text}"
        encoded = self.text_tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            text_emb = self.text_model(**encoded).last_hidden_state[:, 0, :]  # CLS token
        text_emb = text_emb.squeeze(0)
        return {
            'text_emb': text_emb,
            'label': torch.tensor(label)
        }

# --- Inference Function ---
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    classifier = StandaloneClassifier().to(device)
    classifier.load_state_dict(torch.load(f"./saved_classifier/{args.epochs}_{args.lang}_standalone_classifier.pt", map_location=device))
    classifier.eval()

    # Load dataset
    if os.path.exists(args.caption_csv):
        print(f"Loading captions from {args.caption_csv}")
        df = pd.read_csv(args.caption_csv)
    else:
        print(f"Generating captions and saving to {args.caption_csv}")
        dataset = load_dataset("josecols/damage-mmd", split="test")
        rows = []
        for item in tqdm(dataset, desc="Generating captions"):
            image_path = f"test_data/{item['image_path']}"
            image = Image.open(image_path).convert('RGB')
            text = item[f'tweet_text_{args.lang}']
            if args.preprocess_text:
                text = clean_tweet(text)
            label = item['label']
            inputs = caption_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = caption_model.generate(**inputs)
            image_caption = caption_processor.decode(generated_ids[0], skip_special_tokens=True)
            rows.append({
                'image_path': image_path,
                'caption': image_caption,
                f'tweet_text_{args.lang}': text,
                'label': label
            })
        df = pd.DataFrame(rows)
        df.to_csv(args.caption_csv, index=False)

    crisis_dataset = CrisisTestDataset(df, tokenizer, text_model, device=device)
    crisis_dataset.preprocess_text = args.preprocess_text
    dataloader = DataLoader(crisis_dataset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_emb = batch['text_emb'].to(device)
            labels = batch['label'].to(device)
            logits = classifier(text_emb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            print(f"Preds: {preds}")

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    return accuracy, all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lang", type=str, default="")
    parser.add_argument("--caption_csv", type=str, default="captions_test.csv", help="CSV file to store/load image captions for test set")
    parser.add_argument("--preprocess_text", action="store_true", help="Whether to clean tweet text")
    args = parser.parse_args()

    t1 = time.time()

    evaluate(args)

    print(f"Evaluation completed in {time.time() - t1:.2f} seconds.")
