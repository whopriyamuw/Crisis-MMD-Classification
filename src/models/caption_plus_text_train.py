import time
import argparse
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import re


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

def clean_tweet(text):
    # Remove RT at the start
    text = re.sub(r'^RT\s+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove #words
    text = re.sub(r'#\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Training Function ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f"Using device: {device}")

    # Load vision model (BLIP) for captioning
    caption_model = BlipForConditionalGeneration.from_pretrained(args.image_model_name)
    caption_processor = AutoProcessor.from_pretrained(args.image_model_name)

    # Load text model
    text_model = AutoModel.from_pretrained(args.text_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # Dynamically determine text embedding dimension
    dummy_input = tokenizer("hello", return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_emb = text_model(**dummy_input).last_hidden_state[:, 0, :]
    text_dim = dummy_emb.shape[1]

    # Standalone classifier
    hidden_dim = 512
    classifier = StandaloneClassifier(text_dim=text_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
    optimizer = AdamW(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load your dataset
    dataset = load_dataset("josecols/damage-mmd", split="train")
    val_dataset = load_dataset("josecols/damage-mmd", split="dev")

    # --- Caption CSV logic for train ---
    if os.path.exists(args.caption_csv):
        print(f"Loading captions from {args.caption_csv}")
        df = pd.read_csv(args.caption_csv)
        print (df['label'].value_counts())
    else:
        print(f"Generating captions and saving to {args.caption_csv}")
        rows = []
        for item in tqdm(dataset, desc="Generating captions"):
            image_path = item['image_path']
            image = Image.open(image_path).convert('RGB')
            text = item[f'tweet_text_{args.lang}']
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

    # --- Caption CSV logic for validation ---
    if os.path.exists(args.val_caption_csv):
        print(f"Loading validation captions from {args.val_caption_csv}")
        val_df = pd.read_csv(args.val_caption_csv)
        print (val_df['label'].value_counts())
    else:
        print(f"Generating validation captions and saving to {args.val_caption_csv}")
        rows = []
        for item in tqdm(val_dataset, desc="Generating validation captions"):
            image_path = item['image_path']
            image = Image.open(image_path).convert('RGB')
            text = item[f'tweet_text_{args.lang}']
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
        val_df = pd.DataFrame(rows)
        val_df.to_csv(args.val_caption_csv, index=False)

    class CombinedTextDataset(Dataset):
        def __init__(self, df, text_tokenizer, text_model, max_length=128, device='cpu'):
            self.df = df
            self.text_tokenizer = text_tokenizer
            self.text_model = text_model.to(device)
            self.max_length = max_length
            self.device = device

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_caption = row['caption']
            text = row[f'tweet_text_{args.lang}']
            if args.preprocess_text:
                text = clean_tweet(str(text))
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

    combined_dataset = CombinedTextDataset(df, tokenizer, text_model, device=device)
    val_combined_dataset = CombinedTextDataset(val_df, tokenizer, text_model, device=device)
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_combined_dataset, batch_size=args.batch_size, shuffle=False)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch['text_emb'].to(device)
                labels = batch['label'].to(device)
                logits = model(text_emb)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    classifier.train()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            text_emb = batch['text_emb'].to(device)
            labels = batch['label'].to(device)
            logits = classifier(text_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        val_loss, val_acc = evaluate(classifier, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(classifier.state_dict(), args.output_dir + f"/{args.lang}_standalone_classifier.pt")
            print(f"New best model saved with Val Acc: {val_acc:.4f}")
    
    print("Standalone classifier training completed.")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./saved_classifier")
    parser.add_argument('--text_model_name', type=str, default='xlm-roberta-base') # can use xlm-roberta-base or sentence-transformers/LaBSE
    parser.add_argument('--image_model_name', type=str, default='Salesforce/blip-image-captioning-base')
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--caption_csv", type=str, default="captions.csv")
    parser.add_argument("--val_caption_csv", type=str, default="captions_val.csv")
    parser.add_argument("--preprocess_text", action="store_true")
    args = parser.parse_args()

    t1 = time.time()
    
    train(args)

    print (f"Training completed in {time.time() - t1:.2f} seconds.")
