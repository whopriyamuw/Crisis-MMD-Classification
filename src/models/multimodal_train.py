import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
import re


# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

# --- Model Definitions ---
class DeepClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(input_dim, hidden_dim * 2),
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

    def forward(self, x):
        return self.classifier(x)

class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.classifier = DeepClassifier(hidden_dim * 2, num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)
        x = torch.cat([t, i], dim=1)
        return self.classifier(x)

class TextOnlyClassifier(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.classifier = DeepClassifier(hidden_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, text_emb):
        t = self.text_proj(text_emb)
        return self.classifier(t)

class ImageOnlyClassifier(nn.Module):
    def __init__(self, image_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.classifier = DeepClassifier(hidden_dim, num_classes=num_classes, hidden_dim=hidden_dim)

    def forward(self, image_emb):
        i = self.image_proj(image_emb)
        return self.classifier(i)

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

def rename_text_field(example):
    if 'tweet_text' in example:
        example['tweet_text_en'] = example['tweet_text']
    return example

# --- Embedding Functions ---
def get_text_embedding(texts, tokenizer, model):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    output = model(**tokens)
    return output.last_hidden_state[:, 0, :]

def get_image_embedding(images, processor, model):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    output = model(**inputs)
    return output.pooler_output

# --- Dataset Utilities ---
def image_loader(path):
    image = Image.open(path).convert("RGB")
    return image

def collate_fn(batch, text_tokenizer, text_model, clip_processor, clip_model, lang):
    if args.preprocess_text:
        texts = [clean_tweet(item[f"tweet_text_{lang}"]) for item in batch]
    else:
        texts = [item[f"tweet_text_{lang}"] for item in batch]
    images = [image_loader(item["image_path"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    text_embs = get_text_embedding(texts, text_tokenizer, text_model)
    image_embs = get_image_embedding(images, clip_processor, clip_model)
    return text_embs, image_embs, labels

def collate_fn_text_only(batch, text_tokenizer, text_model, lang):
    if args.preprocess_text:
        texts = [clean_tweet(item[f"tweet_text_{lang}"]) for item in batch]
    else:
        texts = [item[f"tweet_text_{lang}"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    text_embs = get_text_embedding(texts, text_tokenizer, text_model)
    return text_embs, labels

def collate_fn_image_only(batch, clip_processor, clip_model):
    images = [image_loader(item["image_path"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    image_embs = get_image_embedding(images, clip_processor, clip_model)
    return image_embs, labels

# --- Training Utilities ---
def train_multimodal(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for text_embs, image_embs, labels in tqdm(dataloader, desc=f"Multimodal Epoch {epoch+1}"):
            text_embs, image_embs, labels = text_embs.to(device), image_embs.to(device), labels.to(device)
            logits = model(text_embs, image_embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def train_text_only(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for text_embs, _, labels in tqdm(dataloader, desc=f"Text-Only Epoch {epoch+1}"):
            text_embs, labels = text_embs.to(device), labels.to(device)
            logits = model(text_embs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"[Text-Only] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def evaluate_multimodal(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for text_embs, image_embs, labels in dataloader:
            text_embs, image_embs, labels = text_embs.to(device), image_embs.to(device), labels.to(device)
            logits = model(text_embs, image_embs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate_text_only(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for text_embs, labels in dataloader:
            text_embs, labels = text_embs.to(device), labels.to(device)
            logits = model(text_embs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate_image_only(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for image_embs, labels in dataloader:
            image_embs, labels = image_embs.to(device), labels.to(device)
            logits = model(image_embs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# --- Main ---
def main(args):
    t1 = time.time()
    print(f"Using device: {device}")

    # Use arguments instead of hardcoded config
    text_model_name = args.text_model_name
    image_model_name = args.image_model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    lang = args.lang

    # Load text model and tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    # Dynamically determine text embedding dimension (like image_cap_train)
    dummy_input = text_tokenizer("hello", return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_emb = text_model(**dummy_input).last_hidden_state[:, 0, :]
    text_dim = dummy_emb.shape[1]
    # Load image model and processor
    clip_model = CLIPModel.from_pretrained(image_model_name).vision_model.to(device)
    clip_processor = CLIPProcessor.from_pretrained(image_model_name)
    image_dim = 768  # CLIP ViT-base-patch32 default
    hidden_dim = 512

    # Load dataset
    full_data = load_dataset("josecols/damage-mmd", split="train")
    val_data = load_dataset("josecols/damage-mmd", split="dev")
    
    full_data = full_data.map(rename_text_field)
    val_data = val_data.map(rename_text_field)
    train_data = full_data
    print(f"Number of training samples: {len(train_data)}")

    # Dataloaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, text_tokenizer, text_model, clip_processor, clip_model, lang)
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, text_tokenizer, text_model, clip_processor, clip_model, lang)
    )
    val_loader_text = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn_text_only(batch, text_tokenizer, text_model, lang)
    )
    val_loader_image = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn_image_only(batch, clip_processor, clip_model)
    )
    train_loader_image = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn_image_only(batch, clip_processor, clip_model)
    )

    if args.mode in ['multimodal', 'all']:
        # --- Multimodal Training ---
        model = MultimodalClassifier(text_dim=text_dim, image_dim=image_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for text_embs, image_embs, labels in tqdm(train_loader, desc=f"Multimodal Epoch {epoch+1}"):
                text_embs, image_embs, labels = text_embs.to(device), image_embs.to(device), labels.to(device)
                logits = model(text_embs, image_embs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            val_loss, val_acc = evaluate_multimodal(model, val_loader, criterion, device)
            print(f"[Multimodal] Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"model_weights/{lang}_multimodal_classifier.pt")
                print(f"[Multimodal] New best model saved with Val Acc: {val_acc:.4f}")
        print(f"Training completed in {time.time() - t1:.2f} seconds.")

    if args.mode in ['text', 'all']:
        # --- Text-Only Training ---
        text_only_model = TextOnlyClassifier(text_dim=text_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        text_only_optimizer = torch.optim.AdamW(text_only_model.parameters(), lr=learning_rate)
        text_only_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            text_only_model.train()
            total_loss = 0
            for text_embs, _, labels in tqdm(train_loader, desc=f"Text-Only Epoch {epoch+1}"):
                text_embs, labels = text_embs.to(device), labels.to(device)
                logits = text_only_model(text_embs)
                loss = text_only_criterion(logits, labels)
                loss.backward()
                text_only_optimizer.step()
                text_only_optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            val_loss, val_acc = evaluate_text_only(text_only_model, val_loader_text, text_only_criterion, device)
            print(f"[Text-Only] Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(text_only_model.state_dict(), f"model_weights/{lang}_text_only_classifier.pt")
                print(f"[Text-Only] New best model saved with Val Acc: {val_acc:.4f}")
        print("Text-only training completed.")

    if args.mode in ['image', 'all']:
        # --- Image-Only Training ---
        image_only_model = ImageOnlyClassifier(image_dim=image_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        image_only_optimizer = torch.optim.AdamW(image_only_model.parameters(), lr=learning_rate)
        image_only_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            image_only_model.train()
            total_loss = 0
            for image_embs, labels in tqdm(train_loader_image, desc=f"Image-Only Epoch {epoch+1}"):
                image_embs, labels = image_embs.to(device), labels.to(device)
                logits = image_only_model(image_embs)
                loss = image_only_criterion(logits, labels)
                loss.backward()
                image_only_optimizer.step()
                image_only_optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader_image)
            val_loss, val_acc = evaluate_image_only(image_only_model, val_loader_image, image_only_criterion, device)
            print(f"[Image-Only] Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(image_only_model.state_dict(), f"model_weights/{lang}_image_only_classifier.pt")
                print(f"[Image-Only] New best model saved with Val Acc: {val_acc:.4f}")
        print("Image-only training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['multimodal', 'text', 'image', 'all'], default='multimodal')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--text_model_name', type=str, default='sentence-transformers/LaBSE')
    parser.add_argument('--image_model_name', type=str, default='openai/clip-vit-base-patch32') # can also use sentence-transformers/LaBSE
    parser.add_argument('--lang', type=str, default='en') # en,hi,es
    parser.add_argument("--preprocess_text", action="store_true")
    args = parser.parse_args()
    main(args)
