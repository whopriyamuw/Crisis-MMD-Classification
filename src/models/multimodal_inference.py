import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :]

def get_image_embedding(images, processor, model):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    return output.pooler_output

# --- Dataset Utilities ---
class TorchDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, image_dir="test_data", lang="es"):
        self.dataset = hf_dataset
        self.image_dir = image_dir
        self.lang = lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(f"{self.image_dir}/{item['image_path']}").convert("RGB")
        return {
            "text": item[f"tweet_text_{args.lang}"],
            "image": image,
            "label": item["label"]
        }

def collate_fn(batch, text_tokenizer, text_model, clip_processor, clip_model):
    if args.preprocess_text:
        texts = [clean_tweet(item["text"]) for item in batch]
    else:
        texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long).to(device)
    text_embs = get_text_embedding(texts, text_tokenizer, text_model)
    image_embs = get_image_embedding(images, clip_processor, clip_model)
    return text_embs, image_embs, labels

def collate_fn_text_only(batch, text_tokenizer, text_model):
    if args.preprocess_text:
        texts = [clean_tweet(item["text"]) for item in batch]
    else:
        texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long).to(device)
    text_embs = get_text_embedding(texts, text_tokenizer, text_model)
    return text_embs, labels

def collate_fn_image_only(batch, clip_processor, clip_model):
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long).to(device)
    image_embs = get_image_embedding(images, clip_processor, clip_model)
    return image_embs, labels

# --- Evaluation Utilities ---
def evaluate_multimodal(model, dataloader):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for text_embs, image_embs, labels in tqdm(dataloader, desc="Evaluating Multimodal"):
            outputs = model(text_embs, image_embs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds

def evaluate_text_only(model, dataloader):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for text_embs, labels in tqdm(dataloader, desc="Evaluating Text-Only"):
            outputs = model(text_embs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds

def evaluate_image_only(model, dataloader):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for image_embs, labels in tqdm(dataloader, desc="Evaluating Image-Only"):
            outputs = model(image_embs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds

def print_metrics(labels, preds, prefix=""):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    print(f"{prefix}✅ Accuracy:  {accuracy * 100:.2f}%")
    print(f"{prefix}🎯 Precision: {precision:.4f}")
    print(f"{prefix}🔁 Recall:    {recall:.4f}")
    print(f"{prefix}📊 F1 Score:  {f1:.4f}")

# --- Main ---
def main(args):
    # Use arguments instead of hardcoded config
    text_model_name = args.text_model_name
    image_model_name = args.image_model_name
    batch_size = args.batch_size
    lang = args.lang

    # Load models and processors
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    # Dynamically determine text embedding dimension
    dummy_input = text_tokenizer("hello", return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_emb = text_model(**dummy_input).last_hidden_state[:, 0, :]
    text_dim = dummy_emb.shape[1]
    clip_model = CLIPModel.from_pretrained(image_model_name).vision_model.to(device)
    clip_processor = CLIPProcessor.from_pretrained(image_model_name)
    image_dim = 768  # CLIP ViT-base-patch32 default
    hidden_dim = 512

    # Load dataset
    full_data = load_dataset("josecols/damage-mmd", split="test")
    full_data = full_data.map(rename_text_field)
    test_data_torch = TorchDatasetWrapper(full_data, lang=lang)

    # Dataloaders
    test_loader = DataLoader(
        test_data_torch, batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, text_tokenizer, text_model, clip_processor, clip_model)
    )
    text_only_test_loader = DataLoader(
        test_data_torch, batch_size=batch_size,
        collate_fn=lambda batch: collate_fn_text_only(batch, text_tokenizer, text_model)
    )
    image_only_test_loader = DataLoader(
        test_data_torch, batch_size=batch_size,
        collate_fn=lambda batch: collate_fn_image_only(batch, clip_processor, clip_model)
    )

    if args.mode in ['multimodal', 'all']:
        # Evaluate Multimodal
        multimodal_model = MultimodalClassifier(text_dim=text_dim, image_dim=image_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        multimodal_model.load_state_dict(torch.load(f"model_weights/{lang}_multimodal_classifier.pt"))
        mm_labels, mm_preds = evaluate_multimodal(multimodal_model, test_loader)
        print("Multimodal predictions:", mm_preds)
        print_metrics(mm_labels, mm_preds)

    if args.mode in ['text', 'all']:
        # Evaluate Text-Only
        text_only_model = TextOnlyClassifier(text_dim=text_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        text_only_model.load_state_dict(torch.load(f"model_weights/{lang}_text_only_classifier.pt"))
        to_labels, to_preds = evaluate_text_only(text_only_model, text_only_test_loader)
        print("Text-only predictions:", to_preds)
        print_metrics(to_labels, to_preds, prefix="[Text-Only] ")

    if args.mode in ['image', 'all']:
        image_only_model = ImageOnlyClassifier(image_dim=image_dim, hidden_dim=hidden_dim, num_classes=3).to(device)
        image_only_model.load_state_dict(torch.load(f"model_weights/{lang}_image_only_classifier.pt"))
        io_labels, io_preds = evaluate_image_only(image_only_model, image_only_test_loader)
        print("Image-only predictions:", io_preds)
        print_metrics(io_labels, io_preds, prefix="[Image-Only] ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['multimodal', 'text', 'image', 'all'], default='multimodal', help='Which model(s) to evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--text_model_name', type=str, default='sentence-transformers/LaBSE') # can use xlm-roberta-base or sentence-transformers/LaBSE
    parser.add_argument('--image_model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP or other image model name')
    parser.add_argument('--lang', type=str, default='en', help='Language code for model weights')
    parser.add_argument("--preprocess_text", action="store_true", help="Whether to clean tweet text")
    args = parser.parse_args()
    main(args)
