import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Config ---
text_model_name = "xlm-roberta-base"
clip_model_name = "openai/clip-vit-base-patch32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8

# --- Load models ---
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).to(device)
clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# --- Multimodal Classifier ---
class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)
        x = torch.cat([t, i], dim=1)
        return self.classifier(x)

# Load trained model
model = MultimodalClassifier().to(device)
model.load_state_dict(torch.load(f"model_weights/10_multimodal_classifier.pt"))
model.eval()

# --- Embedding Functions ---
def get_text_embedding(texts):
    tokens = text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output = text_model(**tokens)
    return output.last_hidden_state[:, 0, :]  # CLS token

def get_image_embedding(images):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        output = clip_model(**inputs)
    return output.pooler_output

# --- Dataset Wrapper ---
class TorchDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(f"test_data/{item['image_path']}").convert("RGB")
        return {
            "text": item["tweet_text"],
            "image": image,
            "label": item["label"]
        }

# --- Collate Function ---
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long).to(device)
    text_embs = get_text_embedding(texts)
    image_embs = get_image_embedding(images)
    return text_embs, image_embs, labels

# --- Load dataset and wrap ---
full_data = load_dataset("QCRI/CrisisMMD", "damage", split="test")
test_data = full_data
test_data_torch = TorchDatasetWrapper(test_data)
test_loader = DataLoader(test_data_torch, batch_size=batch_size, collate_fn=collate_fn)

# --- Evaluate ---
correct = 0
total = 0
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for text_embs, image_embs, labels in tqdm(test_loader, desc="Evaluating"):
        outputs = model(text_embs, image_embs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# --- Metrics ---
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

print (all_preds)

print(f"✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔁 Recall:    {recall:.4f}")
print(f"📊 F1 Score:  {f1:.4f}")

