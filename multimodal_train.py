import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# --- Config ---
text_model_name = "xlm-roberta-base"
clip_model_name = "openai/clip-vit-base-patch32"

t1 = time.time()
subset = "damage"
batch_size = 8
num_epochs = 10
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

# --- Load models ---
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).to(device)
clip_model = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# --- Dataset ---
# Replace with your actual dataset
full_data = load_dataset("QCRI/CrisisMMD", subset, split="train")
train_data = full_data
print (f"Number of training samples: {len(train_data)}")

# --- Preprocessing ---
def image_loader(path):
    image = Image.open(path).convert("RGB")
    return image

# --- Model ---
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

model = MultimodalClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# --- Embedding Functions ---
# def get_text_embedding(texts):
#     tokens = text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#     with torch.no_grad():
#         output = text_model(**tokens)
#     return output.last_hidden_state[:, 0, :]

def get_text_embedding(texts):
    tokens = text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    output = text_model(**tokens)
    return output.last_hidden_state[:, 0, :]  # CLS token


# def get_image_embedding(images):
#     processed = clip_processor(images=images, return_tensors="pt").to(device)
#     with torch.no_grad():
#         output = clip_model(**processed)
#     return output.pooler_output

def get_image_embedding(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
    output = clip_model(**inputs)
    return output.pooler_output  # shape: (batch_size, 768)


# --- Collate function ---
def collate_fn(batch):
    texts = [item["tweet_text"] for item in batch]
    images = [image_loader(item["image_path"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    text_embs = get_text_embedding(texts)
    image_embs = get_image_embedding(images)
    return text_embs, image_embs, labels

# --- DataLoader ---
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print (train_loader)

# --- Training Loop ---
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for text_embs, image_embs, labels in tqdm(train_loader):
        text_embs, image_embs, labels = text_embs.to(device), image_embs.to(device), labels.to(device)
        logits = model(text_embs, image_embs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# --- Save Model ---
torch.save(model.state_dict(), f"model_weights/{num_epochs}_multimodal_classifier.pt")
print (f"Training completed in {time.time() - t1:.2f} seconds.")
