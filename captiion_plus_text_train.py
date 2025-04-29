import time
import argparse
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

subset = "damage"
text_model_name = "xlm-roberta-base"

# --- Dataset Class ---
class CrisisDataset(Dataset):
    def __init__(self, data, vision_processor, caption_model, text_tokenizer, max_length=128, device='cpu'):
        self.data = data
        self.vision_processor = vision_processor
        self.caption_model = caption_model.to(device)
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        text = item['tweet_text']
        label = item['label']

        # Generate image caption
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.caption_model.generate(**inputs)
        image_caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True)


        # Combine original text and image caption
        combined_text = f"Image is of - {image_caption}. The tweet attached is - {text}"

        encoded = self.text_tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

# --- Training Function ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f"Using device: {device}")

    # Load vision model (BLIP) for captioning
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load text classification model: XLM-Roberta
    classifier_model = AutoModelForSequenceClassification.from_pretrained(text_model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    classifier_model = classifier_model.to(device)

    # Load your dataset
    dataset = load_dataset("QCRI/CrisisMMD", subset, split="train")

    crisis_dataset = CrisisDataset(dataset, caption_processor, caption_model, tokenizer, device=device)
    dataloader = DataLoader(crisis_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(classifier_model.parameters(), lr=args.lr)

    classifier_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = classifier_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    classifier_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./saved_classifier2")
    args = parser.parse_args()

    t1 = time.time()
    
    train(args)

    print (f"Training completed in {time.time() - t1:.2f} seconds.")
