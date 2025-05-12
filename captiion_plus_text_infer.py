import time
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

subset = "damage"
text_model_name = "xlm-roberta-base"

# --- Dataset Class for Inference ---
class CrisisTestDataset(Dataset):
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
        image = Image.open(f"test_data/{item['image_path']}").convert("RGB")
        text = item['tweet_text']
        label = item['label']

        # Generate image caption
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.caption_model.generate(**inputs)
        image_caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True)

        # Combine image caption and original text
        combined_text = f"{image_caption}. {text}"

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

# --- Inference Function ---
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    classifier_model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load dataset
    dataset = load_dataset("QCRI/CrisisMMD", subset, split="test")
    crisis_dataset = CrisisTestDataset(dataset, caption_processor, caption_model, tokenizer, device=device)
    dataloader = DataLoader(crisis_dataset, batch_size=args.batch_size, shuffle=False)

    classifier_model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    return accuracy, all_preds, all_labels

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    t1 = time.time()

    evaluate(args)

    print(f"Evaluation completed in {time.time() - t1:.2f} seconds.")
