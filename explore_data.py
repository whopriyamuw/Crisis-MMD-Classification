import os
import requests

from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt


def download_images():
    # Load the dataset
    dataset = load_dataset("QCRI/CrisisMMD", "damage", split="test")

    # Base URL for images
    base_url = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/"

    # Directory to save images
    save_dir = "data_image"

    for example in tqdm(dataset):
        image_path = example['image_path']  # e.g., 'data_image/hurricane_harvey/8_9_2017/905960092822003712_0.jpg'
        image_url = base_url + image_path
        local_path = f"test_data/{os.path.join(save_dir, *image_path.split('/')[1:])}"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download and save the image
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"Failed to download {image_url}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")

def plot_image(img_path):
    # Load the dataset

    # Load and display the image
    img = Image.open(img_path)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Image from CrisisMMD")
    plt.savefig("sample_image.png", bbox_inches='tight', dpi=300)

download_images()
dataset = load_dataset("QCRI/CrisisMMD", "damage", split="test")
img_path = dataset[0]['image_path']

plot_image(f"test_data/{img_path}")
