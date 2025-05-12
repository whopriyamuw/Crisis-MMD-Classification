import multiprocessing as mp

from datasets import load_dataset, Image, DatasetDict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_NAME = "QCRI/CrisisMMD"


def download_file(row):
    return hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=row["image_path"],
        local_files_only=False,
    )


def download_images(dataset: DatasetDict):
    with mp.Pool() as pool:
        for split in dataset:
            paths = list(
                tqdm(
                    pool.imap(download_file, dataset[split]),
                    total=len(dataset[split]),
                    desc=f"Downloading {split} images",
                )
            )
            dataset[split] = dataset[split].remove_columns(["image"])
            dataset[split] = dataset[split].add_column("image", [p for p in paths])

    return dataset


def main():
    dataset = load_dataset(DATASET_NAME, "damage")
    dataset = download_images(dataset)
    dataset = dataset.cast_column("image", Image(decode=True))


if __name__ == "__main__":
    main()
