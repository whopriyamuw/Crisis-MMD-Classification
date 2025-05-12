import argparse
import enum

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import pipeline


class TargetLanguage(enum.Enum):
    SPANISH = "spa_Latn"
    HINDI = "hin_Deva"


class HfModels(enum.Enum):
    COMET22_QE = "Unbabel/wmt22-cometkiwi-da"
    NLLB_600M = "facebook/nllb-200-distilled-600M"


class HfTranslator:
    def __init__(self, output_dataset: str):
        self.batch_size = 16
        self.output_dataset = output_dataset

        self._data_splits = None
        self._pipeline = None
        self._load_datasets()

    @staticmethod
    def _get_language_code(language: TargetLanguage) -> str:
        languages = {
            TargetLanguage.SPANISH: "es",
            TargetLanguage.HINDI: "hi",
        }

        return languages[language]

    def _load_pipeline(self, target_language: str):
        self._pipeline = pipeline(
            "translation",
            model=HfModels.NLLB_600M.value,
            device_map="auto",
            src_lang="eng_Latn",
            tgt_lang=target_language,
        )

    def _load_datasets(self):
        self._data_splits = load_dataset("QCRI/CrisisMMD", "damage")

    def _translate_batch(self, texts: list[str]) -> list[str]:
        outputs = self._pipeline(texts, max_length=1024)
        return [output["translation_text"].strip() for output in outputs]

    def _translate_split(self, split: str):
        split_data = self._data_splits[split]
        tweets = split_data["tweet_text"]

        translations = []
        for i in range(0, len(tweets), self.batch_size):
            batch = tweets[i : i + self.batch_size]
            translations.extend(self._translate_batch(batch))

        return translations

    def translate(self):
        splits = list(self._data_splits.keys())

        translated_datasets = {language: {} for language in TargetLanguage}
        for language in TargetLanguage:
            print(f"\nTranslating to {language.value}")
            self._load_pipeline(language.value)

            for split in tqdm(splits, desc="Processing splits", unit="split"):
                translated_datasets[language][split] = self._translate_split(split)

            del self._pipeline

        output = {}
        for split in splits:
            dataset = self._data_splits[split]
            for language in TargetLanguage:
                data = translated_datasets[language][split]
                column = f"tweet_text_{self._get_language_code(language)}"
                dataset = dataset.add_column(column, data)

            output[split] = dataset

        return output

    def save(self, translated_datasets: dict[str, Dataset]):
        print(f"\nPushing translated dataset to HuggingFace Hub: {self.output_dataset}")
        api = HfApi()

        try:
            api.create_repo(
                repo_id=self.output_dataset, repo_type="dataset", exist_ok=True
            )
            for split, data in translated_datasets.items():
                data.push_to_hub(
                    self.output_dataset,
                    split=split,
                    private=False,
                    embed_external_files=False,
                )
            print(
                f"Successfully uploaded to: https://huggingface.co/datasets/{self.output_dataset}"
            )
        except Exception as e:
            print(f"Failed to upload to HuggingFace Hub: {str(e)}")


def main(output_dataset: str) -> None:
    translator = HfTranslator(output_dataset=output_dataset)
    translations = translator.translate()
    translator.save(translations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate CrisisMMD[damage][tweet_text] to Spanish and Hindi"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="josecols/damage-mmd",
        help="HuggingFace Hub dataset ID to upload translated output",
    )

    args = parser.parse_args()
    main(output_dataset=args.dataset_id)
