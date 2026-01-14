import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Dict, Optional


class CustomDataset(Dataset):
    def __init__(
        self,
        config: Dict,
        mode: str,
        tokenizer,
        label2id: Dict[str, int],
        max_len: int = 128,
        dedup_train: bool = True,
        return_sentence: bool = True,
    ):
        assert mode in {"train", "valid", "test"}
        self.mode = mode
        self.max_len = int(max_len)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.return_sentence = return_sentence

        path_key = {
            "train": "train_data_path",
            "valid": "valid_data_path",
            "test": "test_data_path",
        }[mode]
        data_path = config[path_key]

        df = pd.read_excel(
            data_path,
            engine="openpyxl",
            usecols=["句子序号", "句子", "情感"],
        )
        df = df.rename(columns={"句子序号": "id", "句子": "sentence", "情感": "sentiment"})

        if mode == "train" and dedup_train:
            df = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)

        self.df = df.reset_index(drop=True)

        self.sentences = self.df["sentence"]
        self.sentiments = self.df["sentiment"] if "sentiment" in self.df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sentence = str(self.sentences.iloc[idx])

        encoded = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0).to(torch.long)          # [L]
        attention_mask = encoded["attention_mask"].squeeze(0).to(torch.long)  # [L]

        label_str = None
        if self.sentiments is not None:
            label_str = self.sentiments.iloc[idx]

        label = self.label2id.get(label_str, -1)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.return_sentence:
            sample["sentence"] = sentence

        return sample


def print_dataset_info(dataset: CustomDataset, name: str = "Dataset"):
    print(f"\n{name} 样本总数: {len(dataset)}")

    if getattr(dataset, "sentiments", None) is None:
        print(f"{name} 无标签信息")
        return

    counter = Counter(dataset.sentiments.tolist())
    print(f"{name} 类别分布:")
    for k, v in counter.items():
        print(f"  {k}: {v}")
