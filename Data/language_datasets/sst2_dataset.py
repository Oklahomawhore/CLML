import os
import json
import torch
import pickle
import pandas as pd
from definitions import LowShotConfig
from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class SST2_Dataset_VILT(Dataset):
    def __init__(self,
                 VILT_ckpt_dir,
                 VILT_tokenizer,
                 data_dir: str,
                 split: str,
                 low_shot_config=LowShotConfig,):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.max_text_length = 40
        
        if split == "train":
            if low_shot_config.key:
                data_path = os.path.join(data_dir, "sst2", "train-00000-of-00001.parquet")
                cached_data = os.path.join(data_dir, "sst2", "cached_data_file", f"sst2_train_{low_shot_config.num_low_shot}shot.pkl")
            else:
                data_path = os.path.join(data_dir, "sst2", "train-00000-of-00001.parquet")
                cached_data = os.path.join(data_dir, "sst2", "cached_data_file", "sst2_train.pkl")
        elif split == "val":
            data_path = os.path.join(data_dir, "sst2", "validation-00000-of-00001.parquet")
            cached_data = os.path.join(data_dir, "sst2", "cached_data_file", "sst2_val.pkl")
            
        self.mean_image = Image.open("Utils/coco_mean_image.png")
        self.mean_image = self.mean_image.convert('RGB')
        
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        if os.path.exists(cached_data):
            self.data = pickle.load(open(cached_data, "rb"))
        else:
            # Read parquet file using pandas
            df = pd.read_parquet(data_path)
            
            label_list = ["0", "1"]  # 0: negative, 1: positive
            label_map = {label: int(label) for label in label_list}
            
            self.data = []
            if low_shot_config.key and split == "train":
                # For low-shot learning, balance the classes
                num_class = {0: 0, 1: 0}
                for idx, row in df.iterrows():
                    text = str(row['sentence'])
                    label = int(row['label'])
                    
                    labeled_data = {
                        "example_id": idx,
                        "text": text,
                        "label": label,
                        "description": "Sentiment Analysis; Binary Classification"
                    }
                    
                    if num_class[label] < low_shot_config.num_low_shot:
                        self.data.append(labeled_data)
                        num_class[label] += 1
            else:
                for idx, row in df.iterrows():
                    text = str(row['sentence'])
                    label = int(row['label'])
                    
                    labeled_data = {
                        "example_id": idx,
                        "text": text,
                        "label": label,
                        "description": "Sentiment Analysis; Binary Classification"
                    }
                    self.data.append(labeled_data)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cached_data), exist_ok=True)
            pickle.dump(self.data, open(cached_data, "wb"))
        
        self.n_samples = len(self.data)
        
    def __getitem__(self, index):
        batch = self.data[index]
        
        example_id = batch["example_id"]
        text = batch["text"]
        label = batch["label"]
        
        return {
            "example_id": example_id,
            "text": text,
            "label": label
        }
        
    def __len__(self):
        return self.n_samples
        
    def sst2_batch_collate(self, batch: List[Dict]):
        """Collates each model input for all batch items into a single model input."""
        
        example_id = [x["example_id"] for x in batch]
        text = [x["text"] for x in batch]
        
        # For SST-2, we use the text as both text_a and text_b (single sentence classification)
        text_pairs = [[t, t] for t in text]  # Duplicate text for VILT format
        
        image = [self.mean_image for _ in batch]
        
        encodings = self.processor(images=image,
                                   text=text_pairs,
                                   padding=True,
                                   max_length=self.max_text_length,
                                   truncation=True,
                                   return_tensors='pt')
        
        label = [x["label"] for x in batch]
        label = torch.tensor(label)
        
        return {
            "encodings": encodings,
            "labels": label
        }
