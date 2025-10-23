"""
Helper Modality Datasets for UML (Unpaired Multimodal Learning)

These datasets provide few-shot examples with class-label based text descriptions 
and COCO mean images for auxiliary modality training.
"""

import os
import torch
from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class UMLHelperDataset_SST2(Dataset):
    """Helper modality dataset for SST-2 sentiment analysis task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 40
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # Class labels for SST-2: 0=negative, 1=positive
        self.class_names = ["negative", "positive"]
        
        # Create few-shot examples with label-based descriptions
        self.data = []
        for label_idx, class_name in enumerate(self.class_names):
            for _ in range(num_shots):
                self.data.append({
                    "text": f"A statement of {class_name}",
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        # Encode with COCO mean image
        encodings = self.processor(
            images=self.mean_image,
            text=text,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": {k: v.squeeze(0) for k, v in encodings.items()},
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


class UMLHelperDataset_SNLIVE(Dataset):
    """Helper modality dataset for SNLI-VE entailment task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 40
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # Class labels for SNLI-VE: 0=entailment, 1=contradiction, 2=neutral
        self.class_names = ["entailment", "contradiction", "neutral"]
        
        # Create few-shot examples
        self.data = []
        for label_idx, class_name in enumerate(self.class_names):
            for _ in range(num_shots):
                self.data.append({
                    "text": f"A statement of {class_name}",
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        encodings = self.processor(
            images=self.mean_image,
            text=text,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": {k: v.squeeze(0) for k, v in encodings.items()},
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


class UMLHelperDataset_VQA(Dataset):
    """Helper modality dataset for VQA task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16, vqa_label_list=None):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 40
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # For VQA, we use top-k most common answers
        # If not provided, use a small subset
        self.class_names = vqa_label_list[:100] if vqa_label_list else ["yes", "no", "1", "2", "3"]
        
        # Create few-shot examples
        self.data = []
        for label_idx, class_name in enumerate(self.class_names):
            for _ in range(num_shots):
                self.data.append({
                    "text": f"A question of which the answer is {class_name}",
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        encodings = self.processor(
            images=self.mean_image,
            text=text,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": {k: v.squeeze(0) for k, v in encodings.items()},
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


class UMLHelperDataset_PIQA(Dataset):
    """Helper modality dataset for PIQA physical commonsense reasoning task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 40
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # For PIQA: binary classification (correct vs incorrect answer)
        self.class_descriptions = [
            "A physical commonsense reasoning question, A correct answer to the question",
            "A physical commonsense reasoning question, A false answer to the question"
        ]
        
        # Create few-shot examples
        self.data = []
        for label_idx, description in enumerate(self.class_descriptions):
            for _ in range(num_shots):
                # For PIQA, we need to create pairs
                self.data.append({
                    "text": description,
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        # For PIQA with binary choice, create two text pairs
        text_pairs = [text, text]  # Simplified version
        
        encodings = self.processor(
            images=[self.mean_image, self.mean_image],
            text=text_pairs,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": encodings,
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


class UMLHelperDataset_iNaturalist(Dataset):
    """Helper modality dataset for iNaturalist image classification task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16, class_names=None):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 20
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # For iNaturalist, use provided class names or a subset
        # In practice, you'd load the actual class names from the dataset
        if class_names is None:
            # Use a small subset for demonstration
            self.class_names = [f"species_{i}" for i in range(100)]
        else:
            self.class_names = class_names[:100]  # Use top 100 classes for efficiency
        
        # Create few-shot examples
        self.data = []
        for label_idx, class_name in enumerate(self.class_names):
            for _ in range(num_shots):
                self.data.append({
                    "text": f"A photo of {class_name}",
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        encodings = self.processor(
            images=self.mean_image,
            text=text,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": {k: v.squeeze(0) for k, v in encodings.items()},
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


class UMLHelperDataset_Places365(Dataset):
    """Helper modality dataset for Places365 scene classification task"""
    
    def __init__(self, VILT_ckpt_dir, VILT_tokenizer, num_shots=16, class_names=None):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir, "vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 20
        
        # Load COCO mean image
        self.mean_image = Image.open("Utils/coco_mean_image.png").convert('RGB')
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        # For Places365, use provided class names
        if class_names is None:
            self.class_names = [f"scene_{i}" for i in range(100)]
        else:
            self.class_names = class_names[:100]
        
        # Create few-shot examples
        self.data = []
        for label_idx, class_name in enumerate(self.class_names):
            for _ in range(num_shots):
                self.data.append({
                    "text": f"A photo of {class_name}",
                    "label": label_idx
                })
        
        self.n_samples = len(self.data)
    
    def __getitem__(self, index):
        batch = self.data[index]
        text = batch["text"]
        label = batch["label"]
        
        encodings = self.processor(
            images=self.mean_image,
            text=text,
            padding=True,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "encodings": {k: v.squeeze(0) for k, v in encodings.items()},
            "label": label
        }
    
    def __len__(self):
        return self.n_samples


# Collate functions for helper datasets
def uml_helper_collate_simple(batch: List[Dict]):
    """Collate function for simple single-input helper datasets (SST2, SNLIVE, iNat, Places365)"""
    encodings = {}
    
    # Stack all encoding tensors
    for key in batch[0]["encodings"].keys():
        encodings[key] = torch.stack([x["encodings"][key] for x in batch])
    
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    
    return {
        "encodings": encodings,
        "labels": labels
    }


def uml_helper_collate_piqa(batch: List[Dict]):
    """Collate function for PIQA helper dataset"""
    encodings = {}
    
    # For PIQA, encodings are already batched from processor
    # Just concatenate across batch dimension
    for key in batch[0]["encodings"].keys():
        encodings[key] = torch.cat([x["encodings"][key] for x in batch], dim=0)
    
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    
    return {
        "encodings": encodings,
        "labels": labels
    }

