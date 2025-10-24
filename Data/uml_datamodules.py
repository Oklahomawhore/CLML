"""
UML (Unpaired Multimodal Learning) DataModules

These datamodules provide both target modality and helper modality data loaders
for training with unpaired multimodal data.
"""

import torch
from typing import Any
from definitions import LowShotConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from Data.visionlanguage_datasets import SnliVeDataset_ViLT, VQADataset_ViLT
from Data.vision_datasets import iNaturalist2019_Dataset_VILT, Places365_Dataset_VILT
from Data.language_datasets import PIQA_Dataset_VILT, CommonsenseQA_VILT
from Data.uml_helper_datasets import (
    UMLHelperDataset_SST2,
    UMLHelperDataset_SNLIVE,
    UMLHelperDataset_VQA,
    UMLHelperDataset_PIQA,
    UMLHelperDataset_iNaturalist,
    UMLHelperDataset_Places365,
    uml_helper_collate_simple,
    uml_helper_collate_piqa
)


class UML_iNaturalistDataModule(LightningDataModule):
    """UML DataModule for iNaturalist with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.helper_num_shots = helper_num_shots
        
    def setup(self, stage=None):
        # Target modality dataset (main task)
        self.train_dataset = iNaturalist2019_Dataset_VILT(
            self.data_dir,
            split="train",
            low_shot_config=self.low_shot_config,
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer
        )
        self.valid_dataset = iNaturalist2019_Dataset_VILT(
            self.data_dir,
            split="val",
            low_shot_config=self.low_shot_config,
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer
        )
        
        # Helper modality dataset (text descriptions with COCO mean image)
        self.helper_dataset = UMLHelperDataset_iNaturalist(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            num_shots=self.helper_num_shots
        )
        
        self.collate_fn = self.train_dataset.iNatural2019_batch_collate
        
    def train_dataloader(self):
        # Return both target and helper data loaders
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Helper dataset is small, no need for workers
            collate_fn=uml_helper_collate_simple,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )


class UML_Places365DataModule(LightningDataModule):
    """UML DataModule for Places365 with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.helper_num_shots = helper_num_shots
    
    def setup(self, stage=None):
        self.train_dataset = Places365_Dataset_VILT(
            self.data_dir,
            split="train",
            low_shot_config=self.low_shot_config,
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer
        )
        self.valid_dataset = Places365_Dataset_VILT(
            self.data_dir,
            split="val",
            low_shot_config=self.low_shot_config,
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer
        )
        
        self.helper_dataset = UMLHelperDataset_Places365(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            num_shots=self.helper_num_shots
        )
        
        self.collate_fn = self.train_dataset.Places365_batch_collate
    
    def train_dataloader(self):
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=uml_helper_collate_simple,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )


class UML_PIQADataModule(LightningDataModule):
    """UML DataModule for PIQA with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.helper_num_shots = helper_num_shots
        
    def setup(self, stage=None):
        self.train_dataset = PIQA_Dataset_VILT(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            data_dir=self.data_dir,
            split="train",
            low_shot_config=self.low_shot_config
        )
        self.val_dataset = PIQA_Dataset_VILT(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            data_dir=self.data_dir,
            split="val",
            low_shot_config=self.low_shot_config
        )
        
        self.helper_dataset = UMLHelperDataset_PIQA(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            num_shots=self.helper_num_shots
        )
        
        self.collate_fn = self.train_dataset.piqa_batch_collate
    
    def train_dataloader(self):
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=uml_helper_collate_piqa,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: self.collate_fn(x),
            pin_memory=True
        )


class UML_SnliveDataModule(LightningDataModule):
    """UML DataModule for SNLI-VE with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int = 32,
                 test_batch_size: int = 32,
                 num_workers: int = 4,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True):
        super().__init__()
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.model_key = model_key
        self.data_root = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.low_shot_config = low_shot_config
        self.helper_num_shots = helper_num_shots
        
    def setup(self, stage=None):
        if self.model_key == "VILT":
            self.train_dataset = SnliVeDataset_ViLT(
                split="train",
                low_shot_config=self.low_shot_config,
                data_dir=self.data_root,
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer
            )
            self.val_dataset = SnliVeDataset_ViLT(
                split="dev",
                low_shot_config=self.low_shot_config,
                data_dir=self.data_root,
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer
            )
            
            self.helper_dataset = UMLHelperDataset_SNLIVE(
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer,
                num_shots=self.helper_num_shots
            )

    def train_dataloader(self):
        collate_fn = self.train_dataset.snlive_batch_collate_ViLT
        
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x)
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=uml_helper_collate_simple,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
    
    def val_dataloader(self):
        collate_fn = self.train_dataset.snlive_batch_collate_ViLT
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x)
        )


class UML_VQADataModule(LightningDataModule):
    """UML DataModule for VQA with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.helper_num_shots = helper_num_shots
        
    def setup(self, stage=None):
        if self.model_key == "VILT":
            self.train_dataset = VQADataset_ViLT(
                data_dir=self.data_dir,
                split="train",
                low_shot_config=self.low_shot_config,
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer
            )
            self.val_dataset = VQADataset_ViLT(
                data_dir=self.data_dir,
                split="val",
                low_shot_config=self.low_shot_config,
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer
            )
            
            self.helper_dataset = UMLHelperDataset_VQA(
                VILT_ckpt_dir=self.VILT_ckpt_dir,
                VILT_tokenizer=self.VILT_tokenizer,
                num_shots=self.helper_num_shots
            )
            
            self.collate_fn = self.train_dataset.vqa_batch_collate_ViLT
    
    def train_dataloader(self):
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda x: self.collate_fn(x)
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=uml_helper_collate_simple,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: self.collate_fn(x)
        )


class UML_SST2DataModule(LightningDataModule):
    """UML DataModule for SST-2 with helper modality"""
    
    def __init__(self,
                 low_shot_config: LowShotConfig,
                 model_key: str,
                 VILT_ckpt_dir: str,
                 VILT_tokenizer: str,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int,
                 helper_num_shots: int = 16,
                 allow_uneven_batches: bool = True,
                 **kwargs: Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches
        self.helper_num_shots = helper_num_shots
        
    def setup(self, stage=None):
        from Data.language_datasets.sst2_dataset import SST2_Dataset_VILT
        
        self.train_dataset = SST2_Dataset_VILT(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            data_dir=self.data_dir,
            split="train",
            low_shot_config=self.low_shot_config
        )
        self.val_dataset = SST2_Dataset_VILT(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            data_dir=self.data_dir,
            split="val",
            low_shot_config=self.low_shot_config
        )
        
        self.helper_dataset = UMLHelperDataset_SST2(
            VILT_ckpt_dir=self.VILT_ckpt_dir,
            VILT_tokenizer=self.VILT_tokenizer,
            num_shots=self.helper_num_shots
        )
        
    def train_dataloader(self):
        collate_fn = self.train_dataset.sst2_batch_collate
        
        target_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x)
        )
        
        helper_loader = DataLoader(
            dataset=self.helper_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=uml_helper_collate_simple,
            pin_memory=True
        )
        
        return {"target": target_loader, "helper": helper_loader}
    
    def val_dataloader(self):
        collate_fn = self.val_dataset.sst2_batch_collate
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x)
        )



