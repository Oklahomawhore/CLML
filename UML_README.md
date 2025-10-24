# UML (Unpaired Multimodal Learning) Implementation for VILT

This document describes the implementation of **Unpaired Multimodal Learning (UML)** for the VILT model in continual learning settings, based on the paper ["Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models"](https://github.com/Sharut/Unpaired-Multimodal-Learning).

## Overview

UML leverages unpaired auxiliary modality data to improve unimodal model performance. The key idea is to train a model on both:
1. **Target modality data**: The main task dataset (e.g., images with real captions, questions with answers)
2. **Helper modality data**: Few-shot examples with class-label based text descriptions and COCO mean images

The model alternately processes inputs from different modalities while sharing parameters, allowing it to benefit from cross-modal structure without requiring explicit paired data.

## Architecture Changes

### 1. Helper Modality Datasets (`Data/uml_helper_datasets.py`)

Created specialized datasets that provide few-shot examples with:
- **Text descriptions**: Class-label based templates (e.g., "A photo of {class_name}")
- **COCO mean image**: A neutral average image from COCO dataset

Supported tasks:
- **SST-2**: "A statement of {negative/positive}"
- **SNLI-VE**: "A statement of {entailment/contradiction/neutral}"
- **VQA**: "A question of which the answer is {class_label}"
- **PIQA**: "A physical commonsense reasoning question, A {correct/false} answer to the question"
- **iNaturalist**: "A photo of {class_name}"
- **Places365**: "A photo of {scene_name}"

### 2. UML DataModules (`Data/uml_datamodules.py`)

Each UML datamodule returns a dictionary with two data loaders:
- `'target'`: Main task data loader
- `'helper'`: Helper modality data loader

The helper loader cycles through few-shot examples repeatedly during training.

### 3. Model Training (`Model/vilt_model.py`)

Modified `VILTLightningModule` to support UML training:
- Added `use_uml` flag to enable/disable UML mode
- Added `uml_alpha` parameter to weight helper modality loss
- Modified `training_step()` to:
  1. Compute loss on target modality batch
  2. Sample and compute loss on helper modality batch
  3. Combine losses: `L = L_target + alpha * L_helper`

### 4. Training Script Updates (`train.py`)

Updated the training pipeline to:
- Check for UML configuration parameters
- Use UML datamodules when `use_uml=True`
- Initialize helper loader iterator
- Pass UML parameters to the model

## Configuration

UML training is controlled via YAML configuration files. Three new parameters are added to the `training` section:

```yaml
training:
  # ... existing parameters ...
  
  # UML parameters
  use_uml: True          # Enable UML training
  uml_alpha: 1.0         # Weight for helper modality loss (recommended: 0.5-1.5)
  helper_num_shots: 16   # Number of few-shot examples per class (recommended: 8-32)
```

### Configuration Parameters (`definitions.py`)

Added to `TrainingArguments`:
- `use_uml` (bool): Enable UML training mode
- `uml_alpha` (float): Loss weighting coefficient for helper modality
- `helper_num_shots` (int): Number of few-shot examples per class in helper dataset

## Usage

### 1. Basic UML Training

Use the provided UML config files:

```bash
# SNLI-VE with UML
python train.py --config Config/uml_train_snlive.yaml

# iNaturalist with UML
python train.py --config Config/uml_train_iNaturalist.yaml

# PIQA with UML
python train.py --config Config/uml_train_piqa.yaml
```

### 2. Creating Custom UML Configurations

Start from an existing config and add UML parameters:

```yaml
_target_: definitions.MMCLArguments

training:
  _target_: definitions.TrainingArguments
  cur_dataset: "snlive"  # or "vqa", "piqa", "iNaturalist", "places365", "sst2"
  # ... other training parameters ...
  
  # Enable UML
  use_uml: True
  uml_alpha: 1.0
  helper_num_shots: 16

# ... datasets and model configuration ...
```

### 3. Hyperparameter Tuning

Key hyperparameters to tune:

- **`uml_alpha`**: Controls the contribution of helper modality
  - Start with 1.0
  - Increase (1.5-2.0) if target modality has very limited data
  - Decrease (0.3-0.7) if target modality is abundant
  
- **`helper_num_shots`**: Number of examples per class in helper dataset
  - 8-16: For datasets with many classes (e.g., iNaturalist, Places365)
  - 16-32: For datasets with fewer classes (e.g., SST-2, SNLI-VE, PIQA)

## Implementation Details

### Loss Computation

During training, each step computes:

```python
# Target modality (main task)
target_loss, target_acc = compute_loss(target_batch)

# Helper modality (few-shot examples)
helper_batch = next(helper_loader_iter)
helper_loss, helper_acc = compute_loss(helper_batch)

# Combined loss
total_loss = target_loss + alpha * helper_loss
```

### Helper Dataset Format

Each helper dataset example contains:
- **Encodings**: VILT processor output with COCO mean image + text description
- **Label**: Class label corresponding to the text description

Example for iNaturalist:
```python
{
    "text": "A photo of species_123",
    "image": coco_mean_image,
    "label": 123
}
```

### Data Loader Management

The helper loader iterator is managed by the model:
- Initialized before training starts
- Automatically resets when exhausted
- Stored in `model.helper_loader_iter`

## Expected Results

Based on the UML paper, you should expect:

1. **Improved accuracy**: 2-5% improvement on target modality tasks
2. **Better sample efficiency**: Especially noticeable in few-shot settings
3. **Wider inter-class margins**: More separable feature representations
4. **Cross-modal alignment**: Better alignment between modalities

## File Structure

```
ATLAS/
├── Data/
│   ├── uml_helper_datasets.py      # Helper modality datasets
│   ├── uml_datamodules.py          # UML-enabled datamodules
│   └── __init__.py                 # Updated exports
├── Model/
│   └── vilt_model.py               # Modified VILTLightningModule
├── Config/
│   ├── uml_train_snlive.yaml       # Example UML config for SNLI-VE
│   ├── uml_train_iNaturalist.yaml  # Example UML config for iNaturalist
│   └── uml_train_piqa.yaml         # Example UML config for PIQA
├── train.py                        # Updated training script
├── definitions.py                  # Added UML parameters
└── UML_README.md                   # This file
```

## Logging and Monitoring

UML training logs additional metrics:

- `train_losses`: Combined loss (target + alpha * helper)
- `train_target_loss`: Loss on target modality only
- `train_helper_loss`: Loss on helper modality only
- `train_acc`: Accuracy on target modality
- `train_helper_acc`: Accuracy on helper modality

Monitor these to ensure:
1. Both losses are decreasing
2. Helper loss doesn't dominate (adjust `uml_alpha` if needed)
3. Target accuracy is improving

## Troubleshooting

### Helper loss is too high
- Decrease `uml_alpha` (try 0.5 or 0.7)
- Increase `helper_num_shots` to provide more examples

### No improvement over baseline
- Increase `uml_alpha` (try 1.5 or 2.0)
- Check that helper dataset is being loaded correctly
- Verify COCO mean image exists at `Utils/coco_mean_image.png`

### Out of memory
- Reduce `helper_num_shots`
- Use gradient accumulation
- Reduce batch size

## References

1. [Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models](https://github.com/Sharut/Unpaired-Multimodal-Learning)
2. Original paper: Gupta et al., "Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models", 2025

## Citation

If you use this implementation, please cite both the original VILT model and the UML paper:

```bibtex
@article{gupta2025better,
  title={Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models},
  author={Gupta, Sharut and Sundaram, Shobhita and Wang, Chenyu and Jegelka, Stefanie and Isola, Phillip},
  journal={arXiv preprint},
  year={2025}
}
```



