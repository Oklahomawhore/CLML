# UML Quick Start Guide

## 1-Minute Setup

### Step 1: Verify Prerequisites

```bash
# Ensure COCO mean image exists
ls Utils/coco_mean_image.png

# If missing, you'll need to create it from COCO dataset
```

### Step 2: Run UML Training

```bash
# Choose a pre-configured task:

# Vision-Language Task (SNLI-VE)
python train.py --config Config/uml_train_snlive.yaml

# Vision Task (iNaturalist)
python train.py --config Config/uml_train_iNaturalist.yaml

# Language Task (PIQA)
python train.py --config Config/uml_train_piqa.yaml
```

### Step 3: Monitor Training

```bash
# View logs
tensorboard --logdir Checkpoint/

# Check metrics
cat Checkpoint/{dataset}/{expt_name}/metrics.csv
```

## Configuration Template

To enable UML on any existing config:

```yaml
training:
  # Add these three lines:
  use_uml: True
  uml_alpha: 1.0
  helper_num_shots: 16
```

## Tuning Guide

| Scenario | Recommended Settings |
|----------|---------------------|
| Few-shot learning | `alpha=1.5`, `shots=32` |
| Full dataset | `alpha=0.7`, `shots=16` |
| Many classes (>100) | `alpha=1.0`, `shots=8` |
| Few classes (<10) | `alpha=1.0`, `shots=32` |

## Expected Results

- **Accuracy gain**: +2-5% over baseline
- **Training time**: ~10-20% longer (two batches per step)
- **Memory usage**: +10-15% (dual loaders)

## Troubleshooting

```bash
# Problem: "COCO mean image not found"
# Solution: Check path in helper dataset files, should be "Utils/coco_mean_image.png"

# Problem: "No improvement over baseline"
# Solution: Increase uml_alpha to 1.5 or 2.0

# Problem: "Out of memory"
# Solution: Reduce helper_num_shots to 8 or batch_size
```

## Comparison Example

```bash
# Baseline training
python train.py --config Config/1_train_snlive.yaml

# UML training
python train.py --config Config/uml_train_snlive.yaml

# Compare final val_acc in metrics.csv
```

## Quick Test

Verify UML is working:

```python
# Check logs for these metrics:
# - train_losses (combined)
# - train_target_loss
# - train_helper_loss
# - train_helper_acc

# All should be present if UML is active
```

That's it! For detailed documentation, see `UML_README.md`.



