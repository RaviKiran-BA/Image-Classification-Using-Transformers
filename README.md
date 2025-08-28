## Image Classification Using Transformers and Rice Images Classification using Mobilenet

## Tranformers Training Dataset Link: https://www.kaggle.com/code/hassanraof/intel-image-classification/input
## Rice Image Dataset: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Transformer Model Accuracy: 93.06%
## Rice Model Accuracy: 95.27%

## Workflow of the Project:
1. Dataset Collection and Setup.
2. Data Preprocessing:
   - Uses MobileViTImageProcessor normalization.
   - Training augmentations:
   RandomResizedCrop(224)
   RandomHorizontalFlip
   ColorJitter
   RandomRotation
   - Validation: Resize → CenterCrop → Normalize.
3. Model Creation:
   - Choose from MobileViT variants:
   xxs → 1.3M params (ultra-lightweight)
   xs → 2.3M params (recommended)
   s → 5.6M params (higher accuracy)
   - Loads pretrained weights and adapts the classification head to match dataset classes.
4. Training:
   - Optimizer: AdamW with weight decay.
   - Scheduler: CosineAnnealingLR.
   - Training loop:
   Forward pass with images & labels.
   Compute loss (outputs.loss).
   Backpropagation + Gradient Clipping.
   Optimizer & scheduler step.
   - Metrics:
   Training Loss & Accuracy
   Validation Loss & Accuracy
   - Saves best checkpoint based on validation accuracy at:checkpoints/mobilevit_<variant>_best.pt
5. Evaluation:
   - Loads saved checkpoint & model weights.
   - Evaluates on validation set:
   Overall Accuracy
   Training Accuracy (from checkpoint)
   Best Validation Accuracy
   Per-Class Accuracy breakdown
6. Usage:
   Run for Training: python main.py --mode train --model xs
   Evaluate: python main.py --mode eval --ckpt checkpoints/mobilevit_xs_best.pt
7. Outputs
   - Model checkpoints in checkpoints/.
   - Training logs (printed in console).
   - Final results include:
   📊 Best validation accuracy
   📋 Per-class accuracy table
   🔢 Model size (parameters)
