import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transformers import (
    MobileViTForImageClassification,
    MobileViTImageProcessor,
    get_scheduler,
)
from tqdm.auto import tqdm

DATA_ROOT      = Path("E:\\Classify")                
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 32  # Reduced for stability
NUM_WORKERS    = 2   # Reduced to avoid multiprocessing issues
NUM_EPOCHS     = 15
LEARNING_RATE  = 5e-4  # Slightly higher LR for smaller model
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0

# Set environment variables to reduce warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# MobileViT model variants - choose one:
MODEL_OPTIONS = {
    "xxs": "apple/mobilevit-xx-small",    # 1.3M params - Ultra lightweight
    "xs":  "apple/mobilevit-x-small",     # 2.3M params - Recommended
    "s":   "apple/mobilevit-small"        # 5.6M params - Best accuracy
}
MODEL_NAME     = MODEL_OPTIONS["xs"]  # Change to "xxs" or "s" as needed

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH= CHECKPOINT_DIR / "mobilevit_best.pt"

# make output directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def build_image_processor():
    """Returns a MobileViTImageProcessor."""
    return MobileViTImageProcessor.from_pretrained(MODEL_NAME)

def build_transforms(processor: MobileViTImageProcessor, is_training=False):
    """Build transforms with data augmentation for training."""
    
    # Get normalization values from processor
    # MobileViT uses ImageNet normalization by default
    mean = getattr(processor, 'image_mean', [0.485, 0.456, 0.406])
    std = getattr(processor, 'image_std', [0.229, 0.224, 0.225])
    
    # Base transforms
    base_transforms = [
        T.Resize((256, 256)),  # MobileViT typically uses 256x256
        T.CenterCrop(224),
    ]
    
    # Add augmentation for training
    if is_training:
        augment_transforms = [
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.RandomRotation(degrees=10),
        ]
        transforms_list = augment_transforms
    else:
        transforms_list = base_transforms
    
    # Add normalization
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    return T.Compose(transforms_list)

def get_dataloaders(processor: MobileViTImageProcessor):
    """ImageFolder → DataLoader with augmentation."""
    
    train_ds = ImageFolder(
        str(DATA_ROOT / "train"),
        transform=build_transforms(processor, is_training=True)
    )
    val_ds = ImageFolder(
        str(DATA_ROOT / "val"),
        transform=build_transforms(processor, is_training=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    return train_loader, val_loader, train_ds.classes

def create_model(num_labels):
    """Load pre-trained MobileViT and adapt the final layer."""
    model = MobileViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    processor = build_image_processor()
    train_loader, val_loader, class_names = get_dataloaders(processor)

    model = create_model(num_labels=len(class_names)).to(DEVICE)
    
    # Print model info
    param_count = count_parameters(model)
    print(f"📱 Model: {MODEL_NAME}")
    print(f"🔢 Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"📂 Classes: {len(class_names)} - {', '.join(class_names)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Cosine annealing scheduler - better for smaller models
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS * len(train_loader)
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        correct_train, total_train = 0, 0
        
        for batch in tqdm(train_loader, desc=f"[{epoch}/{NUM_EPOCHS}] Train", leave=False):
            imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            
            # Training accuracy
            preds = outputs.logits.argmax(-1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_acc = correct_train / total_train

        model.eval()
        correct, total = 0, 0
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                outputs = model(pixel_values=imgs, labels=labels)
                
                val_loss = outputs.loss
                val_losses.append(val_loss.item())
                
                preds = outputs.logits.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = sum(val_losses) / len(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:2d}] train_acc={train_acc*100:5.2f}% train_loss={avg_train_loss:.4f} | "
              f"val_acc={val_acc*100:5.2f}% val_loss={avg_val_loss:.4f} | lr={current_lr:.2e}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": class_names,
                    "model_name": MODEL_NAME,
                    "parameters": param_count,
                },
                CHECKPOINT_PATH,
            )
            print(f"  ✅ New best model (val_acc={val_acc*100:.2f}%) saved!")

    print(f"\n🎉 Training finished!")
    print(f"📊 Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"💾 Model saved to: {CHECKPOINT_PATH}")
    print(f"📱 Model size: {param_count/1e6:.1f}M parameters")

def evaluate(ckpt_path=CHECKPOINT_PATH):
    processor = build_image_processor()
    _, val_loader, class_names = get_dataloaders(processor)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model_name = ckpt.get("model_name", MODEL_NAME)
    
    model = MobileViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    correct, total = 0, 0
    all_preds, all_labels = [], []
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            imgs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(pixel_values=imgs)
            preds = outputs.logits.argmax(-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    acc = correct / total
    
    print(f"\n🎯 === MobileViT Evaluation Results ===")
    print(f"📱 Model: {model_name}")
    print(f"🔢 Parameters: {ckpt.get('parameters', 'unknown'):,}")
    print(f"📊 Overall Accuracy: {acc*100:.2f}%")
    print(f"📈 Training Accuracy: {ckpt.get('train_acc', 'unknown')}")
    print(f"🎯 Best Val Accuracy: {ckpt.get('val_acc', 'unknown')}")
    
    print(f"\n📋 Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

def main():
    global MODEL_NAME, CHECKPOINT_PATH  # Declare global variables first
    
    parser = argparse.ArgumentParser(description="MobileViT Image Classification")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--model", choices=["xxs", "xs", "s"], default="xs", 
                       help="Model size: xxs (1.3M), xs (2.3M), s (5.6M)")
    parser.add_argument("--ckpt", type=str, default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    # Update model name based on argument
    MODEL_NAME = MODEL_OPTIONS[args.model]
    CHECKPOINT_PATH = Path(f"checkpoints/mobilevit_{args.model}_best.pt")

    if args.mode == "train":
        train()
    else:
        evaluate(Path(args.ckpt))

if __name__ == "__main__":
    main()
