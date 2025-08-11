import os
import json  # Added missing import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.mixed_precision import set_global_policy
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

# Parameters
img_height, img_width = 224, 224  # Reduced size for speed
batch_size = 32  # Increased for GPU efficiency
train_dir = 'E:\\Rice_Image_Dataset'  # Update to your dataset path

# Create saved_models directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Lighter data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reduced
    width_shift_range=0.1,  # Reduced
    height_shift_range=0.1,  # Reduced
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create generators first
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get number of classes from the generator
num_classes = len(train_generator.class_indices)

train_dataset = train_generator
val_dataset = val_generator
test_generator = val_generator  # Reuse validation for testing

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

def build_transfer_learning_model(num_classes, input_shape=(224, 224, 3)):  # Fixed input shape
    """Build a lightweight transfer learning model using MobileNetV3Small"""
    base_model = MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model

    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),  # Reduced size
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', dtype='float32')  # Mixed precision compatibility
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build and compile model
model = build_transfer_learning_model(num_classes)
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('saved_models/best_rice_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train model
print("Training Model...")
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_dataset,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

# Fine-tuning
print("Fine-tuning Model...")
model.layers[1].trainable = True  # Unfreeze base model
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_finetune = model.fit(
    train_dataset,
    steps_per_epoch=len(train_generator),
    epochs=8,  # Fewer epochs for fine-tuning
    validation_data=val_dataset,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_history(history, 'MobileNetV3Small')
if history_finetune:
    plot_history(history_finetune, 'Fine-tuned MobileNetV3Small')

# Evaluate model
def evaluate_model(model, generator, model_name):
    print(f"\n=== {model_name} Evaluation ===")
    generator.reset()
    
    # Use generator directly for evaluation
    test_loss, test_accuracy = model.evaluate(generator, verbose=0)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Reset generator and get predictions
    generator.reset()
    predictions = model.predict(generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = generator.classes[:len(predicted_classes)]
    class_names = list(generator.class_indices.keys())
    
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return test_accuracy

accuracy = evaluate_model(model, test_generator, "MobileNetV3Small")

# Save model info
model_info = {
    'class_indices': train_generator.class_indices,
    'class_names': list(train_generator.class_indices.keys()),
    'img_height': img_height,
    'img_width': img_width,
    'num_classes': num_classes,
    'accuracy': float(accuracy)  # Ensure JSON serializable
}

with open('saved_models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print("\nModel information saved to 'saved_models/model_info.json'")
print(f"Best model saved to 'saved_models/best_rice_model.h5'")