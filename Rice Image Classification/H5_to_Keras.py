import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV3Small

# Paths
h5_path = 'C:\\Users\\DELL\\Desktop\\C++\\Rice\\best_rice_model.h5'
keras_path = 'C:\\Users\\DELL\\Desktop\\C++\\Rice\\best_rice_model.keras'

# Rebuild the model architecture (same as in your training code)
def build_transfer_learning_model(num_classes=5, input_shape=(224, 224, 3)):
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
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

# Build the model
num_classes = 5  # Adjust based on your dataset (e.g., 5 rice types)
model = build_transfer_learning_model(num_classes)

# Load weights from the .h5 file
try:
    model.load_weights(h5_path)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Compile the model (optional, since you're not training)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the model in the new .keras format
model.save(keras_path, save_format='keras_v3')
print(f"Converted to Keras format: {keras_path}")
print(f"You can now load it with tf.keras.models.load_model('{keras_path}')")