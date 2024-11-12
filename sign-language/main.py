import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Model definition
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Parameters
input_shape = (64, 64, 3)  # Input shape
num_classes = 26  # A-Z
batch_size = 32
image_dir = r'C:\Users\hp\Documents\Assignment_Series\ML\sign-language-alphabet-recognizer\dataset'

# Data loading and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80% training, 20% validation
train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create and compile model
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Display a few sample predictions
class_names = list(train_generator.class_indices.keys())
plt.figure(figsize=(15, 6))

for idx, (images, labels) in enumerate(validation_generator):
    if idx >= 5:  # Limit to display 5 images
        break

    image = images[0]
    label = np.argmax(labels[0])
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)
    predicted_class_name = class_names[predicted_label]

    plt.subplot(1, 5, idx + 1)
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_class_name}')
    plt.axis('off')

plt.tight_layout()
plt.show()
