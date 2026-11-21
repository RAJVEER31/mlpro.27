# Image Classifier using CNN, Create an Image Classifier model to predict t present whether it is a cat or a dog in an image using Keras. In Part-1, perform data preprocessing and prepare data for feeding into the neural network. Also, create a baseline model for the same.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Set random seed for reproducibility
np.random.seed(42)
#folder is provided in the dataset, you can change it according to your dataset path
data_dir = 'C:/Users/jptech/Desktop/machine learning/ML_pro.26/cats and dogs data folder'
# Load and preprocess the data
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    for label in ['cats', 'dogs']:
        folder_path = os.path.join(data_dir, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(0 if label == 'cats' else 1)  # 0 for cats, 1 for dogs
    return np.array(images), np.array(labels)
# Load the data
images, labels = load_data(data_dir)
# training and testing data is already seperated into seperate folders, so we can directly use them for training and testing
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0
# Create a baseline CNN model
def create_baseline_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model
# Create the model
input_shape = (128, 128, 3)  # Image size and channels
model = create_baseline_model(input_shape)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Print the model summary
model.summary()
# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# Print the test accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Show the plots
plt.tight_layout()
plt.show()