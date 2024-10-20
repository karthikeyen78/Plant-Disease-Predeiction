# Import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt  # For plotting

# Initializing the CNN
np.random.seed(1337)  # For reproducibility
classifier = Sequential()

# First convolutional layer and max pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer and max pooling
classifier.add(Conv2D(64, (3, 3), activation='relu'))  # Increased filters to 64 for more features
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer and max pooling
classifier.add(Conv2D(128, (3, 3), activation='relu'))  # Increased filters to 128 for more features
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection (hidden layer)
classifier.add(Dense(units=256, activation='relu'))  # Increased units to 256 for better performance
classifier.add(Dropout(rate=0.5))

# Output layer
classifier.add(Dense(units=10, activation='softmax'))  # Adjust output units based on your classes

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
print(classifier.summary())

# Part 2 - Preparing the dataset using ImageDataGenerator

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data preprocessing for the test set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training set
training_set = train_datagen.flow_from_directory(
    'train',  # Path to training dataset folder
    target_size=(128, 128),  # Resize all images to 128x128
    batch_size=64,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Print label mapping to verify
label_map = training_set.class_indices
print("Class indices (label map):", label_map)

# Load the validation set
test_set = test_datagen.flow_from_directory(
    'val',  # Path to validation dataset folder
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Part 3 - Training the CNN model and visualizing the results

history = classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,  # Adjust steps per epoch based on your dataset size
    epochs=50,  # Adjust epochs to a reasonable number (start with 20-100)
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size  # Adjust validation steps based on dataset size
)

# Saving the trained model and weights
classifier.save('keras_trained_model_weights.weights.h5')  # Save the entire model (architecture + weights)
print('Model saved as keras_trained_model.weights.h5')

# Part 4 - Plotting training results

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Show plots
plt.tight_layout()
plt.show()
# Train the model and save history
history = classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=100,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size
)

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()