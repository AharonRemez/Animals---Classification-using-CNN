# Animal Image Classification Using Convolutional Neural Networks (CNN)
This project implements a Convolutional Neural Network (CNN) to classify images of animals into ten different categories. The model was trained on the Kaggle dataset: [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10), which includes images of dogs, cats, horses, butterflies, chickens, cows, sheep, spiders, and squirrels.

## Project Overview
The goal of this project is to develop a deep learning model capable of accurately classifying animal images into their respective categories. By utilizing a customized CNN architecture, the model was trained and achieved promising results in classifying images. This project can serve as a foundation for further applications in computer vision and image recognition.

## Model Performance
- **Training Accuracy:** 95.68%
- **Validation Accuracy:** 74.69%
- **Training Loss:** 0.1425
- **Validation Loss:** 1.1836
#### Accuracy and Loss Graphs

#### Confusion Matrix

## Configuration
- The model was trained with the following configuration:

```python
CONFIGURATION = {
    'VALIDATION_SPLIT': 0.2,  # 20% validation data
    'BATCH_SIZE': 64,         # 64 images per batch
    'IMG_HEIGHT': 300,        # Image height: 300 pixels
    'IMG_WIDTH': 300,         # Image width: 300 pixels
    'EPOCHS': 20,             # Number of epochs
    'CLASS_NAMES': ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
}
```
## Dataset
- The dataset used for training the model is available on Kaggle: [Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10). The dataset contains 26,179 images divided into ten animal categories.

- **To download the dataset, use the following command in your terminal:**

```python
pip install kaggle
kaggle datasets download -d alessiocorrado99/animals10
unzip animals10.zip
```
**Or in Google Colab:**
```python
!pip install kaggle
!kaggle datasets download -d alessiocorrado99/animals10
!unzip animals10.zip
```
## Preprocessing and Augmentation
- Images were resized to 300x300 pixels to match the input requirements of the model.
- Data was split into training (80%) and validation (20%) sets.
- Pixel values were normalized to range between 0 and 1 using Rescaling.
- Data was cached and prefetched to improve training performance.
## Model Architecture
- The model consists of alternating convolutional and MaxPooling layers, with a Dropout layer to prevent overfitting. Finally, the images are flattened and passed through dense layers for final classification.

```python
model = Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(CLASS_NAMES))  # Number of categories
])
```
- The model was compiled using the Adam optimizer and the SparseCategoricalCrossentropy loss function.

## Results and Evaluation
- The model achieved a validation accuracy of approximately 74.69% after 20 epochs.
- From the graphs, it is evident that the training accuracy is significantly higher than the validation accuracy, which may indicate overfitting.
- The confusion matrix helps identify which categories the model struggles to classify correctly.
