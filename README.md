# 🐶🐱 Dogs vs Cats Classifier using CNN (TensorFlow + Google Colab)

Welcome to the **Dogs vs Cats Classifier**! This project shows how to build an effective **Convolutional Neural Network (CNN)** using **TensorFlow** to classify images of dogs and cats. It leverages cutting-edge deep learning techniques to train a model capable of recognizing these two iconic animals in images.

The dataset used is the famous **Dogs vs Cats dataset** from Kaggle. Whether you're a beginner or seasoned AI enthusiast, this project offers great insights into image classification tasks!

## 🧑‍💻 Project Overview

This repository contains a **Google Colab notebook** where you can train a CNN model to classify images of dogs and cats. The dataset is downloaded directly from Kaggle using the **Kaggle API**, and the images are processed and normalized for optimal training.

### What makes this project cool:
- The **Dogs vs Cats dataset** is an excellent resource for training image classification models.
- The notebook handles the entire pipeline — from data download to training and evaluating the CNN model.
- You will learn how to work with **TensorFlow**, **Keras**, and **OpenCV**.

### The model architecture:
- **Convolutional Layers (Conv2D)** to extract important features from the images.
- **MaxPooling Layers** to downsample and retain significant information.
- **Fully Connected Layers (Dense)** for decision-making.
- **Dropout Layers** to reduce overfitting and increase model generalization.

## 🚀 Key Steps in the Project

### 1. **Dataset Download:**
   - Fetches the dataset directly from Kaggle using the **Kaggle API**. No manual downloading required!
   - The images are then **unzipped** for use in training and validation.

### 2. **Image Preprocessing:**
   - Images are resized to 256x256 pixels.
   - **Normalization**: Images are scaled to a range of [0, 1] for faster convergence during training.

### 3. **CNN Model Architecture:**
   - The model consists of several convolutional layers to extract features from the images.
   - Pooling layers help reduce spatial dimensions and focus on the most important information.
   - Fully connected layers at the end make predictions, classifying images as either **Dog** or **Cat**.

### 4. **Training the Model:**
   - Trains the CNN model for 10 epochs using **binary cross-entropy loss**.
   - The **Adam optimizer** is used to minimize the loss function.
   - The performance of the model is validated on a separate test set of images.

### 5. **Visualization:**
   - **Training and validation accuracy** and **loss** are plotted for better insight into the model's learning curve.
   
### 6. **Prediction:**
   - Once trained, the model can predict whether a given image is a **dog** or a **cat**.

## 📊 Example Training Output

Here’s a sneak peek into the model’s performance across 10 epochs:

| Epoch | Accuracy  | Validation Accuracy |
|-------|-----------|---------------------|
| 1     | 56.03%    | 72.58%              |
| 5     | 92.48%    | 78.82%              |
| 10    | 98.36%    | 79.10%              |

### 💡 *Note*:
The validation accuracy plateaus at later epochs due to **overfitting**. Overfitting occurs when the model performs exceptionally well on the training data but struggles to generalize to unseen validation data. To mitigate this:
- **Regularization** techniques, such as **dropout**, are used in the model to prevent overfitting and ensure better performance on new data.

## ✨ Features

- **Seamless Kaggle API Integration**: Automatically download the dataset without leaving the notebook.
- **Data Preprocessing**: Images are resized and normalized for faster training.
- **CNN Model**: Build a deep learning model from scratch with layers like **Conv2D**, **MaxPooling2D**, **Dense**, and **Dropout**.
- **Model Training & Evaluation**: Track accuracy, loss, and validation performance in real time.
- **Image Prediction**: Test your model with new images and see predictions in action!

## 📚 Dataset Information

The dataset used for this project is the **Dogs vs Cats** dataset from Kaggle. It contains 25,000 images of cats and dogs. The dataset is divided into:
- **20,000 images** for training.
- **5,000 images** for testing and validation.

- **Dataset URL**: [Dogs vs Cats Dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **License**: Unknown (Please check Kaggle for details).

## 🛠 Requirements

To run the notebook, you'll need the following Python libraries:
- **TensorFlow**: For building and training the CNN model.
- **Keras**: For the neural network layers and utilities.
- **OpenCV**: For image processing and manipulation.
- **Matplotlib**: For visualizing the training process and metrics.

You can install these libraries using pip:

```bash
pip install tensorflow opencv-python matplotlib
```
## 🚀 Usage Instructions

- Clone or open this repository in **Google Colab**.
- Upload your **Kaggle API Key** (`kaggle.json`) to Google Colab to enable dataset access.
- Run the notebook to download the dataset, preprocess the images, and train the CNN model.
- You can use the trained model to predict whether new images contain a **dog** or **cat**.

---

## ⚖ License

The dataset license is unknown, as it is hosted on **Kaggle**.  
Please check the [Kaggle page](https://www.kaggle.com/datasets/salader/dogs-vs-cats) for more details.

---

## 🙏 Acknowledgments

- [Kaggle - Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **TensorFlow/Keras** for building the deep learning model.
- **Google Colab** for providing an easy-to-use environment for training models.
- The **Kaggle community** for providing such a great dataset!

