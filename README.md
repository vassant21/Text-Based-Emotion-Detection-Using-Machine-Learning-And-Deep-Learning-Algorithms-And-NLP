# Emotion Classification Model Documentation

## Introduction
This documentation provides an overview of the emotion classification model implemented in Python using various machine learning and deep learning algorithms. The model is designed to classify the emotions present in textual data into one of five categories: "anger," "joy," "sadness," "fear," and "surprise." The model utilizes a combination of traditional machine learning algorithms such as Random Forest, Naive Bayes, and K-Nearest Neighbors (KNN), along with a deep learning model consisting of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers.

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Web Application for Real-Time Prediction](#web-application-for-real-time-prediction)
7. [Conclusion](#conclusion)

## 1. Requirements
- Python 3.x
- Libraries: seaborn, scikit-learn, numpy, pandas, matplotlib, nltk, keras, tensorflow, Flask

Install the required libraries using pip:
```bash
pip install seaborn scikit-learn numpy pandas matplotlib nltk keras tensorflow Flask
```
## 2. Dataset Description
The emotion classification model is trained and evaluated on three separate datasets:

Training data: Used to train the models.
Testing data: Used to evaluate the models' performance.
Validation data: Used for hyperparameter tuning and model optimization.
The data is provided in text format and is divided into two columns: "Text" (containing the textual data) and "Emotion" (the corresponding emotion label).

## 3. Data Preprocessing
The data preprocessing steps involve the following:

Removing duplicates from the training and validation datasets.
Normalizing the text data by converting it to lowercase, removing stop words, punctuation, numbers, and URLs, and applying lemmatization.

## 4. Feature Extraction
To convert textual data into numerical features, the following methods are used:

Term Frequency-Inverse Document Frequency (TF-IDF): For traditional machine learning models like Random Forest, Naive Bayes, and K-Nearest Neighbors.
Tokenization and Padding: For the CNN-LSTM deep learning model.
## 5. Model Training and Evaluation
The emotion classification models are trained and evaluated using the following techniques:

Random Forest Classifier: Achieved an accuracy of 92% on the test dataset.
Naive Bayes Classifier: Achieved an accuracy of 40% on the test dataset.
K-Nearest Neighbors (KNN) Classifier: Achieved an accuracy of 86% on the test dataset.
CNN-LSTM Model: A deep learning model with layers of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM). Achieved an accuracy of 90% on the test dataset.
The models are saved using the "pickle" library for traditional models and as an HDF5 file for the CNN-LSTM model.

## 6. Web Application for Real-Time Prediction
A Flask-based web application is developed to provide real-time emotion classification for input text. The application loads the pre-trained models and TF-IDF vectorizers and uses them to predict the emotion for the input sentence. The predicted emotions along with their probabilities are displayed as the output.

To run the web application, run the "app.py" file and open a web browser to the specified local host.

## 7 Conclusion
The emotion classification model effectively categorizes text into different emotions using both traditional machine learning algorithms and deep learning models. The web application provides a user-friendly interface to input text and get real-time emotion predictions.
example # Emotion Classification Model Documentation
