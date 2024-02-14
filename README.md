# Cyberbullying Detection in Bangla

This repository focuses on the development and evaluation of machine learning and deep learning models for cyberbullying detection in Bangla language text. The following methodologies and models are employed:

## Methodology

### A. Preprocessing
The initial step involves preparing the data for training and evaluation. Techniques such as text tokenization, stemming or lemmatization, and elimination of stop words and unnecessary letters are applied. The processed data is then split into training and testing sets.

### B. Feature Representation
Different algorithms require distinct feature representation methods. Classic algorithms like RF, SVM, DT, and LR use a bag-of-words or TF-IDF representation. Deep learning models like BERT, RNN, ANN, CNN, and BiLSTM benefit from word embeddings or contextual embeddings for capturing semantic connections within the text.

### C. Models
1. **Recurrent Neural Network (RNN):** Collects sequential information, overcoming gradient difficulties with versions like LSTM and BiLSTM.
2. **Artificial Neural Network (ANN):** The core of deep learning, adaptable to various tasks.
3. **Convolutional Neural Network (CNN):** Commonly used for image recognition, showing good results in text classification.
4. **Support Vector Machine (SVM):** A robust classification technique seeking hyperplanes for distinctive categorization.
5. **Logistic Regression (LR):** A linear model used for binary or multiclass classification.

### D. Evaluation
The performance of each algorithm is assessed using metrics such as accuracy, precision, recall, and F1-Score.

## Bangla Dataset

The Bangla dataset categorizes cyberbullying into four groups: "Troll," "Sexual," "Religious," and "Threat." The "Not Bully" category indicates material not fitting into any cyberbullying criteria. This dataset enables focused study and model development for multiple forms of cyberbullying in the Bangla language.

## Classification Report for Bangla Dataset

| Algorithm        | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| RNN              | 0.75     | 0.75      | 0.75   | 0.74     |
| ANN              | 0.64     | 0.63      | 0.64   | 0.63     |
| CNN              | 0.63     | 0.63      | 0.63   | 0.62     |
| SVM              | 0.57     | 0.56      | 0.57   | 0.55     |
| LR               | 0.56     | 0.55      | 0.56   | 0.54     |


## Streamlit GUI
[Link to Cyberbullying Detection App (Bangla)](https://amiruzzaman-cbbangla.hf.space/#cyberbullying-detection-app-bangla)

### Usage:
1. Enter Bangla text for cyberbullying detection.
2. The app provides predictions, cyberbullying type, bad words, and filtered text.

## Sample Texts
1. ভেবেছিলাম তুই একটা ছেলে!!! এখন দেখি এটা একটা হিজরা?
2. প্রতিটি নাটক কয়েকবার করে দেখা হয়ে গেছে

## Link to the Application
[Link to the Bangla Cyberbullying Detection App](https://amiruzzaman-cbbangla.hf.space/#cyberbullying-detection-app-bangla)
