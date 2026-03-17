# mini-project-9

# NLP Content Moderation with DistilBERT

This project focuses on automated content moderation for social media text using natural language processing. The task is to classify tweets into three categories:

- Hate Speech
- Offensive Language
- Neither

The project compares a traditional machine learning baseline using **TF-IDF + Logistic Regression** with a transformer-based model using **DistilBERT**.

## Dataset

The project uses the **Twitter Hate Speech and Offensive Language Dataset**.  
Each tweet is classified into one of three classes:

- `0` = Hate Speech
- `1` = Offensive Language
- `2` = Neither

## Models Used

### 1. TF-IDF + Logistic Regression
A traditional text classification baseline using TF-IDF features and Logistic Regression.

**Test Results:**
- Accuracy: **0.8650**
- Macro F1: **0.7223**
- Weighted F1: **0.8738**

### 2. DistilBERT
A transformer-based model fine-tuned for 3-class tweet classification.

**Test Results:**
- Accuracy: **0.8687**
- Macro F1: **0.7341**
- Weighted F1: **0.8792**

## Key Observation

DistilBERT performed slightly better overall than the TF-IDF baseline, especially in overall class balance. However, both models found **Hate Speech** to be the most difficult class.

## Project Structure

```bash
project-folder/
│── notebook/
│   └── your_notebook.ipynb
│── .gitignore
│── README.md

```

## Authors

- Sepehr Mansouri  
- Tanishq Rawat
