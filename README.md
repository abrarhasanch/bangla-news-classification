# Bangla News Classification (Machine Learning Project)

This project focuses on classifying Bangla news articles using machine learning techniques. The dataset contains **408,000+ Bangla news articles**, making it one of the largest Bangla text classification datasets used in research. The project applies extensive preprocessing, TF‑IDF vectorization, hybrid resampling, and multiple ML algorithms to identify the most effective model.

---

## 🚀 Project Overview

- Total dataset size: **408K+ Bangla news articles**
- Applied **Bangla-specific text preprocessing**
- Used **TF-IDF** for feature extraction
- Used hybrid sampling (**SMOTE + Random Undersampling**) for class balance
- Trained and evaluated multiple ML models
- Achieved **98.27% accuracy** with **Random Forest**

---

## 📌 Features

### 🔹 Bangla Text Preprocessing
- Stopword removal  
- Unicode normalization  
- Tokenization  
- Punctuation & symbol cleanup  
- Lemmatization/stemming (dataset-dependent)

### 🔹 Machine Learning Models Used
- Naive Bayes (NB)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF) — **Best performance**

### 🔹 Vectorization
- Term Frequency–Inverse Document Frequency (**TF-IDF**)

### 🔹 Resampling Techniques
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Random Undersampling**
- Combined hybrid technique to improve class distribution

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 93.12% |
| Logistic Regression | 96.45% |
| SVM | 97.02% |
| **Random Forest** | **98.27%** |

Random Forest delivered the most stable results across precision, recall, and F1-score metrics.

---

## 📁 Project Structure

```
Bangla-News-Classification/
│── data/                # Dataset (external due to size)
│── preprocessing/       # Text cleaning scripts
│── models/              # Training and evaluation scripts
│── results/             # Charts, confusion matrices, reports
│── bangla_news.ipynb    # Main notebook
│── README.md            # Documentation
```

---

## 🛠️ Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- imbalanced-learn (SMOTE)  
- TF-IDF Vectorizer  

---

## 📝 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the notebook:
```bash
jupyter notebook Bangla_Text_Classification.ipynb
```

3. Load dataset and execute cells step by step.

---

## 📄 Research Paper

Full paper available here:

**Paper Link:**  
https://drive.google.com/file/d/1szgVdBwoOgKhl6gaxxp_lOQGWymqThLw/view?usp=sharing

---

## 🔗 GitHub Repository

https://github.com/abrarhasanch/bangla-news-classification

---

## 📌 Future Improvements

- Deep learning models (BiLSTM, GRU, Transformer)  
- FastText and Word2Vec embeddings  
- Bangla BERT fine‑tuning  
- Multi-label classification  
- Deployment as a web API  

---

## 📜 License

This project is intended for research and educational purposes. Feel free to modify and extend.
