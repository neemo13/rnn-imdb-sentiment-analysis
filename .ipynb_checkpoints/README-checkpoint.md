# 🎬 IMDb Sentiment Analysis using RNN (PyTorch)

## 📌 Overview

This project performs **sentiment analysis** on movie reviews from the IMDb dataset.
The goal is to classify reviews as **positive** or **negative** using a **Recurrent Neural Network (RNN)** built with PyTorch.

---

## 📂 Dataset

* **Source:** IMDb Movie Reviews Dataset
* **Structure:**

| review      | sentiment           |
| ----------- | ------------------- |
| Text review | positive / negative |

Example:

```
"One of the other reviewers has mentioned that ..." → positive
```

---

## 🧹 Data Preprocessing

The text data undergoes several cleaning steps:

* Remove **HTML tags**
* Remove **URLs**
* Remove **punctuation**
* Convert text to **lowercase**
* **Tokenization** using NLTK
* Remove **stopwords**
* Apply **stemming** (Porter Stemmer)

### Libraries used:

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

---

## 🔤 Label Encoding

Sentiment labels are converted into numeric form:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])
```

* `positive → 1`
* `negative → 0`

---

## 🔢 Vectorization

Text data is converted into numerical tensors for model training.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
```

---

## 🧠 Model

* Model Type: **Recurrent Neural Network (RNN)**
* Framework: **PyTorch**
* Task: Binary classification (positive / negative)

---

## 🏋️ Training

* Total Epochs: **10**

### Training Loss:

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.2025 |
| 2     | 0.1352 |
| 3     | 0.2598 |
| 4     | 0.2019 |
| 5     | 0.2222 |
| 6     | 0.2855 |
| 7     | 0.1817 |
| 8     | 0.2006 |
| 9     | 0.2745 |
| 10    | 0.2767 |

---

## 📊 Performance

* ✅ **Accuracy:** **87.31%**

---

## ⚙️ Tech Stack

* Python
* Pandas
* NLTK
* Scikit-learn
* PyTorch

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas nltk scikit-learn torch
```

2. Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

3. Run preprocessing and training scripts.

---

## 📈 Future Improvements

* Use **LSTM/GRU** instead of vanilla RNN
* Add **pretrained embeddings (GloVe / Word2Vec)**
* Hyperparameter tuning
* Use **transformer models (BERT)** for better accuracy

---

## 💬 Notes

* Preprocessing plays a major role in performance.
* Stemming and stopword removal helped reduce noise.
* RNN works well but may struggle with long dependencies.

---

## 🏁 Conclusion

This project demonstrates a complete NLP pipeline:

* Text preprocessing
* Feature engineering
* Deep learning model training

Achieving **87% accuracy** shows solid performance for a baseline RNN model on IMDb data.

---
