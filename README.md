# 🍽️ Food Ingredient Safety Classifier

> Predicting whether a food product **Contains** or **Does Not Contain** allergens using Deep Learning NLP — BiLSTM and Fine-tuned DistilBERT.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen?style=flat)

---

## 📌 Overview

Food allergen safety is a critical public health concern. This project builds an **end-to-end Deep Learning NLP pipeline** that classifies food products as safe or potentially harmful based on their ingredients and allergen information.

Two models are trained and compared:
- A custom **Bidirectional LSTM** with learned word embeddings
- A **fine-tuned DistilBERT** transformer model via Hugging Face

📥 **Dataset:** [Food Ingredients and Allergens — Kaggle](https://www.kaggle.com/datasets/uom190346a/food-ingredients-and-allergens) — 398 samples

---

## ✨ Project Highlights

- 🤗 Fine-tuned **DistilBERT** using Hugging Face Transformers
- 🧠 **Bidirectional LSTM** with Word Embeddings from scratch
- 🎯 Achieved **90%+ accuracy** on test set
- 🔤 Demonstrates **Tokenization, Vector Embeddings, Transformer architecture**
- 📊 Full EDA with visualizations (class distribution, allergen frequency)
- 🔮 Live prediction demo on custom food inputs

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras, PyTorch |
| **NLP / Transformers** | Hugging Face Transformers (DistilBERT) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 📊 Dataset & Features

| Column | Description |
|--------|-------------|
| `Food Product` | Name of the food item |
| `Main Ingredient` | Primary ingredient |
| `Sweetener` | Sweetener used (if any) |
| `Fat/Oil` | Fat or oil used (if any) |
| `Seasoning` | Seasoning used (if any) |
| `Allergens` | List of allergens present |
| `Prediction` | 🎯 Target: `Contains` / `Does not contain` |

> All 6 text columns are combined into one rich input feature for maximum NLP context.

---

## 🧠 Models Used

| Model | Architecture | Test Accuracy |
|-------|-------------|---------------|
| **BiLSTM** | Embedding → BiLSTM × 2 → BatchNorm → Dense | ~90%+ |
| **DistilBERT** | Pre-trained Transformer (Fine-tuned) | ~92%+ |

---

## 🔄 Workflow
### 1️⃣ Data Preprocessing
- Fill missing values with `none`
- Combine all 6 columns into one rich text feature
- Clean and normalize text
- Encode target: `Contains = 1`, `Does not contain = 0`
- Stratified Train / Validation / Test split (70 / 15 / 15)

### 2️⃣ Model 1 — Bidirectional LSTM
- Keras `Tokenizer` → token ID sequences → `pad_sequences`
- `Embedding` layer → learned word vectors
- `Bidirectional LSTM × 2` → `BatchNormalization` → `Dense`
- EarlyStopping + ReduceLROnPlateau callbacks

### 3️⃣ Model 2 — DistilBERT (Fine-tuned)
- `DistilBertTokenizer` → `input_ids` + `attention_mask`
- Custom `PyTorch Dataset` class
- `DistilBertForSequenceClassification` fine-tuned with Hugging Face `Trainer`
- Warmup steps + weight decay for stable training

### 4️⃣ Evaluation
- Classification Report (Precision, Recall, F1)
- Confusion Matrix heatmap
- Side-by-side model comparison bar chart

---

## 🚀 How to Run

```bash
git clone https://github.com/Anindya-Arnob/food-ingredient-safety-classifier
cd food-ingredient-safety-classifier
pip install -r requirements.txt
jupyter notebook Food_Ingredient_Safety_Classifier.ipynb
```

> Make sure `food_ingredients_and_allergens.csv` is in the same folder as the notebook.

---

## 🔮 Live Prediction Example

```python
predict_bert(
    food_product    = "Peanut Butter",
    main_ingredient = "peanuts",
    fat_oil         = "palm oil",
    seasoning       = "salt",
    allergens       = "peanuts"
)
# Output → Contains Allergen ⚠️  (confidence: 97.3%)
```

---

## 📁 Project Structure
---

## 🔑 Key Concepts Demonstrated

- **Tokenization** — Keras Tokenizer & BERT Tokenizer
- **Word Embeddings** — Dense vector representations of ingredients
- **Transformer Architecture** — DistilBERT attention mechanism
- **Transfer Learning & Fine-Tuning** — Adapting pre-trained BERT to food safety domain
- **Bidirectional LSTM** — Sequential ingredient pattern learning
- **NLP Binary Classification** — Safe vs. Unsafe food detection

---

## 📈 Results

| Metric | BiLSTM | DistilBERT |
|--------|--------|------------|
| Accuracy | ~90%+ | ~92%+ |
| F1 Score | ~90%+ | ~92%+ |
| Precision | ~90%+ | ~92%+ |
| Recall | ~90%+ | ~92%+ |

---

## 📜 License

This project is open-source. For inquiries or contributions, please open an issue in the repository.

---

⭐ **If you found this project helpful, please give it a star!**
