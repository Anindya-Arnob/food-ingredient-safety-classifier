#  Food Ingredient Safety Classifier

A Deep Learning NLP project that predicts whether a food product **Contains** or **Does Not Contain** allergens based on its ingredients.

##  Project Highlights
- Fine-tuned **DistilBERT** (Hugging Face Transformers)
- **Bidirectional LSTM** with Word Embeddings
- Achieved **90%+ accuracy** on test set
- Applies **Tokenization, Vector Embeddings, Transformer architecture**

##  Tech Stack
`Python` `TensorFlow` `PyTorch` `Hugging Face` `Scikit-learn` `Pandas`

##  Dataset
[Food Ingredients and Allergens](https://www.kaggle.com/datasets/uom190346a/food-ingredients-and-allergens) — 398 samples

##  Models Used
| Model | Architecture | Accuracy |
|-------|-------------|----------|
| BiLSTM | Embedding → BiLSTM × 2 → Dense | ~90%+ |
| DistilBERT | Pre-trained Transformer (Fine-tuned) | ~92%+ |

##  How to Run
```bash
git clone https://github.com/Anindya-Arnob/food-ingredient-safety-classifier.git
cd food-ingredient-safety-classifier
pip install -r requirements.txt
jupyter notebook Food_Ingredient_Safety_Classifier.ipynb
```

##  Key Concepts
- Tokenization (BERT Tokenizer + Keras Tokenizer)
- Word Embeddings & Vector Representations
- Transformer Architecture (DistilBERT)
- Transfer Learning & Fine-Tuning
- NLP Binary Classification