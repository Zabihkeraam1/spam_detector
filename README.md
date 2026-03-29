# Spam Detector 📨

Detect spam messages using **Python** and **NLP techniques**.  
This project provides a full roadmap from raw emails/SMS to a working spam classification system.

---

## 🧭 1. Understand the Problem

Before building anything, clarify:

- **Spam** → ads, phishing, scams, etc.  
- **Not Spam (Ham)** → personal messages, important emails  

This is a **binary classification problem**:  
**Output → Spam (1) or Ham (0)**

---

## 📦 2. Collect & Prepare Data

### 📊 Dataset

You need labeled data:

- Spam emails / SMS  
- Legitimate (ham) emails / SMS  

Popular datasets:

- Enron Email Dataset  
- SMS Spam Collection Dataset  

### 🧹 Data Cleaning

Raw data is messy. You’ll need to:

- Remove HTML tags  
- Lowercase all text  
- Remove punctuation  
- Remove stopwords (e.g., the, is, and)  
- Handle special characters and numbers  

---

## 🔍 3. Text Preprocessing

Convert text into a form machines can understand:

**Techniques:**

- **Tokenization** → split sentences into words  
- **Stemming / Lemmatization** → reduce words to base form  
- **Stopword removal** → remove useless words  

---

## 🔢 4. Feature Extraction

Turn text into numbers so ML models can process it:

- Bag of Words (BoW)  
- TF-IDF (Term Frequency-Inverse Document Frequency)  
- Word embeddings (advanced)  

👉 At this stage, each message becomes a **numeric vector**.

---

## 🤖 5. Choose a Model

Start simple and beginner-friendly:

- **Naive Bayes** ⭐ (best for spam detection)  
- Logistic Regression  
- Decision Tree  

**Why Naive Bayes?**  
- Works very well with text  
- Fast and easy to understand  

---

## 🏋️ 6. Train the Model

- Split dataset: Training set (80%), Testing set (20%)  
- Train your chosen model on the training set  

---

## 📏 7. Evaluate Performance

Don’t just check accuracy! Use:

- Accuracy  
- Precision (important for spam)  
- Recall  
- F1-score  

👉 Why? You don’t want important emails marked as spam.

---

## ⚙️ 8. Improve the Model

Once the basic model works:

- Tune hyperparameters  
- Try different models  
- Improve preprocessing  
- Handle class imbalance (spam vs non-spam)  

---

## 🧠 9. Add Intelligence (Advanced Stage)

After mastering the basics:

- Use advanced NLP techniques  
- Deep learning models (LSTM, Transformers)  
- Detect phishing patterns, suspicious links, repeated phrases  

---

## 🔌 10. Build a Simple System

Create a pipeline:

1. Input email/SMS  
2. Clean text  
3. Convert text into features  
4. Predict using trained model  
5. Output: Spam / Not Spam  

---

## 🌐 11. (Optional) Deployment

Make your system real:

- Build an API using **FastAPI**  
- Connect to email client  
- Create a web dashboard  

---

## ⚡ Tech Stack

- Python 3.x  
- Pandas (data handling)  
- NLTK (text preprocessing)  
- Regex (text cleaning)  

---

## 📂 Project Structure

spam_detector/
├─ data/
│ ├─ raw/ # Raw datasets
│ └─ processed/ # Cleaned datasets
├─ src/
│ ├─ data_cleaning.py
│ ├─ train_model.py
│ └─ utils.py
├─ venv/ # Virtual environment
├─ .gitignore
└─ README.md

---

## 🚀 Usage

```bash
# Clone the repo
git clone git@github.com:Zabihkeraam1/spam_detector.git

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/data_cleaning.py

# Train model
python src/train_model.py