# 💬 Emotion Detection from Text using NLP and Machine Learning

This project is a complete end-to-end pipeline for detecting human emotions in text data using classical Natural Language Processing (NLP) techniques and supervised Machine Learning models.

---

## 📌 Objective

To build a robust text classification system that can accurately detect the **emotional tone** (such as *happy, sad, anger, fear*, etc.) from English sentences using standard NLP preprocessing and ML classifiers like **Naive Bayes**, **Logistic Regression**, and **SVM**.

---

## 🔍 Problem Statement

Given a sentence or piece of text, the model should classify it into one of several predefined emotional categories. This has real-world applications in:
- Mental health analysis
- Customer feedback classification
- Chatbot emotion understanding
- Social media sentiment tracking

---

## 🧰 Tech Stack

- **Programming Language**: Python
- **NLP Libraries**: NLTK, Scikit-learn
- **Visualization**: Seaborn, Matplotlib
- **Web App**: Streamlit
- **Vectorization**: CountVectorizer, TF-IDF
- **Models**: Multinomial Naive Bayes, Logistic Regression, Support Vector Machine

---

## 🧪 Dataset

- **Source**: [train.txt] – a labeled dataset with sentences and their corresponding emotion tags.
- **Format**: `text;emotion`
- **Examples**:
  - `I am so excited for this trip! ; happy`
  - `I feel terrible about what happened. ; sad`

---

## 🛠️ Project Pipeline

```text
1. Data Loading and Exploration
2. Text Preprocessing
   - Lowercasing
   - Removing Punctuation, Numbers, Emojis, Stopwords
   - Tokenization
3. Label Encoding (emotion to number mapping)
4. Train-Test Split
5. Feature Extraction
   - CountVectorizer
   - TF-IDF
6. Model Building and Evaluation
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
7. Accuracy & Metrics (confusion matrix, accuracy score)
8. Streamlit Web App for User Input
```

---

## 📈 Model Performance

| Model                  | Vectorizer     | Accuracy   |
|------------------------|----------------|------------|
| Multinomial Naive Bayes| CountVectorizer| ~84–86%    |
| Logistic Regression    | TF-IDF         | ~87–89%    |
| SVM                    | TF-IDF         | ~89–90%    |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/emotion-nlp-ml.git
cd emotion-nlp-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Jupyter Notebook
jupyter notebook

# 4. Run Streamlit App
streamlit run app.py
```

---

## 🌐 Streamlit Web App

- Input your own sentence  
- See live emotion prediction  
- Visualize emojis, results, and confidence  
- 🎈 Includes optional Streamlit effects like balloons or party emojis after prediction!

---


## 🧠 Future Improvements

- Add more emotion classes (e.g., surprise, disgust)  
- Use BERT or transformer-based embeddings  
- Deploy with Flask or FastAPI backend  
- Add user feedback collection in app

---

