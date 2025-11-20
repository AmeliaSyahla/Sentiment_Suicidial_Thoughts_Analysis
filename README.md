# Indonesian Sentiment Analysis 

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)
![Status](https://img.shields.io/badge/status-ongoing-yellow.svg)

A machine learning sentiment analysis for Indonesian text, built with Streamlit and scikit-learn.

</div>

---

## 🎯 About The Project

This project is a **sentiment analysis application** specifically designed for **Indonesian language** text. It analyzes user input and classifies the sentiment as **Positive**, **Negative**, or **Neutral** with confidence scores.

- **Indonesian Language Focus**: Specifically tailored for Indonesian text preprocessing
- **Machine Learning**: Uses Logistic Regression with TF-IDF vectorization
- **Real-time Analysis**: Instant sentiment prediction with confidence scores
- **Comprehensive Preprocessing**: Includes slang normalization, stemming, and stopword removal

---

## ✨ Features

### Current Features 
- [x] **Three-class Classification** - Positive, Negative, and Neutral sentiments
- [x] **Confidence Scores** - View probability distribution across all classes
- [x] **Text Preprocessing Pipeline**
  - Twitter-specific cleaning (mentions, hashtags, URLs)
  - Emoji removal
  - Slang normalization
  - Stopword removal
  - Stemming (Sastrawi)
  - Capital ratio feature extraction

---

## 🎬 Demo on Local

<img src="Assets/demo.png" width="max">


---

## 🛠️ Technology Stack

### Core Technologies

- **Python 3.8+** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **NLTK** - Natural language processing

### NLP Libraries

- **Sastrawi** - Indonesian stemmer
- **TextBlob** - Text processing
- **NLTK Stopwords** - Stopword removal

### Machine Learning

- **TF-IDF Vectorization** - Text feature extraction
- **Logistic Regression** - Classification model
- **scipy** - Sparse matrix operations

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/AmeliaSyahla/sentiment-analysis.git
cd sentiment-analysis
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 5. Prepare Required Files

Ensure you have these files in your project directory:

```
sentiment_analysis/
├── Assets/
│   ├── Fix_Final_Berita_dan_Tweet.csv     # Your dataset
│   ├── full_lexicon.csv                    # Sentiment lexicon
│   └── combined_slang_words.txt           # Slang dictionary (optional)
├── app.py
├── train_model.py
└── requirements.txt
```

---

## 🚀 Usage

### Training the Model

Before using the application, you need to train the sentiment analysis model:

```bash
python train_model.py
```

**This will generate:**
- `model.pkl` - Trained Logistic Regression model
- `vectorizer.pkl` - TF-IDF vectorizer
- `model_config.json` - Model configuration

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🔄 Project Workflow

### Detailed Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   SENTIMENT ANALYSIS WORKFLOW                │
└─────────────────────────────────────────────────────────────┘

STEP 1: DATA PREPARATION
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Collection                                          │
│    └─ Load dataset (CSV with Twitter/News data)            │
│    └─ Check data quality and completeness                  │
│                                                             │
│ 2. Data Exploration                                         │
│    └─ Analyze text length distribution                     │
│    └─ Check for missing values                             │
│    └─ Identify data imbalances                             │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 2: TEXT PREPROCESSING
┌─────────────────────────────────────────────────────────────┐
│ 1. Text Cleaning                                            │
│    └─ Remove URLs, mentions (@user), hashtags              │
│    └─ Remove special characters and punctuation            │
│    └─ Remove emojis (with pattern matching)                │
│                                                             │
│ 2. Feature Extraction                                       │
│    └─ Calculate capital ratio (emotion indicator)          │
│    └─ Save original text for reference                     │
│                                                             │
│ 3. Text Normalization                                       │
│    └─ Convert to lowercase                                 │
│    └─ Normalize slang words (using dictionary)             │
│    └─ Standardize repeated characters                      │
│                                                             │
│ 4. Tokenization                                             │
│    └─ Split text into individual words                     │
│    └─ Remove words less than 3 characters                  │
│                                                             │
│ 5. Stopword Removal (Minimal)                               │
│    └─ Remove only non-informative words                    │
│    └─ Preserve sentiment-bearing words                     │
│                                                             │
│ 6. Stemming                                                 │
│    └─ Apply Sastrawi stemmer                               │
│    └─ Convert words to root form                           │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 3: SENTIMENT LABELING
┌─────────────────────────────────────────────────────────────┐
│ 1. Lexicon-Based Labeling                                   │
│    └─ Load sentiment lexicon (word:score pairs)            │
│    └─ Calculate sentiment scores for each text             │
│    └─ Assign labels based on thresholds:                   │
│        • Positive: score > 0.5                              │
│        • Negative: score < -0.5                             │
│        • Neutral: -0.5 ≤ score ≤ 0.5                        │
│                                                             │
│ 2. Label Distribution Analysis                              │
│    └─ Check class balance                                  │
│    └─ Visualize distribution                               │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 4: FEATURE ENGINEERING
┌─────────────────────────────────────────────────────────────┐
│ 1. TF-IDF Vectorization                                     │
│    └─ Max features: 500                                    │
│    └─ N-gram range: (1, 2) - unigrams and bigrams          │
│    └─ Minimum document frequency: 1                        │
│    └─ Sublinear TF scaling: True                           │
│                                                             │
│ 2. Additional Features                                      │
│    └─ Capital ratio (emotional intensity indicator)        │
│                                                             │
│ 3. Feature Combination                                      │
│    └─ Concatenate TF-IDF + Capital ratio                   │
│    └─ Result: 501 features total                           │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 5: MODEL TRAINING
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Splitting                                           │
│    └─ Train: 80% | Test: 20%                               │
│    └─ Stratified split (balanced classes)                  │
│                                                             │
│ 2. Model Selection                                          │
│    └─ Algorithm: Logistic Regression                       │
│    └─ Parameters:                                           │
│        • max_iter: 2000                                     │
│        • class_weight: 'balanced'                           │
│        • C: 1.0 (regularization)                            │
│        • solver: 'lbfgs'                                    │
│                                                             │
│ 3. Model Training                                           │
│    └─ Fit model on training data                           │
│    └─ Learn weights for each feature                       │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 6: MODEL EVALUATION
┌─────────────────────────────────────────────────────────────┐
│ 1. Performance Metrics                                      │
│    └─ Accuracy score                                       │
│    └─ Precision, Recall, F1-score (per class)              │
│    └─ Confusion matrix                                     │
│                                                             │
│ 2. Sample Testing                                           │
│    └─ Test on predefined examples                          │
│    └─ Verify sentiment predictions                         │
│    └─ Check probability distributions                      │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 7: MODEL DEPLOYMENT
┌─────────────────────────────────────────────────────────────┐
│ 1. Model Serialization                                      │
│    └─ Save trained model (pickle)                          │
│    └─ Save vectorizer (pickle)                             │
│    └─ Save configuration (JSON)                            │
│                                                             │
│ 2. Web Application (Streamlit)                              │
│    └─ Load saved models                                    │
│    └─ Create user interface                                │
│    └─ Implement real-time prediction                       │
└─────────────────────────────────────────────────────────────┘
                            ↓

STEP 8: USER INTERACTION
┌─────────────────────────────────────────────────────────────┐
│ 1. Input Processing                                         │
│    └─ User enters text                                     │
│    └─ Apply same preprocessing pipeline                    │
│    └─ Transform to feature vector                          │
│                                                             │
│ 2. Prediction                                               │
│    └─ Model predicts sentiment                             │
│    └─ Calculate confidence scores                          │
│                                                             │
│ 3. Results Display                                          │
│    └─ Show sentiment label with color coding               │
│    └─ Display probability distribution                     │
│    └─ Show preprocessed text (optional)                    │
└─────────────────────────────────────────────────────────────┘

---
