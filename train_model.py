import pandas as pd
import re
import string
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load your dataset with better error handling
print("Loading dataset...")
try:
    df = pd.read_csv(
        "Assets/Fix_Final_Berita_dan_Tweet.csv",
        sep=';',
        encoding='utf-8',
        on_bad_lines='skip',
        engine='python',
        quoting=1,
        escapechar='\\'
    )
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("\nDataset Info:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Identify the text column
text_column = None
possible_text_columns = ['Content', 'content', 'full_text', 'text', 'tweet', 'Tweet', 'message', 'Message']

for col in possible_text_columns:
    if col in df.columns:
        text_column = col
        print(f"\nFound text column: '{text_column}'")
        break

if text_column is None:
    print("\nAvailable columns:", df.columns.tolist())
    text_column = input("Enter the name of the column containing the text data: ").strip()

if text_column != 'full_text':
    df = df.rename(columns={text_column: 'full_text'})

# Remove rows with missing text
print(f"\nOriginal dataset size: {df.shape}")
df = df.dropna(subset=['full_text'])
df = df[df['full_text'].astype(str).str.strip() != '']
print(f"Dataset after removing empty rows: {df.shape}")

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load slang dictionary
print("\nLoading slang dictionary...")
try:
    with open('Assets/combined_slang_words.json', 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)
    print(f"✓ Loaded {len(slang_dict)} slang words")
except FileNotFoundError:
    print("⚠ Warning: combined_slang_words.json not found, using empty dict")
    slang_dict = {}

# Preprocessing functions
def clean_twitter_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # Keep spaces!
    text = text.replace("&amp;", "&")
    
    emoji_pattern = re.compile(
        "["
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def hitung_rasio_kapital(teks):
    if not isinstance(teks, str) or len(teks) == 0:
        return 0
    kata_kata = teks.split()
    if len(kata_kata) == 0:
        return 0
    kata_kapital = sum(1 for kata in kata_kata if kata.isupper() and len(kata) > 1)
    rasio = kata_kapital / len(kata_kata)
    return rasio

def normalize_text(text, slang_dictionary):
    if not isinstance(text, str):
        return ""
    tokens = text.lower().split()  # Lowercase here
    normalized_tokens = [slang_dictionary.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

def remove_stopwords_light(tokens):
    # Use fewer stopwords to preserve more meaning
    stopwords_indonesia = set(stopwords.words('indonesian'))
    stopwords_english = set(stopwords.words('english'))
    
    # Only remove very common words that don't carry sentiment
    minimal_stopwords = ['yang', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 
                         'the', 'a', 'an', 'and', 'or', 'but']
    stopwords_final = set(minimal_stopwords)
    
    return [word for word in tokens if word not in stopwords_final and len(word) > 2]

def stem_token_list(tokens):
    return [stemmer.stem(word) for word in tokens]

# Load lexicon for labeling
print("Loading lexicon...")
try:
    try:
        lexicon = pd.read_csv('Assets/full_lexicon.csv', encoding='utf-8')
    except:
        try:
            lexicon = pd.read_csv('Assets/full_lexicon.csv', encoding='latin-1')
        except:
            lexicon = pd.read_csv('Assets/full_lexicon.csv', sep=';', encoding='utf-8')
    
    print(f"Lexicon shape: {lexicon.shape}")
    print(f"Lexicon columns: {lexicon.columns.tolist()}")
    
    if 'word' not in lexicon.columns or 'weight' not in lexicon.columns:
        print(f"Available columns in lexicon: {lexicon.columns.tolist()}")
        exit(1)
    
    lexicon = lexicon[['word', 'weight']]
    lexicon_weight = pd.Series(lexicon['weight'].values, index=lexicon['word']).to_dict()
    print(f"✓ Loaded {len(lexicon_weight)} lexicon words")
except Exception as e:
    print(f"❌ ERROR loading lexicon: {e}")
    exit(1)

def get_sentiment_label(text, lexicon):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 'netral'
    
    tokens = text.split()  # Already tokenized
    total_score = 0
    matched_words = 0
    
    for token in tokens:
        score = lexicon.get(token, 0)
        if score != 0:
            total_score += score
            matched_words += 1
    
    # Use threshold for better separation
    if total_score > 0.5:
        return 'positif'
    elif total_score < -0.5:
        return 'negatif'
    else:
        return 'netral'

# Preprocessing pipeline
print("\n" + "="*60)
print("STARTING PREPROCESSING PIPELINE")
print("="*60)

# Save original text
df['original_text'] = df['full_text'].copy()

# Clean text
print("1. Cleaning text...")
df['full_text'] = df['full_text'].astype(str).apply(clean_twitter_text)
df = df[df['full_text'].str.strip() != '']
print(f"   ✓ Rows after cleaning: {len(df)}")

# Calculate capital ratio BEFORE lowercasing
print("2. Calculating capital ratio...")
df['rasio_kapital'] = df['full_text'].apply(hitung_rasio_kapital)
print(f"   ✓ Done")

# Normalize (includes lowercasing)
print("3. Normalizing text...")
df['text_normalized'] = df['full_text'].apply(lambda x: normalize_text(x, slang_dict))
print(f"   ✓ Done")

# Tokenize
print("4. Tokenizing...")
df['text_tokens'] = df['text_normalized'].apply(lambda x: x.split())
print(f"   ✓ Done")

# Remove stopwords (lighter version)
print("5. Removing stopwords (minimal)...")
df['text_tokens'] = df['text_tokens'].apply(remove_stopwords_light)
print(f"   ✓ Done")

# Stemming
print("6. Stemming...")
df['text_tokens'] = df['text_tokens'].apply(stem_token_list)
print(f"   ✓ Done")

# Join back
df['text_preprocessed'] = df['text_tokens'].apply(lambda x: ' '.join(x))

# Remove empty rows after preprocessing
df = df[df['text_preprocessed'].str.strip() != '']
print(f"   ✓ Rows after preprocessing: {len(df)}")

# Show some examples
print("\n" + "="*60)
print("PREPROCESSING EXAMPLES")
print("="*60)
for i in range(min(3, len(df))):
    print(f"\nOriginal: {df.iloc[i]['original_text'][:100]}")
    print(f"Processed: {df.iloc[i]['text_preprocessed']}")

# Label with lexicon
print("\n8. Labeling data with sentiment...")
df['label'] = df['text_preprocessed'].apply(lambda text: get_sentiment_label(text, lexicon_weight))
print(f"   ✓ Done")

print("\n" + "="*60)
print("LABEL DISTRIBUTION")
print("="*60)
print(df['label'].value_counts())
print(f"\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Check if we have all classes
if len(df['label'].unique()) < 3:
    print("\n⚠️ WARNING: Not all sentiment classes present in data!")
    print("This may cause issues with training.")

# Prepare features with MORE features
print("\n" + "="*60)
print("PREPARING FEATURES")
print("="*60)

# Increase max_features and reduce min_df
vectorizer = TfidfVectorizer(
    max_features=500,  # Increased from 100
    min_df=1,  # Allow words that appear at least once
    ngram_range=(1, 2),  # Include bigrams
    sublinear_tf=True
)

X_text = vectorizer.fit_transform(df['text_preprocessed'])
X_rasio = df[['rasio_kapital']].values
X_gabungan = hstack([X_text, X_rasio])

print(f"Feature matrix shape: {X_gabungan.shape}")
print(f"Number of features: {X_gabungan.shape[1]}")

y = df['label']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_gabungan, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train model with better parameters
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight='balanced',
    C=1.0,  # Regularization
    solver='lbfgs'
)

model.fit(X_train, y_train)
print("✓ Model training complete!")

# Evaluate
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("Training Set Performance:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['negatif', 'netral', 'positif'])
print("\nConfusion Matrix:")
print("Predicted ->")
print(pd.DataFrame(cm, 
                   index=['negatif', 'netral', 'positif'], 
                   columns=['negatif', 'netral', 'positif']))

# Test on some examples
print("\n" + "="*60)
print("TESTING ON SAMPLE TEXTS")
print("="*60)

test_texts = [
    "Saya sangat senang dan bahagia hari ini!",
    "Hari ini cuaca buruk sekali, saya sedih",
    "Makan siang di kantor"
]

for text in test_texts:
    # Preprocess
    cleaned = clean_twitter_text(text)
    rasio = hitung_rasio_kapital(cleaned)
    normalized = normalize_text(cleaned, slang_dict)
    tokens = normalized.split()
    tokens = remove_stopwords_light(tokens)
    tokens = stem_token_list(tokens)
    processed = ' '.join(tokens)
    
    # Predict
    X_text_test = vectorizer.transform([processed])
    X_combined_test = hstack([X_text_test, [[rasio]]])
    pred = model.predict(X_combined_test)[0]
    proba = model.predict_proba(X_combined_test)[0]
    
    print(f"\nText: {text}")
    print(f"Processed: {processed}")
    print(f"Prediction: {pred}")
    print(f"Probabilities: neg={proba[0]:.2f}, net={proba[1]:.2f}, pos={proba[2]:.2f}")

# Save model and vectorizer
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save preprocessing config
config = {
    'slang_dict_size': len(slang_dict),
    'vectorizer_features': X_gabungan.shape[1],
    'classes': list(model.classes_)
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print("Files saved:")
print("  - model.pkl")
print("  - vectorizer.pkl")
print("  - model_config.json")
print("\nYou can now run the Streamlit app with:")
print("  streamlit run app.py")
print("="*60)