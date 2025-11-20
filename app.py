import streamlit as st

# MUST BE FIRST
st.set_page_config(
    page_title="Sentiment Analysis Demo",
    layout="centered"
)

import pandas as pd
import re
import string
import json
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from scipy.sparse import hstack

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

download_nltk_data()

# Initialize stemmer
@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = get_stemmer()

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Load slang dictionary
@st.cache_resource
def load_slang_dict():
    possible_paths = [
        'Assets/combined_slang_words.json',
        'combined_slang_words.json',
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    
    return {}

slang_dict = load_slang_dict()

# Text preprocessing functions - MUST MATCH TRAINING
def clean_twitter_text(text):
    if not isinstance(text, str):
        return ""
        
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
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
    tokens = text.lower().split()
    normalized_tokens = [slang_dictionary.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)

def remove_stopwords_light(tokens):
    minimal_stopwords = ['yang', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 
                         'the', 'a', 'an', 'and', 'or', 'but']
    stopwords_final = set(minimal_stopwords)
    
    return [word for word in tokens if word not in stopwords_final and len(word) > 2]

def stem_token_list(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    # Clean text
    cleaned = clean_twitter_text(text)
    
    # Calculate capital ratio BEFORE lowercasing
    rasio_kapital = hitung_rasio_kapital(cleaned)
    
    # Normalize (includes lowercasing)
    normalized = normalize_text(cleaned, slang_dict)
    
    # Tokenize
    tokens = normalized.split()
    
    # Remove stopwords (minimal)
    tokens = remove_stopwords_light(tokens)
    
    # Stemming
    tokens = stem_token_list(tokens)
    
    # Join back
    final_text = ' '.join(tokens)
    
    return final_text, rasio_kapital

def predict_sentiment(text, model, vectorizer):
    # Preprocess
    processed_text, rasio_kapital = preprocess_text(text)
    
    if not processed_text or len(processed_text.strip()) == 0:
        return 'netral', [0.0, 1.0, 0.0], "(empty after preprocessing)"
    
    # Vectorize
    X_text = vectorizer.transform([processed_text])
    X_rasio = [[rasio_kapital]]
    X_combined = hstack([X_text, X_rasio])
    
    # Predict
    prediction = model.predict(X_combined)[0]
    probabilities = model.predict_proba(X_combined)[0]
    
    return prediction, probabilities, processed_text

# Streamlit UI
def main():
    st.title("🎭 Sentiment Analysis")
    st.markdown("---")
    
    # Check if slang dict is loaded
    if len(slang_dict) == 0:
        st.info("ℹ️ Running without slang dictionary. Predictions may be less accurate.")
    else:
        st.success(f"✓ Loaded {len(slang_dict)} slang words")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("❌ Model files not found! Please train the model first using `python train_model.py`")
        st.stop()
    
    # Show model info
    with st.expander("Note"):
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Classes: {list(model.classes_)}")
        st.write(f"Number of features: {vectorizer.max_features}")
    
    # Input text area
    st.subheader("📝 Enter your text")
    
    # Example buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Positive Example"):
            st.session_state.input_text = "Saya sangat senang dan bahagia hari ini! Terima kasih banyak!"
    with col2:
        if st.button("Neutral Example"):
            st.session_state.input_text = "Hari ini saya makan siang di kantor"
    with col3:
        if st.button("Negative Example"):
            st.session_state.input_text = "Saya sangat sedih dan kecewa dengan layanan ini. Sangat buruk!"
    
    user_input = st.text_area(
        "Type or paste your text here:",
        height=150,
        value=st.session_state.get('input_text', ''),
        placeholder="Example: Hari ini saya sangat bahagia sekali!"
    )
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Analyze Sentiment", use_container_width=True, type="primary")
    
    if predict_button and user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                prediction, probabilities, processed_text = predict_sentiment(user_input, model, vectorizer)
                
                st.markdown("---")
                st.subheader("Results")
                
                sentiment_bg_colors = {
                    'positif': "#268e3e",
                    'negatif': "#d62d3b",
                    'netral': '#fff3cd'
                }
                
                st.markdown(
                    f"""
                    <div style="background-color: {sentiment_bg_colors[prediction]}; 
                                padding: 20px; 
                                border-radius: 10px; 
                                text-align: center;
                                margin: 20px 0;">
                        <h2>{[prediction]} Sentiment: {prediction.upper()}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display probabilities
                st.subheader("Confidence Score")
                col1, col2, col3 = st.columns(3)
                
                labels = ['negatif', 'netral', 'positif']
                
                for i, (label, prob) in enumerate(zip(labels, probabilities)):
                    with [col1, col2, col3][i]:
                        st.metric(
                            label=label.capitalize(),
                            value=f"{prob*100:.2f}%"
                        )
                        st.progress(float(prob))
                
                # Show processed text
                with st.expander("Preprocessed Text"):
                    if processed_text:
                        st.code(processed_text)
                        st.caption(f"Word count: {len(processed_text.split())} words")
                    else:
                        st.warning("No text remaining after preprocessing")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    elif predict_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
            <p>Built with Streamlit | Sentiment Analysis Model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()