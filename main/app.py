import streamlit as st

# MUST BE FIRST
st.set_page_config(
    page_title="Sentiment Analysis Demo",
    layout="centered"
)

import pandas as pd
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        # Load config
        with open('indobert_config.json', 'r') as f:
            config = json.load(f)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('./indobert_sentiment_model')
        model = AutoModelForSequenceClassification.from_pretrained('./indobert_sentiment_model')
        model.to(device)
        model.eval()
        
        return model, tokenizer, config
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Text preprocessing function (lighter for BERT)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove mentions and hashtags symbols (keep the text)
    text = re.sub(r'@([A-Za-z0-9_]+)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove RT
    text = re.sub(r'RT[\s]+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_sentiment(text, model, tokenizer, config):
    # Preprocess
    processed_text = clean_text(text)
    
    if not processed_text or len(processed_text.strip()) == 0:
        return 'netral', [0.0, 1.0, 0.0], "(empty after preprocessing)"
    
    # Tokenize
    encoding = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    # Map prediction to label
    id2label = {int(k): v for k, v in config['id2label'].items()}
    prediction = id2label[pred_id]
    
    # Probabilities in order: [negatif, netral, positif]
    probabilities = probs.tolist()
    
    return prediction, probabilities, processed_text

# Streamlit UI
def main():
    st.title("🎭 Sentiment Analysis with IndoBERT")
    st.markdown("---")
    
    # Load model
    model, tokenizer, config = load_model()
    
    if model is None or tokenizer is None or config is None:
        st.error("Model files not found! Please train the model first using `python train_model.py`")
        st.info("Expected files:\n- indobert_sentiment_model/ (directory)\n- indobert_config.json")
        st.stop()
    
    # Show model info
    st.success(f"✓ IndoBERT model loaded successfully!")
    
    with st.expander("Model Information"):
        st.write(f"**Model:** {config['model_name']}")
        st.write(f"**Device:** {device}")
        st.write(f"**Max Length:** {config['max_length']}")
        st.write(f"**Classes:** {list(config['label2id'].keys())}")
        if 'test_accuracy' in config:
            st.write(f"**Test Accuracy:** {config['test_accuracy']*100:.2f}%")
    
    # Input text area
    st.subheader("Enter your text")
    
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
        with st.spinner("Analyzing with IndoBERT..."):
            try:
                prediction, probabilities, processed_text = predict_sentiment(
                    user_input, model, tokenizer, config
                )
                
                st.markdown("---")
                st.subheader("Results")
                
                # Sentiment emoji mapping
                sentiment_emoji = {
                    'positif': "😊",
                    'negatif': "😞",
                    'netral': "😐"
                }
                
                sentiment_colors = {
                    'positif': "#d4edda",
                    'negatif': "#f8d7da",
                    'netral': '#fff3cd'
                }
                
                sentiment_text_colors = {
                    'positif': "#155724",
                    'negatif': "#721c24",
                    'netral': '#856404'
                }
                
                st.markdown(
                    f"""
                    <div style="background-color: {sentiment_colors[prediction]}; 
                                padding: 30px; 
                                border-radius: 10px; 
                                text-align: center;
                                margin: 20px 0;
                                border: 2px solid {sentiment_text_colors[prediction]};">
                        <h1 style="color: {sentiment_text_colors[prediction]}; margin: 0;">
                            {sentiment_emoji[prediction]} {prediction.upper()}
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display probabilities
                st.subheader("Confidence Score")
                col1, col2, col3 = st.columns(3)
                
                labels = ['negatif', 'netral', 'positif']
                label_emoji = ['😞', '😐', '😊']
                
                for i, (label, prob, emoji) in enumerate(zip(labels, probabilities, label_emoji)):
                    with [col1, col2, col3][i]:
                        st.metric(
                            label=f"{emoji} {label.capitalize()}",
                            value=f"{prob*100:.2f}%"
                        )
                        st.progress(float(prob))
                
                # Show processed text
                with st.expander("View Processed Text"):
                    st.caption(f"Word count: {len(processed_text.split())} words")
                    
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
            <p> by Amelia Syahla</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()