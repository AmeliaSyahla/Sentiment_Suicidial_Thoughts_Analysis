import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and config
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained('./indobert_sentiment_model')
tokenizer = AutoTokenizer.from_pretrained('./indobert_sentiment_model')

with open('indobert_config.json', 'r') as f:
    config = json.load(f)

print("\n" + "="*60)
print("MODEL CONFIGURATION CHECK")
print("="*60)

print("\nConfig file id2label:", config['id2label'])
print("Config file label2id:", config['label2id'])
print("\nModel config id2label:", model.config.id2label)
print("Model config label2id:", model.config.label2id)

# Test predictions
test_texts = [
    "Saya sangat senang dan bahagia hari ini!",  # Should be positive
    "Hari ini cuaca buruk sekali, saya sedih",    # Should be negative
    "Makan siang di kantor"                        # Should be neutral
]

print("\n" + "="*60)
print("TEST PREDICTIONS")
print("="*60)

model.eval()
for text in test_texts:
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    print(f"\nText: {text}")
    print(f"Logits: {logits.cpu().numpy()[0]}")
    print(f"Predicted ID: {pred_id}")
    print(f"Predicted Label: {model.config.id2label[int(pred_id)]}")
    print(f"Probabilities by ID:")
    for i, prob in enumerate(probs):
        print(f"  ID {i} ({model.config.id2label[i]}): {prob:.4f}")