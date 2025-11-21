import pandas as pd
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv(
        "../Assets/Fix_Final_Berita_dan_Tweet.csv",
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

# Identify text column
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

# Basic text cleaning (lighter for BERT)
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

# Load lexicon for labeling
print("\nLoading lexicon...")
try:
    try:
        lexicon = pd.read_csv('../Assets/full_lexicon.csv', encoding='utf-8')
    except:
        try:
            lexicon = pd.read_csv('../Assets/full_lexicon.csv', encoding='latin-1')
        except:
            lexicon = pd.read_csv('../Assets/full_lexicon.csv', sep=';', encoding='utf-8')
    
    lexicon = lexicon[['word', 'weight']]
    lexicon_weight = pd.Series(lexicon['weight'].values, index=lexicon['word']).to_dict()
    print(f"✓ Loaded {len(lexicon_weight)} lexicon words")
except Exception as e:
    print(f"❌ ERROR loading lexicon: {e}")
    exit(1)

def get_sentiment_label(text, lexicon):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 'netral'
    
    tokens = text.lower().split()
    total_score = 0
    matched_words = 0
    
    for token in tokens:
        score = lexicon.get(token, 0)
        if score != 0:
            total_score += score
            matched_words += 1
    
    if total_score > 0.5:
        return 'positif'
    elif total_score < -0.5:
        return 'negatif'
    else:
        return 'netral'

# Clean and label data
print("\n" + "="*60)
print("PREPROCESSING DATA")
print("="*60)

df['cleaned_text'] = df['full_text'].apply(clean_text)
df = df[df['cleaned_text'].str.strip() != '']
print(f"Rows after cleaning: {len(df)}")

# Label data
print("\nLabeling data with sentiment...")
df['label'] = df['cleaned_text'].apply(lambda text: get_sentiment_label(text, lexicon_weight))

print("\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Map labels to integers
label2id = {'negatif': 0, 'netral': 1, 'positif': 2}
id2label = {0: 'negatif', 1: 'netral', 2: 'positif'}
df['label_id'] = df['label'].map(label2id)

# Initialize tokenizer
print("\n" + "="*60)
print("LOADING INDOBERT TOKENIZER")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
print("✓ Tokenizer loaded")

# Create PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'].values,
    df['label_id'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label_id']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create datasets and dataloaders
train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
test_dataset = SentimentDataset(X_test, y_test, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load model
print("\n" + "="*60)
print("LOADING INDOBERT MODEL")
print("="*60)

model = AutoModelForSequenceClassification.from_pretrained(
    "indolem/indobert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)
model.to(device)
print("✓ Model loaded and moved to device")

# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), 
            predictions, 
            true_labels)

# Training loop
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 60)
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
    
    val_acc, val_loss, _, _ = eval_model(model, test_loader, device)
    print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'indobert_sentiment_best.pth')
        print('✓ Best model saved!')

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

model.load_state_dict(torch.load('indobert_sentiment_best.pth'))
test_acc, test_loss, y_pred, y_true = eval_model(model, test_loader, device)

print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Classification report
y_pred_labels = [id2label[pred] for pred in y_pred]
y_true_labels = [id2label[true] for true in y_true]

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['negatif', 'netral', 'positif'])
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, 
                   index=['negatif', 'netral', 'positif'], 
                   columns=['negatif', 'netral', 'positif']))

# Test on sample texts
print("\n" + "="*60)
print("TESTING ON SAMPLE TEXTS")
print("="*60)

test_texts = [
    "Saya sangat senang dan bahagia hari ini!",
    "Hari ini cuaca buruk sekali, saya sedih",
    "Makan siang di kantor"
]

model.eval()
for text in test_texts:
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    print(f"\nText: {text}")
    print(f"Prediction: {id2label[pred]}")
    print(f"Probabilities: neg={probs[0]:.2f}, net={probs[1]:.2f}, pos={probs[2]:.2f}")

# Save model and config
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save full model
model.save_pretrained('./indobert_sentiment_model')
tokenizer.save_pretrained('./indobert_sentiment_model')

# Save config
config = {
    'model_name': 'indolem/indobert-base-uncased',
    'max_length': MAX_LENGTH,
    'num_labels': 3,
    'label2id': label2id,
    'id2label': {int(k): v for k, v in id2label.items()},
    'test_accuracy': float(test_acc)
}

with open('indobert_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print("Files saved:")
print("  - indobert_sentiment_best.pth")
print("  - indobert_sentiment_model/ (directory)")
print("  - indobert_config.json")
print("\nNext steps:")
print("  1. Update app.py to use the IndoBERT model")
print("  2. Run: streamlit run app.py")
print("="*60)