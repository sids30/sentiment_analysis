import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import torch.nn.functional as F

# Load data
data = pd.read_csv("archive2/bing.csv")

# Encode sentiments into numerical labels
sentiment_mapping = {"negative": 0, "positive": 1, "neutral": 2}
data['label'] = data['sentiment'].map(sentiment_mapping)

# Split data into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['word']
        label = self.data.iloc[idx]['label']

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs['labels'] = torch.tensor(label)  # Remove .unsqueeze(0)

        return inputs


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize BERT model for sequence classification
num_labels = len(sentiment_mapping)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Convert data to DataLoader using CustomDataset
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

# Custom collate function
def custom_collate(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch])

    # Pad sequences to same length within each batch
    max_length = max(len(ids[0]) for ids in input_ids)
    input_ids = [ids[0].tolist() + [tokenizer.pad_token_id] * (max_length - len(ids[0])) for ids in input_ids]
    attention_masks = [masks[0].tolist() + [0] * (max_length - len(masks[0])) for masks in attention_masks]

    return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_masks), 'labels': labels}

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate)

# "Neutral" classification threshold
threshold = 0.2

# Fine-tune BERT
def fine_tune_bert(model, train_loader, num_epochs=5, learning_rate=1e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        for i, data in enumerate(train_loader, 0):
            inputs = data['input_ids']
            masks = data['attention_mask']
            labels = data['labels']

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=masks, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Apply "neutral" threshold classification
            probabilities = F.softmax(logits, dim=1)
            diff = torch.abs(probabilities[:, 0] - probabilities[:, 1])
            neutral_mask = diff < threshold
            labels[neutral_mask] = 2

            # Predictions and true labels for classification report
            all_predictions.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Mini-batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Print classification report
        print("Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=sentiment_mapping.keys()))

        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pt")

        print()

# Run fine-tuning
fine_tune_bert(model, train_loader)
