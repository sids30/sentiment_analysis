import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load fine-tuned model and tokenizer
model_path = "model_epoch_4.pt"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define function to get sentiment predictions
def get_sentiment(text, threshold=0.2):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        diff = torch.abs(probs[:, 0] - probs[:, 1])
        neutral_mask = diff < threshold

        if neutral_mask.any():
            predicted_class = 2
        else:
            predicted_class = torch.argmax(probs, dim=1).item()

    sentiment_mapping = {0: "negative", 1: "positive", 2: "neutral"}
    predicted_sentiment = sentiment_mapping[predicted_class]

    return predicted_sentiment

# User input
while True:
    sentence = input("Enter a sentence ('model_exit' to quit): ").strip()
    if sentence.lower() == 'model_exit':
        break

    if sentence:
        predicted_sentiment = get_sentiment(sentence)
        print(f"Predicted Sentiment: {predicted_sentiment}")
        print()
    else:
        print("Please enter a valid sentence.")
