# Task

Predict either positive, negative, or neutral sentiment from user input.

# Objective

The main and intended objective is to analyze reviews and other text (natural language) content postings throughout the internet and other places, to infer whether the sentences/phrases imply a positive or negative sentiment or message, by its created user.

# Instructions:

- Clone the repository to your working directory
- Install libraries (pip[3] install pandas torch transformers scikit-learn) to your current working directory
- Install any other required dependencies
- Install .zip file from the Google Drive link, that has the required PyTorch (.pt) files that contain model epochs 4 and 5 for running the model
- Set either "model_epoch_4" or "model_epoch_5" as the model_path variable
- Run the "run_sentiment_analysis.py" file to try and test

  # Note

  The model predicts "positive" and "negative" sentiments fairly accurately, however, it produces undesirable results for "neutral" predictions. This will most likely be fixed with some future updates.
