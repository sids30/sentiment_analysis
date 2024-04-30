# Task

Predict either positive, negative, or neutral sentiment from user input.

# Objective

The main and intended objective is to analyze reviews and other text (natural language) content postings throughout the internet and other places, to infer whether the sentences/phrases imply a positive or negative sentiment or message, by its created user.

# Instructions:

- Clone the repository to your working directory
- Install .zip file from the Google Drive link (https://drive.google.com/file/d/1On1BdUn2AAEHLihehvYfLPuTZ4LmMUcj/view?usp=sharing), that has the required PyTorch (.pt) files that contain model epochs 4 and 5 for running the model
- Install libraries (pip[3] install pandas torch transformers scikit-learn) to your current working directory
- Install any other required dependencies
- In the Set either "model_epoch_4.pt" or "model_epoch_5.pt" as the model_path variable
- Run the *run_sentiment_analysis.py* file to try and test

# Fair usage warning (bing.csv)

The data set (*bing.csv*) is provided as is for educational and research purposes only. It is not intended for commercial use. License information for this data set is unknown. Users should independently verify the validity of the license terms before using the dataset for purposes other than education or research. The dataset used was created by the Kaggle user, Andrada and her datasets can be explored here: https://www.kaggle.com/datasets/andradaolteanu/bing-nrc-afinn-lexicons


# Note

- The model only predicts English sentences and phrases accurately, other languages may be added in the future.
- The model predicts "positive" and "negative" sentiments fairly accurately, however, it produces undesirable results for "neutral" predictions. This will most likely be fixed with some future updates.
