# Yelp Fine-Grained Sentiment Analysis with BERT
This project uses BERT(Bidirectional Encoder Representations from Transformers) for sentiment analysis of the Yelp-5 
dataset. The project has two main objectives:

1. Build a custom BERT-based model to perform sentiment analysis (see BertForSentimentAnalysis in model.py)
2. Design a custom loss function that works well with sentiment analysis regression (see masked_smooth_l1_loss in 
model.py)

run_yelp.py and utils_yelp.py are based on huggingface's repository: https://github.com/huggingface/transformers. 
For conciseness, this project only uses the original bert model and does not support parallel computing.