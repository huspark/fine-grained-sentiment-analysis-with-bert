# Yelp Fine-Grained Sentiment Analysis with BERT
This project fine-tunes a customized BERT(Bidirectional Encoder Representations from Transformers)-based model for 
sentiment classification/regression of the Yelp-5 dataset. It has two main objectives:

1. Build a custom BERT-based model that performs classification or regression techniques for sentiment analysis of 
Yelp-5 dataset (see BertForSentimentAnalysis in <a href="model.py">model.py</a>)
2. Design and test custom loss functions that work well with sentiment analysis regression (see masked_smooth_l1_loss in 
<a href="model.py">model.py</a>)

run_yelp.py and utils_yelp.py are based on templates from the <a href="https://github.com/huggingface/transformers"> 
huggingface's PyTorch transformers</a> repository. For conciseness, this project only uses the original BERT model and does not support multi-GPU training.

## Overview

This project focuses on the fine-grained sentiment analysis, which requires a model that scores a review text as 
[0, 1, 2, 3, 4]. When we use BERT-based models for this task, there are two dominant approaches: a classification-based 
approach and a regression-based approach.

### Classification Based Approach
1. Generate the embedding of a review text by extracting the BERT embedding of the [CLS] token.
2. Use a linear layer of size [hidden_size, num_labels = 5] to map the review's BERT embedding to 5 outputs.
 These 5 outputs correspond to the probability of the review's score being [0, 1, 2, 3, 4].
3. Train the model using the cross-entropy loss function to perform a multi-label classification.
4. When testing, use the model to produce a review text's probability for each label and find the label with the 
highest probability.

### Regression Based Approach
1. Generate the embedding of a review text by extracting the BERT embedding of the [CLS] token.
2. Use a linear layer of size [hidden_size, num_labels = 1] to map the review's BERT embedding to a single 
output. This will correspond to the review's score.
3. Train the model using the mean-squared loss function to perform a regression.
4. When testing, use the model to produce a review text's real-valued score and round it up to the nearest integer.

The regression based approach has an advantage over the classification based approach: it produces a real-valued score 
of a review text whereas the classification based approach can only output the review text's probability for each label. 
However, the regression based approach results in a lower accuracy than the classification based approach. In the next 
section, I suggest a method to improve accuracy of the regression based approach of the fine-grained sentiment 
analysis.

## Loss functions for Regression Based Approach of Fine-Grained Sentiment Analysis
The regression based approach has the following objective function
 - minimize ||yhat - y||_2 where yhat belongs to [-inf, inf] and y is in [0, 1, 2, 3, 4]

## Dataset

To download the original Yelp-5 dataset, follow this <a href="bit.ly/2kRWoof">link</a> and download 
"yelp_review_full.csv.tar.gz". The dataset contains 650k examples consisting of 130k examples for each label.

## Requirements
To install required packages for this project, run the following command on your virtual environment.
```shell
pip install -r requirements.txt
```

## Run it on CPU/GPU
To run the project on our machine, copy and paste one of the following consoles. Note that the full dataset takes 
about 2hrs/epoch to train on Nvidia RTX 2080 Ti. For experiments, I recommend using the spit_data function from 
<a href="utils_yelp.py">utils_yelp.py</a> to take a desired fraction of data.  

To test the model with different loss functions for the regression based approach, uncomment the desired loss funciton 
in <a href="model.py">model.py</a>.

### Sample Command for Running Classification on CPU
```shell
python3 run_yelp.py \
    --data_dir ./ \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir masked-loss \
    --max_seq_length  128 \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 32 \
    --save_steps 100000 \
    --seed 1 \
    --overwrite_output_dir \
```
### Sample Command for Running Classification on GPU
```shell
CUDA_VISIBLE_DEVICES=0 python3 run_yelp.py \
    --data_dir ./ \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir masked-loss \
    --max_seq_length  128 \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 32 \
    --save_steps 100000 \
    --seed 1 \
    --overwrite_output_dir \
```
### Sample Command for Running Regression on CPU
```shell
python3 run_yelp.py \
    --data_dir ./ \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir masked-loss \
    --max_seq_length  128 \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 32 \
    --save_steps 100000 \
    --seed 1 \
    --overwrite_output_dir \
    --regression
```
### Sample Command for Running Regression on GPU
```shell
CUDA_VISIBLE_DEVICES=0 python3 run_yelp.py \
    --data_dir ./ \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir masked-loss \
    --max_seq_length  128 \
    --num_train_epochs 3 \
    --per_gpu_train_batch_size 32 \
    --save_steps 100000 \
    --seed 1 \
    --overwrite_output_dir \
    --regression
```
## Citation