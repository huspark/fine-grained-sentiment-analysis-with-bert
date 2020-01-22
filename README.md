# Yelp Fine-Grained Sentiment Analysis with BERT
This project fine-tunes a customized BERT(Bidirectional Encoder Representations from Transformers)-based model for 
sentiment classification/regression of the Yelp-5 dataset. 

1. Build a custom BERT-based model that performs both classification and regression techniques for sentiment analysis 
(see BertForSentimentAnalysis in <a href="#model.py">model.py</a>)
2. Design a custom loss function that works well with sentiment analysis regression (see masked_smooth_l1_loss in 
model.py)

run_yelp.py and utils_yelp.py are based on huggingface's repository: https://github.com/huggingface/transformers. 
For conciseness, this project only uses the original BERT model and does not support multi-GPU training.

## Overview

This project focuses on fine-grained sentiment analysis, which requires a model that scores a review text as 
[0, 1, 2, 3, 4]. When we perform fine-grained sentiment analysis using BERT-based models, there are two training 
pipelines: a classification-based approach and a regression-based approach.

### Classification Based Approach
1. Generate the BERT embedding of a review by extracting the embedding of the [CLS] token.
2. Use a linear layer of size [BERT embedding size, num_labels = 5] to map the review's BERT embedding to 5 outputs.
 These 5 outputs correspond to the probability of the review's score being [0, 1, 2, 3, 4].
3. Use the cross-entropy loss to perform a multi-label classification.

### Regression Based Approach
1. Generate the BERT embedding of a review by extracting the embedding of the [CLS] token.
2. Use a linear layer of size [BERT embedding size, num_labels = 1] to map the review's BERT embedding to a single 
output. This will correspond to the review's score.
3. Use the mean-squared loss to perform a regression.

The regression based approach underperforms the classification based approach because 1) the linear layer with 
output dimension=1 limits the complexity of the model, and 2) the mean squared loss function is not adequate for 
the fine-grained sentiment analysis task.

## Loss function for Regression Based Approach of Sentiment Analysis

## Dataset

To download the original dataset. Please refer to 

## Requirements
To run the code, you need to install

## Run it on CPU/GPU

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
CUDA_VISIBLE_DEVICES=5 python3 run_yelp.py \
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
CUDA_VISIBLE_DEVICES=5 python3 run_yelp.py \
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