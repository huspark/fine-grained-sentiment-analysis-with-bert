# Yelp Fine-Grained Sentiment Analysis with BERT
In this project, we fine-tune a customized BERT[<sup>1</sup>](https://arxiv.org/pdf/1810.04805.pdf) (Bidirectional 
Encoder Representations from Transformers)-based model to fine-grained sentiment analysis of the Yelp-5 dataset. We 
have two main objectives:

1. Build a custom BERT-based model that uses either classification or regression approach to perform fine-grained 
sentiment analysis of the Yelp-5 dataset (see BertForSentimentAnalysis in <a href="model.py">model.py</a>)
2. Design and test custom loss functions that work well with a regression-based sentiment analysis model 
(see masked_mse_loss and masked_smooth_l1_loss in <a href="model.py">model.py</a>)

run_yelp.py and utils_yelp.py are based on the <a href="https://github.com/huggingface/transformers"> huggingface's 
PyTorch transformers</a> repository. For conciseness, this project only uses the original BERT model and does not 
support multi-GPU training.

## Overview

This project focuses on fine-grained sentiment analysis, in which a model predicts a review text's score as 
[0, 1, 2, 3, 4]. When we use a BERT-based model for this task, there are two dominant approaches: a 
classification-based approach and a regression-based approach.

### Classification-Based Model
1. Generate the embedding of a review text by extracting the BERT embedding of the [CLS] token. Typically, the output 
of this step is a two dimensional tensor of size [batch_size, hidden_size].
2. Use a linear layer of size [hidden_size, num_labels = 5] to map the review's BERT embedding to five outputs. 
Each of these five outputs correspond to the probability of the review's text score being [0, 1, 2, 3, 4]. The output 
is a two dimensional tensor of size [batch_size, 5].
3. Train the model using the cross-entropy loss function to perform a multi-label classification.
4. When testing, use the model to produce a review text's probability for each label and find the label with the 
highest probability.

### Regression-Based Model
1. Generate the embedding of a review text by extracting the BERT embedding of the [CLS] token.
2. Use a linear layer of size [hidden_size, num_labels = 1] to map the review's BERT embedding to a single 
output. This will correspond to the review's score.
3. Train the model using the mean-squared loss function to perform a regression.
4. When testing, use the model to produce a review text's real-valued score and round it up to the nearest integer.

The regression based approach has an advantage over the classification based approach: it produces a real-valued score 
of a review text while the classification based approach can only output the review text's probability for each label. 
However, the regression based approach results in a lower accuracy than the classification based approach.  

By designing custom loss functions specific to the regression-based model, we can slightly improve the 
model's accuracy.

## Loss function of a Regression Based Model for Fine-Grained Sentiment Analysis
In this section, we discuss necessary properties of a loss function for regression-based fine-grained sentiment 
analysis. Furthermore, we choose and test 4 different loss functions, two of which are custom-built.

### How Should the Model Compute the Loss in Edge Cases?
The model's prediction of a review text's score can be any real number while the label of the text is one of [0, 1, 2, 
3, 4]. This observation implies the loss function for regression-based fine-grained sentiment analysis model should

 * apply a small loss when the absolute value of (the model's prediction - label) < 0.5
 * apply a small loss when the model predicts a score < 0 for a review text whose label is 0.0
 * apply a small loss when the model predicts a score > 4 for a review text whose label is 4.0
 
The first property comes from the fact that the model rounds its real-valued prediction to the nearest integer. 
This means that if the absolute value of (the model's prediction - label) < 0.5, the rounded prediction will be equal 
to the example's label (i.e., round(the model's prediction) = label). In this circumstance, the model does not need to 
update its parameters by much since the model has already made the correct prediction. Furthermore, we ignore the fact 
that the ground truth scores for reviews with the same label are different. For example, the ground truth score of a 
review text with label=3 can be any real number from 2.5 to 3.5.

The second and third properties comes from the fact that any review with ground truth score less than 0 is
labeled with 0. Similarly, a review with its ground truth score higher than 4 is labeled with 4. Therefore, we can 
assume that the model made a correct prediction if it outputs a score less than 0 or a score higher than 4 for label=0 
or 4, respectively.

## Dataset

To download the original Yelp-5 dataset, follow this <a href="bit.ly/2kRWoof">link</a> and download 
"yelp_review_full.csv.tar.gz". The dataset contains 650k examples consisting of 130k examples for each label.

## Requirements
To install required packages for this project, run the following command on your virgtual environment.
```shell
pip install -r requirements.txt
```

## Run it on CPU/GPU
To run the project on our machine, copy and paste one of the following consoles. Note that the full dataset takes 
about 2hrs/epoch to train on Nvidia RTX 2080 Ti. For experiments, we recommend using the spit_data function from 
<a href="utils_yelp.py">utils_yelp.py</a> to take a desired fraction of data.  

To test the model with different loss functions for the regression based approach, uncomment the desired loss function 
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