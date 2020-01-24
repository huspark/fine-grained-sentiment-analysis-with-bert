from transformers.modeling_bert import *


class BertForSentimentAnalysis(BertPreTrainedModel):
    """
    This model is almost identical to BertForSequenceClassification, but it provides different loss functions in the
    regression mode.

        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss = 'mse'

        self.init_weights()

    def set_loss(self, loss):
        self.loss = loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # regression
            if self.num_labels == 1:
                if self.loss == 'mse':
                    # Use mean squared loss for regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                elif self.loss == 'smoothl1':
                    # Use smooth l1 loss for regression
                    loss_fct = torch.nn.SmoothL1Loss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                elif self.loss == 'masked_mse':
                    # Use masked mean squared loss for regression
                    loss = masked_mse_loss(logits.view(-1), labels.view(-1))
                elif self.loss == 'masked_smoothl1':
                    # Use masked smooth l1 loss for regression
                    loss = masked_smooth_l1_loss(logits.view(-1), labels.view(-1))
                else:
                    print('Loss function not supported.')

            # classification
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def masked_smooth_l1_loss(input, target):
    t = torch.abs(input - target)
    smooth_l1 = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)

    zeros = torch.zeros_like(smooth_l1)

    extreme_target = torch.abs(target - 2)
    extreme_input = torch.abs(input - 2)
    mask = (extreme_target == 2) * (extreme_input > 2)

    return torch.where(mask, zeros, smooth_l1).sum()


def masked_mse_loss(input, target):
    t = torch.abs(input - target)
    mse = t ** 2

    zeros = torch.zeros_like(mse)

    extreme_target = torch.abs(target - 2)
    extreme_input = torch.abs(input - 2)
    mask = (extreme_target == 2) * (extreme_input > 2)

    return torch.where(mask, zeros, mse).sum() / mse.size(-1)
