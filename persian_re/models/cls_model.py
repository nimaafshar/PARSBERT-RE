import torch
import torch.nn as nn
from transformers import BertModel
from ..settings import Config


class CLSModel(nn.Module):

    def __init__(self, config):
        super(CLSModel, self).__init__()

        self.bert = BertModel.from_pretrained(Config.MODEL_NAME_OR_PATH)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def resize_token_embeddings(self, size):
        self.bert.resize_token_embeddings(size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # pooled output is bert's output for [CLS] token
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


