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


class EntityStartModel(nn.Module):

    def __init__(self, config):
        super(EntityStartModel, self).__init__()
        self.e1_start_token_id: int = config.e1_start_token_id
        self.e2_start_token_id: int = config.e2_start_token_id
        self.bert_output_size: int = config.hidden_size

        self.bert = BertModel.from_pretrained(Config.MODEL_NAME_OR_PATH)
        self.dropout = nn.Dropout(2 * config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

    def resize_token_embeddings(self, size):
        self.bert.resize_token_embeddings(size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        # output is bert's output for all tokens, and it has (batch_size,max_len,config.hidden_size) shape
        output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False)

        # create a mask from where <e1> and <e2> tokens are placed in input
        e1_mask = (input_ids == self.e1_start_token_id).unsqueeze(2).repeat(1, 1, self.bert_output_size)
        e2_mask = (input_ids == self.e2_start_token_id).unsqueeze(2).repeat(1, 1, self.bert_output_size)

        # extract e1 and e2 tokens from bert output
        e1_token = torch.sum(torch.where(e1_mask, output, torch.zeros_like(output)), dim=1)
        e2_token = torch.sum(torch.where(e2_mask, output, torch.zeros_like(output)), dim=1)

        result = torch.cat((e1_token, e2_token), dim=1)

        pooled_output = self.dropout(result)
        logits = self.classifier(pooled_output)
        return logits
