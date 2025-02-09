import torch
import torch.nn as nn
from transformers import DistilBertModel

class DistilBertEmbedder(nn.Module):
    def __init__(self, pretrained_name="distilbert-base-uncased", dropout=0.1, use_mean_pooling=True):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.dim

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.use_mean_pooling = use_mean_pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  

        if self.use_mean_pooling:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            emb = self.projection(pooled)
        else:
            cls_emb = last_hidden[:, 0]
            emb = self.projection(cls_emb)

        return emb  
