import transformers
import torch.nn as nn


class BERTBaseCased(nn.Module):
    def __init__(self):
        super(BERTBaseCased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
                ids,
                attention_mask,
                token_type_ids=token_type_ids
        )
        bo = self.bery_drop(o2)
        output = self.out(bo)
        return output
