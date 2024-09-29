import config
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class HFAutoModel(nn.Module):
    def __init__(self):
        super(HFAutoModel, self).__init__()
        self.automodel = AutoModelForSequenceClassification.from_pretrained(config.HF_MODEL_PATH,
                                                                            num_labels=config.NUM_LABELS)
        
    def forward(self, ids, mask, token_type_ids):
        # print(ids.shape)
        # print(mask.shape)
        # print(token_type_ids.shape)
        output = self.automodel(ids,
                       token_type_ids=token_type_ids,
                       attention_mask=mask)
        
        # print(f"Model Output: {type(output)} --> {output.logits.shape}")
        return output.logits
        


if __name__ == "__main__":
    pass