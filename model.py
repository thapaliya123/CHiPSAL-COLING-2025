import config
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class HFAutoModel(nn.Module):
    def __init__(self):
        super(HFAutoModel, self).__init__()
        self.automodel = AutoModelForSequenceClassification.from_pretrained(config.HF_MODEL_PATH,
                                                                            num_labels=config.NUM_LABELS)
        self.freeze_layers() 

    def freeze_layers(self):
        for param in self.automodel.bert.encoder.layer[:20].parameters():
            param.requires_grad = False

    def print_trainable_layers(self):
        for name, param in self.automodel.named_parameters():
            print(f"{name}: {param.requires_grad}")
        
    def forward(self, ids, mask, token_type_ids):
        output = self.automodel(ids,
                       token_type_ids=token_type_ids,
                       attention_mask=mask)
        
        return output.logits
        


if __name__ == "__main__":
    model = HFAutoModel()
    print(model)
