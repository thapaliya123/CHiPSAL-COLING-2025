import config
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class HFAutoModel(nn.Module):
    def __init__(self):
        super(HFAutoModel, self).__init__()
        self.automodel = AutoModelForSequenceClassification.from_pretrained(config.HF_MODEL_PATH,
                                                                            num_labels=config.NUM_LABELS)
        # self.freeze_layers() 

    # def freeze_layers(self):
    #     for param in self.automodel.bert.encoder.layer[:10].parameters():
    #         param.requires_grad = False

    # def print_trainable_layers(self):
    #     for name, param in self.automodel.named_parameters():
    #         print(f"{name}: {param.requires_grad}")
        
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
    # model = HFAutoModel()

    # print(model)
    pass

    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.automodel.bert.encoder.layer[:10].parameters():
    #     param.requires_grad = False

    # # for param in model.automodel.bert.parameters():
    # #     param.requires_grad = False

    # for param in model.automodel.classifier.parameters():
    #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
