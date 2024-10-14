import config
import torch

class HFDataset:
    def __init__(self, tweet, label=None):
        self.tweet = tweet
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        tweet = str(self.tweet[item])

        inputs = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt"
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]


        # print("\n####SPECIAL TOKENS to IDS MAP:####")
        # special_tokens_map = self.tokenizer.special_tokens_map
        # print(f"[CLS]: {self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)}")
        # print(f"[SEP]: {self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)}")
        # print(f"[PAD]: {self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)}")
        # print(f"[MASK]: {self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)}")
        # print("\n")

        if self.label is not None:
            return {
                    "ids": ids.long(),
                    "mask": mask.long(),
                    "token_type_ids": token_type_ids.long(),
                    "targets": torch.tensor(self.label[item], dtype=torch.float)
                }
        else:
            return {
                    "ids": ids.long(),
                    "mask": mask.long(),
                    "token_type_ids": token_type_ids.long()
                }
        
if __name__ == "__main__":
    import pandas as pd
    valid_data_path = "data/taskc/valid.csv"
    valid_df = pd.read_csv(valid_data_path)

    tweet = valid_df.tweet.values
    label = valid_df.label.values
    
    # dataset = HFDataset(tweet=tweet, label=label)
    dataset = HFDataset(tweet=["@cmprachanda, @ncp_madhavnepal , @SherBDeuba , @NepaliCongress नेपाली औँसी  :new_moon_face:"], label=[0])
    # 100 = '[UNK]'
    special_tokens = dataset.tokenizer.special_tokens_map
    special_token_ids = {token: dataset.tokenizer.convert_tokens_to_ids(token) for token in special_tokens.values()}
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, num_workers=4
    )

    for item in data_loader:
        breakpoint()
    print(len(dataset))
    print(item)
    # print(dataset[0])


    