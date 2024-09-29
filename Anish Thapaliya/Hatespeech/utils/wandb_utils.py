import os
import wandb
import config
from enums import wandb_enums

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

class WandbLogger:
    def __init__(self, run_name: str, config: dict, is_log: bool = True):
        self.config = config
        self.run_name = run_name
        self.is_log = is_log

    def initialize(self):
        if self.is_log:
            wandb.login(key=WANDB_API_KEY)
            
            self.run = wandb.init(
                project=self.config.get(wandb_enums.WandbEnum.WANDB_PROJECT_NAME.value),
                config=self.config,
                tags=self.config.get(wandb_enums.WandbEnum.WANDB_TAGS.value),
                name=self.run_name
            )
    def log_metrics(self, metrics_dict: dict):
        if self.is_log:
            wandb.log(metrics_dict)

    def save_model_artifact(self, model_file, model_name="model"):
        if self.is_log:
            print(os.listdir("models"))
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(model_file)
            self.run.log_artifact(artifact)
            print(f"Model artifact {model_name} logged to WandB as {model_file}.")

    def load_model_from_artifact(self):
        pass
    
    def finish(self):
        if self.is_log:
            wandb.finish()
            print("WandB run Finished")



if __name__ == "__main__":
    WANDB_PROJECT_NAME = "NLP CHIPSAL COLING 2025"
    TAGS = ["muril-base", "taskc"]
    CONFIG_DICT = {
                wandb_enums.WandbEnum.WANDB_PROJECT_NAME.value: WANDB_PROJECT_NAME,
                wandb_enums.WandbEnum.WANDB_TAGS.value: TAGS,
                "epochs": 5,
                "batch_size": 128,
                "lr": 1e-3 
    }
    WANDB_RUN_NAME = "test-run"

    wandb_logger = WandbLogger(WANDB_RUN_NAME, CONFIG_DICT)
    wandb_logger.initialize()
    wandb_logger.save_model_artifact("models/muril-base-cased-f1_score-0.2178030303030303.bin")
