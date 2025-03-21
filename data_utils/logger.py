import wandb
from omegaconf import OmegaConf
import json

class WandBLogger:
    def __init__(self, cfg, model):
        if cfg.WandB.name == "":
            raise Exception("Need to pass a name to the Logger, got empty string")

        self.run = wandb.init(project=cfg.WandB.project, config = OmegaConf.to_container(cfg, resolve=True),
                   dir="../scratch/wandb", name=cfg.WandB.name)
        wandb.watch(model)

        self.generation_table_columns = list(["step"] + cfg.validation.prompts)
        self.generation_table = wandb.Table(columns = self.generation_table_columns)
        self.generation_table_rows = []
        self.generation_file = f"{cfg.data.checkpoint_dir}/{cfg.WandB.name}/generation_samples.jsonl"

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step = step)

    def log_text_samples(self, samples, step):
        entry = {"step": step, "samples": samples}
        with open(self.generation_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


