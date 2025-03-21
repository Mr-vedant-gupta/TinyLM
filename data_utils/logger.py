import wandb

class WandBLogger:
    def __init__(self, cfg, model):
        if cfg.WandB.name == "":
            raise Exception("Need to pass a name to the Logger, got empty string")

        wandb.init(project=cfg.WandB.project, config=cfg, dir="../scratch/wandb", name=cfg.WandB.name)
        wandb.watch(model)

        self.generation_table_columns = ["step"] + cfg.validation.prompts
        self.generation_table = wandb.Table(columns = self.generation_table_columns)

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step = step)

    def log_text_samples(self, samples, step):
        generation_samples = [samples[prompt] for prompt in self.generation_table_columns]
        self.generation_table.add_data(step, *generation_samples)
