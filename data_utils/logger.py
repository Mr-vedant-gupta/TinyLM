from typing import Dict, Any

import json
import wandb
from omegaconf import DictConfig, OmegaConf


class WandBLogger:
    """Logger class for tracking training metrics and model outputs using Weights & Biases."""
    
    def __init__(self, cfg: DictConfig, model: Any):
        """Initialize the Weights & Biases logger.
        
        Args:
            cfg: Configuration object containing logging parameters
            model: PyTorch model to track
            
        Raises:
            ValueError: If WandB name is empty
        """
        if not cfg.WandB.name:
            raise ValueError("WandB name cannot be empty")

        self.run = wandb.init(
            project=cfg.WandB.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir="../scratch/wandb",
            name=cfg.WandB.name
        )
        wandb.watch(model)

        # Initialize generation tracking
        self.generation_table_columns = list(["step"] + cfg.validation.prompts)
        self.generation_table = wandb.Table(columns=self.generation_table_columns)
        self.generation_table_rows = []
        self.generation_file = f"{cfg.data.checkpoint_dir}/{cfg.WandB.name}/generation_samples.jsonl"

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics to Weights & Biases.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        wandb.log(metrics, step=step)

    def log_text_samples(self, samples: Dict[str, str], step: int) -> None:
        """Log generated text samples to file (there is a bug in the W&B table API at the moment)
        
        Args:
            samples: Dictionary mapping prompt names to generated text
            step: Current training step
        """
        # Save to JSONL file
        entry = {"step": step, "samples": samples}
        with open(self.generation_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


