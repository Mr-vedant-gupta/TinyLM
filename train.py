import os
from typing import Dict, Tuple, Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
import random

from transformer.transformer import CausalDecoderTransformer
from data_utils.dataloader import create_dataloaders
from data_utils.tokenizer import Tokenizer
from data_utils.logger import WandBLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyLM:
    """A class to handle training and evaluation of the TinyLM transformer model."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the TinyLM trainer.
        
        Args:
            cfg: Hydra configuration object containing model and training parameters
        """
        print(f"Using device {device}")
        self.tlm = CausalDecoderTransformer(cfg).to(device)
        self.tokenizer = Tokenizer(cfg)
        self.optimizer = torch.optim.AdamW(
            self.tlm.parameters(),
            lr=cfg.train.learning_rate,
            betas=(cfg.train.beta1, cfg.train.beta2),
            weight_decay=cfg.train.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize training parameters
        self.cfg = cfg
        self.calculate_model_size(cfg)
        self.train_dataloader, self.valid_dataloader = create_dataloaders(cfg)
        self.gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
        self.grad_clip = cfg.train.grad_clip
        self.eval_interval = cfg.train.eval_interval
        self.eval_iters = cfg.train.eval_iters
        self.checkpoint_interval = cfg.train.checkpoint_interval
        self.prompts = cfg.validation.prompts
        self.temperature = cfg.model.temperature
        self.step = 0

        if cfg.train.load_checkpoint is not None:
            print(f"Loading checkpoint from {cfg.train.load_checkpoint} ...")
            self.load_checkpoint(cfg.train.load_checkpoint)
        
        if not cfg.WandB.name:
            raise ValueError("WandB name cannot be empty")
            
        self.checkpoint_dir = f"{cfg.data.checkpoint_dir}/{cfg.WandB.name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.tlm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']

    def save_checkpoint(self) -> None:
        """Save current model and optimizer state."""
        save_path = f"{self.checkpoint_dir}/step{self.step}.pt"
        save_dict = {
            'model_state_dict': self.tlm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step
        }
        torch.save(save_dict, save_path)

    def calculate_model_size(self, cfg: DictConfig) -> None:
        """Calculate and store the total number of trainable parameters."""
        cfg.model.number_parameters = sum(
            [p.numel() for p in self.tlm.parameters() if p.requires_grad]
        )

    def train_step(self) -> Dict[str, Any]:
        """Perform a single training step with gradient accumulation.
        
        Returns:
            Dictionary containing training metrics
        """
        self.tlm.train()
        self.optimizer.zero_grad()
        total_loss = 0
        
        for _ in range(self.gradient_accumulation_steps):
            inp, targ, epoch = next(self.train_dataloader)
            inp, targ = inp.to(device), targ.to(device)
            logits = self.tlm(inp)
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                targ.reshape(-1)
            )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

        if self.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.tlm.parameters(), self.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {"epoch": epoch, "loss": total_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute validation loss.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.tlm.eval()
        total_loss = 0
        
        for _ in range(self.eval_iters):
            inp, targ, _ = next(self.valid_dataloader)
            inp, targ = inp.to(device), targ.to(device)
            logits = self.tlm(inp)
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                targ.reshape(-1)
            )
            total_loss += loss.item() / self.eval_iters

        return {"validation_loss": total_loss}

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        self.tlm.eval()
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        output_tensor = self.tlm.generate_text(
            input_tensor,
            self.tokenizer.is_eos,
            temperature
        )
        return self.tokenizer.decode(output_tensor[0].cpu().numpy().tolist())

    def generate_samples(self) -> Dict[str, str]:
        """Generate samples for all configured prompts.
        
        Returns:
            Dictionary mapping prompt names to generated text
        """
        self.tlm.eval()
        return {
            prompt_name: self.generate(prompt_name, self.temperature)
            for prompt_name in self.prompts
        }

    def train(self) -> None:
        """Main training loop."""
        self.logger = WandBLogger(self.cfg, self.tlm)
        
        while True:
            # Training step
            metrics = self.train_step()
            self.logger.log_metrics(metrics, self.step)
            print(f"Train metrics: {metrics}")
            
            # Validation and sampling
            if self.step % self.eval_interval == 0:
                validation_metrics = self.validate()
                self.logger.log_metrics(validation_metrics, self.step)
                print(f"Validation metrics: {validation_metrics}")
                
                samples = self.generate_samples()
                self.logger.log_text_samples(samples, self.step)
            
            # Checkpointing
            if self.step % self.checkpoint_interval == 0:
                self.save_checkpoint()
            
            self.step += 1


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    trainer = TinyLM(cfg)
    trainer.train()


if __name__ == "__main__":
    main()