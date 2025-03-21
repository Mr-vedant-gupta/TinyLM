import os
import hydra
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
import random

from transformer.transformer import CausalDecoderTransformer
from data_utils.dataloader import create_dataloaders
from data_utils.tokenizer import Tokenizer
from data_utils.logger import WandBLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyLM:
    def __init__(self, cfg):
        print("Using device {}".format(device))
        self.tlm = CausalDecoderTransformer(cfg).to(device)
        self.tokenizer = Tokenizer(cfg)
        self.optimizer = torch.optim.AdamW(self.tlm.parameters(), lr=cfg.train.learning_rate,
            betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=cfg.train.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        # Add number of params to the cfg
        self.calculate_model_size(cfg)

        self.logger = WandBLogger(cfg, self.tlm)
        self.train_dataloader, self.valid_dataloader = create_dataloaders(cfg)
        self.gradient_accumulation_steps = cfg.train.gradient_accumulation_steps
        self.grad_clip = cfg.train.grad_clip
        self.eval_interval = cfg.train.eval_interval
        self.eval_iters = cfg.train.eval_iters
        self.checkpoint_interval = cfg.train.checkpoint_interval
        self.prompts = cfg.validation.prompts
        self.temperature = cfg.model.temperature

        if cfg.train.load_checkpoint is not None:
            print("Loading checkpoint from {} ...".format(cfg.train.load_checkpoint))
            self.load_checkpoint(cfg.train.load_checkpoint)
        if cfg.WandB.name == "":
            raise Exception("Need to pass a name to the Logger, got empty string")
        self.checkpoint_dir = f"{cfg.data.checkpoint_dir}/{cfg.WandB.name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.step = 0

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise Exception("No checkpoint found at {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.tlm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']

    def save_checkpoint(self):
        save_path = f"{self.checkpoint_dir}/step{self.step}.pt"
        save_dict = {
         'model_state_dict': self.tlm.state_dict(),
         'optimizer_state_dict': self.optimizer.state_dict(),
         'step': self.step}
        torch.save(save_dict, save_path)

    def calculate_model_size(self, cfg):
        cfg.model.number_parameters = sum([p.numel() for p in self.tlm.parameters() if p.requires_grad])

    def train_step(self):
        self.tlm.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for micro_step in range(self.gradient_accumulation_steps):
            inp, targ, epoch = next(self.train_dataloader)
            inp, targ = inp.to(device), targ.to(device)
            logits = self.tlm(inp)
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targ.reshape(-1))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

        grad_clipped = False
        if self.grad_clip != 0.0:
            grad_clipped = True
            torch.nn.utils.clip_grad_norm_(self.tlm.parameters(), self.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"epoch": epoch, "loss": total_loss, "grad_clipped": int(grad_clipped)}

    @torch.no_grad()
    def validate(self):
        self.tlm.eval()
        total_loss = 0
        for _ in range(self.eval_iters):
            inp, targ, _ = next(self.valid_dataloader)
            inp, targ = inp.to(device), targ.to(device)
            logits = self.tlm(inp)
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targ.reshape(-1))
            total_loss += loss.item() / self.eval_iters

        return {"validation_loss": total_loss}

    @torch.no_grad()
    def generate(self, prompt, temperature):
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        output_tensor = self.tlm.generate_text(input_tensor, self.tokenizer.is_eos, temperature)
        output_text = self.tokenizer.decode(output_tensor[0].cpu().numpy().tolist())
        return output_text


    def generate_samples(self):
        self.tlm.eval()
        samples = {}
        for prompt_name in self.prompts:
            output_text =  self.generate(prompt_name, self.temperature)
            samples[prompt_name] = output_text
        return samples



    def train(self):
        while True:
            # Do a gradient update on the training data
            metrics = self.train_step()
            self.logger.log_metrics(metrics, self.step)
            print(f"Train metrics: {metrics}")
            if self.step % self.eval_interval == 0:
                validation_metrics = self.validate()
                self.logger.log_metrics(validation_metrics, self.step)
                print(f"Validation metrics: {validation_metrics}")
                samples = self.generate_samples()
                self.logger.log_text_samples(samples, self.step)


            if self.step % self.checkpoint_interval == 0:
                self.save_checkpoint()

            self.step += 1




@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    trainer = TinyLM(cfg)
    trainer.train()


if __name__ == "__main__":
    main()