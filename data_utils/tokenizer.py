import os
import numpy as np
import sentencepiece as spm
from omegaconf import DictConfig, OmegaConf
import hydra


class Tokenizer:
    def __init__(self, cfg):
        model_location = f"{cfg.tokenization.model_prefix}.model"
        if not os.path.exists(model_location):
            raise FileNotFoundError(f"Model file not found at: {model_location}. Make sure to train tokenizer first")
        self.sp_model = spm.SentencePieceProcessor(model_file=model_location)
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()

    def encode(self, s: str, bos: bool, eos: bool):
        tokens = self.sp_model.encode(s)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens):
        return self.sp_model.decode(tokens)

    def is_eos(self, token: int) -> bool:
        return token == self.eos_id


def train_tokenizer(cfg) -> None:
    spm.SentencePieceTrainer.train(
        input=cfg.data.train,
        model_prefix=cfg.tokenization.model_prefix,
        model_type="bpe",
        vocab_size=cfg.tokenization.vocab_size,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r"\342\201\207 ",
        normalization_rule_name="identity",
        control_symbols=[cfg.data.EOT_token]
    )


def tokenize_data(cfg):
    tokenizer = Tokenizer(cfg)

    def tokenize_file(inp, out):
        with open(inp, "r") as f:
            data = f.read()
            datapoints = data.split(cfg.data.EOT_token)
        tokenized_datapoints = []
        for datapoint in datapoints:
            tokenized_datapoint = tokenizer.encode(datapoint.strip(), bos=True, eos=True)
            tokenized_datapoints.append(tokenized_datapoint)

        tokenized_datapoints = np.array(tokenized_datapoints, dtype=np.uint16)
        with open(out, "wb") as f:
            f.write(tokenized_datapoints.tobytes())

    tokenize_file(cfg.data.train, cfg.tokenization.tokenized_train)
    tokenize_file(cfg.data.valid, cfg.tokenization.tokenized_valid)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.data.train):
        raise FileNotFoundError(f"Training data not found at {cfg.data.train}")
    if not os.path.exists(cfg.data.valid):
        raise FileNotFoundError(f"Validation data not found at {cfg.data.valid}")
    # Train the tokenizer
    train_tokenizer(cfg)
    # Tokenize the training data
    tokenize_data(cfg)

if __name__ == "__main__":
    main()

