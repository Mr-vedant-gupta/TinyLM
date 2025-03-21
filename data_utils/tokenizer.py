import os
import numpy as np
import sentencepiece as spm
from omegaconf import DictConfig
import hydra
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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


def train_tokenizer(cfg):
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


def split_datapoints_into_chunks(datapoints, num_chunks = 50):
    chunk_size = len(datapoints) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = None if i == num_chunks - 1 else (i + 1) * chunk_size
        chunks.append(datapoints[start:end])
    return chunks

def process_chunk(datapoints_chunk, cfg):
    tokenizer = Tokenizer(cfg)
    tokenized_chunk = []
    for datapoint in datapoints_chunk:
        # Remove any extra whitespace and encode with BOS and EOS markers
        tokens = tokenizer.encode(datapoint.strip(), bos=True, eos=True)
        tokenized_chunk.extend(tokens)
    return tokenized_chunk

def tokenize_data(cfg):
    tokenizer = Tokenizer(cfg)

    def tokenize_file(inp, out):
        with open(inp, "r") as f:
            data = f.read()
            datapoints = data.split(cfg.data.EOT_token)
        tokenized_datapoints = []
        print("Chunking datapoints...")
        chunks = split_datapoints_into_chunks(datapoints)
        print("Done splitting into chunks")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, chunk, cfg) for chunk in chunks]

            # Optionally, use tqdm to show a progress bar.
            for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
                tokenized_datapoints.extend(future.result())

        tokenized_datapoints = np.array(tokenized_datapoints, dtype=np.uint16)
        with open(out, "wb") as f:
            f.write(tokenized_datapoints.tobytes())

    tokenize_file(cfg.data.train, cfg.tokenization.tokenized_train)
    tokenize_file(cfg.data.valid, cfg.tokenization.tokenized_valid)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
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

