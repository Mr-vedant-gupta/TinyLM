import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import sentencepiece as spm
from omegaconf import DictConfig
import hydra
from tqdm import tqdm


class Tokenizer:
    """SentencePiece-based tokenizer for text processing."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the tokenizer with a trained SentencePiece model.
        
        Args:
            cfg: Configuration object containing tokenizer parameters
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        model_location = f"{cfg.tokenization.model_prefix}.model"
        if not os.path.exists(model_location):
            raise FileNotFoundError(
                f"Model file not found at: {model_location}. Make sure to train tokenizer first"
            )
        self.sp_model = spm.SentencePieceProcessor(model_file=model_location)
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()

    def encode(self, s: str, bos: bool , eos: bool) -> List[int]:
        """Encode text into token IDs.
        
        Args:
            s: Input text to encode
            bos: Whether to add beginning-of-sequence token
            eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        tokens = self.sp_model.encode(s)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back into text.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            Decoded text
        """
        return self.sp_model.decode(tokens)

    def is_eos(self, token: int) -> bool:
        """Check if a token is the end-of-sequence token.
        
        Args:
            token: Token ID to check
            
        Returns:
            True if the token is EOS, False otherwise
        """
        return token == self.eos_id


def train_tokenizer(cfg: DictConfig) -> None:
    """Train a new SentencePiece tokenizer model.
    
    Args:
        cfg: Configuration object containing tokenizer training parameters
    """
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


def split_datapoints_into_chunks(datapoints: List[str], num_chunks: int = 50) -> List[List[str]]:
    """Split a list of datapoints into roughly equal chunks.
    
    Args:
        datapoints: List of text datapoints to split
        num_chunks: Number of chunks to split into
        
    Returns:
        List of datapoint chunks
    """
    chunk_size = len(datapoints) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = None if i == num_chunks - 1 else (i + 1) * chunk_size
        chunks.append(datapoints[start:end])
    return chunks


def process_chunk(datapoints_chunk: List[str], cfg: DictConfig) -> List[int]:
    """Process a chunk of datapoints into token IDs.
    
    Args:
        datapoints_chunk: List of text datapoints to process
        cfg: Configuration object containing tokenizer parameters
        
    Returns:
        List of token IDs
    """
    tokenizer = Tokenizer(cfg)
    tokenized_chunk = []
    for datapoint in datapoints_chunk:
        # Remove any extra whitespace and encode with BOS and EOS markers
        tokens = tokenizer.encode(datapoint.strip(), bos=True, eos=True)
        tokenized_chunk.extend(tokens)
    return tokenized_chunk


def tokenize_data(cfg: DictConfig) -> None:
    """Tokenize training and validation data using parallel processing.
    
    Args:
        cfg: Configuration object containing data and tokenizer parameters
    """
    tokenizer = Tokenizer(cfg)

    def tokenize_file(inp: str, out: str) -> None:
        """Tokenize a single file and save the results.
        
        Args:
            inp: Input file path
            out: Output file path
        """
        with open(inp, "r") as f:
            data = f.read()
            datapoints = data.split(cfg.data.EOT_token)
            
        tokenized_datapoints = []
        print("Chunking datapoints...")
        chunks = split_datapoints_into_chunks(datapoints)
        print("Done splitting into chunks")
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_chunk, chunk, cfg)
                for chunk in chunks
            ]
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Tokenizing"
            ):
                tokenized_datapoints.extend(future.result())

        tokenized_datapoints = np.array(tokenized_datapoints, dtype=np.uint16)
        with open(out, "wb") as f:
            f.write(tokenized_datapoints.tobytes())

    tokenize_file(cfg.data.train, cfg.tokenization.tokenized_train)
    tokenize_file(cfg.data.valid, cfg.tokenization.tokenized_valid)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for tokenizer training and data tokenization.
    
    Args:
        cfg: Configuration object containing all necessary parameters
        
    Raises:
        FileNotFoundError: If training or validation data files don't exist
    """
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

