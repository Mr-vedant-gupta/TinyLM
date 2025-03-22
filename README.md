# TinyLM: A PyTorch Implementation of a Causal Decoder Transformer

A PyTorch implementation of a causal decoder transformer model for training tiny language models. The implementation includes training utilities, data processing, and an interactive UI for model interaction.


## Sample Generation

Below is a sample generation after ~10 hours of training on a single Quadro RTX 6000 using the [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

**Prompt**: "Once upon a time there was a little girl named Lucy"

**Output**:
```
Once upon a time there was a little girl named Lucy. She was three years old and loved to explore. One day, she went to the park with her mom.

At the park, Lucy saw a big tree with lots of leaves. She wanted to climb it, so she asked her mom if she could. Her mom said yes and Lucy started to climb.

As she climbed higher and higher, she noticed something strange. It was a big, round, green caterpillar! Lucy was so excited and wanted to pick it up.

She reached out and grabbed the caterpillar. But then, the caterpillar started to move! Lucy was so surprised and started to cry.

Her mom came over and said, "Don't worry, Lucy. I'll help you." She took the caterpillar home and put it in a jar.

Lucy was so happy to have a new friend. She hugged her mom and said, "Thank you for helping me!"
```

A sample checkpoint can be found in `saved_checkpoints` and a file showing the progression of text generation capabilities during training can be found in `data/`.

## Project Structure

```
.
├── config/               # Configuration files (Hydra)
├── data/                # Dataset directory
├── data_utils/          # Data processing utilities
│   ├── logger.py       # Weights & Biases logging
│   ├── tokenizer.py    # Custom tokenizer implementation
│   └── dataloader.py   # Data loading utilities
├── interactive_ui/      # Web UI components
│   ├── app.py         # Flask application
│   └── templates/     # HTML templates
├── saved_checkpoints/   # Model checkpoints
├── transformer/        # Core transformer implementation
│   └── transformer.py # Transformer model architecture
└── train.py           # Main training script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mr-vedant-gupta/TinyLM.git
cd tiny-lm
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Data Preparation

1. Download the training and validation datasets into the `data/` directory:
   - [Training data](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt)
   - [Validation data](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-valid.txt)

   Both datasets should be text files with training examples separated by `data.EOTtoken` (in `config/config.yaml`)

2. Train the tokenizer and tokenize the datasets:
```bash
python data_utils/tokenizer.py
```

### Training

Train the model with Weights & Biases logging:
```bash
python train.py WandB.name=<training_run_name> train.debug=False
```

The training script uses Hydra for configuration management. Training parameters can be modified in the `config/` directory.

### Interactive UI

Launch the web interface to interact with a trained model:
```bash
python tiny_ui/app.py train.load_checkpoint=saved_checkpoints/sample_checkpoint.pt WandB.name=dummy
```

