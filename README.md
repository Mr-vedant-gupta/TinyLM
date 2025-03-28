# TinyLM: A PyTorch Implementation of a Causal Decoder Transformer

A PyTorch implementation of a causal decoder transformer model for training tiny language models. The implementation includes training utilities, data processing, and an interactive UI for model interaction.


## Sample Generation

Below is a sample generation after ~10 hours of training on a single Quadro RTX 6000 using the [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

**Prompt**: "Once upon a time there was a little girl named Lucy"

**Output**:
```
Once upon a time there was a little girl named Lucy. She was three years old and loved to explore the world around her. One day, Lucy was walking in the park when she saw a big, red ball. She wanted to play with it, so she ran over to it.

"What is this?" Lucy asked.

"It's a ball," said a friendly voice.\nLucy looked around and saw a little boy. He was wearing a blue shirt and had a big smile on his face.

"Can I play with you?" Lucy asked.\nThe little boy nodded and they started to play together. They had so much fun that they didn't notice the time passing by. Suddenly, the ball started to roll away. Lucy and the little boy ran after it, but it was too fast.

"Let's catch it!" Lucy said.

They ran and ran until they finally caught the ball.

"That was so much fun!" said the little boy.

"Yes, it was!" said Lucy.\nThey both smiled and hugged each other. Then they went back to playing with the ball, happy to have made a new friend.
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

