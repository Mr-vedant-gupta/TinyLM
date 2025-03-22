# TinyLM: A PyTorch Implementation of a Causal Decoder Transformer

This repository contains a PyTorch implementation of a causal decoder transformer model, designed for language modeling tasks. The implementation includes training utilities, data processing, and an interactive UI for model interaction.

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                  # Dataset directory
├── data_utils/           # Data processing utilities
├── interactive_ui/       # Interactive UI components
├── saved_checkpoints/    # Model checkpoints
├── transformer/          # Core transformer implementation
├── train.py             # Main training script
└── requirements.txt     # Project dependencies
```

## Features

- Implementation of a causal decoder transformer architecture
- Configurable model architecture and training parameters
- Data processing utilities with tokenization
- Interactive UI for model interaction
- Training with gradient accumulation
- Checkpoint saving and loading
- Weights & Biases integration for experiment tracking
- Hydra configuration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformers.git
cd transformers
```

2. Create and activate a virtual environment:
```bash
python -m venv transformers_env
source transformers_env/bin/activate  # On Unix/macOS
# or
.\transformers_env\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script uses Hydra for configuration management. You can modify the training parameters in the `config/` directory.

### Interactive UI

To use the interactive UI:

```bash
cd interactive_ui
python app.py
```

## Configuration

The model and training parameters can be configured through Hydra configuration files in the `config/` directory. Key parameters include:

- Model architecture (number of layers, hidden size, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data processing settings
- Logging and checkpointing options

## Dependencies

Key dependencies include:
- PyTorch
- Hydra
- Weights & Biases
- NumPy
- Other dependencies listed in `requirements.txt`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]