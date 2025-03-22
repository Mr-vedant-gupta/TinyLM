# TinyLM: A PyTorch Implementation of a Causal Decoder Transformer

A PyTorch implementation of a causal decoder transformer model to train tiny language models. The implementation includes training utilities, data processing, and an interactive UI for model interaction.

## Sample Generation

Below is a sample generation after ~10 hours of training on a single Quadro RTX 6000 using the [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

**Prompt**: "Once upon a time there was a little girl named Lucy"

**Output**:
Once upon a time there was a little girl named Lucy. She was three years old and loved to explore. One day, she went to the park with her mom.\nAt the park, Lucy saw a big tree with lots of leaves. She wanted to climb it, so she asked her mom if she could. Her mom said yes and Lucy started to climb.\nAs she climbed higher and higher, she noticed something strange. It was a big, round, green caterpillar! Lucy was so excited and wanted to pick it up.\nShe reached out and grabbed the caterpillar. But then, the caterpillar started to move! Lucy was so surprised and started to cry.\nHer mom came over and said, \"Don't worry, Lucy. I'll help you.\" She took the caterpillar home and put it in a jar.\nLucy was so happy to have a new friend. She hugged her mom and said, \"Thank you for helping me!\"

A sample model checkpoint can be found in **checkpoints** and a file showing the progression of text generation capabilities during training can be found in **data**.

## Project Structure

```
.
├── config/               # Config files (used with hydra)
├── data/                 # Dataset directory
├── data_utils/           # Data processing utilities (logger, tokenizer, dataloader)
├── interactive_ui/       # Interactive UI components
├── saved_checkpoints/    # Model checkpoints
├── transformer/          # Core transformer implementation
├── train.py              # Main training script
```

## Usage

### Training
Download train and valid datasets into **data** (e.g. links for TinyStories: [train](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt) [valid](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-valid.txt)). Both datasets should be text files with training examples separated by data.EOTtoken (in **config/config.yaml**)

1. Train the tokenizer and tokenize the training/validation data
```bash
python data_utils/tokenizer.py
```
2. Train a TinyLM with WandB logging
```bash
python train.py WandB.name=<training_run_name> train.debug=False
```

3. Interact with trained TinyLM
```bash
python tiny_ui/app.py train.load_checkpoint=saved_checkpoints/sample_checkpoint.pt WandB.name=dummy

```

The training script uses Hydra for configuration management. Training parameters can be modified in the `config/` directory.
