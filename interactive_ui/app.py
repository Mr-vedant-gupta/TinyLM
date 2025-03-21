from flask import Flask, request, jsonify, render_template, Response
import json
import time
import hydra
from omegaconf import DictConfig
from train import TinyLM
from functools import partial
import threading

# Import your LLM generation function
# Assuming it's in a file called tiny_llm.py in the same directory

def generate(prompt, temperature):
    return "this is a dummy sentence"

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def generate_stream(prompt, temperature):
    """Generate text and yield it in chunks to simulate streaming."""
    # First, get the complete generated text
    complete_text = tiny_lm.generate(prompt, temperature)
    complete_text = complete_text[len(prompt):]
    # Split by spaces to simulate word-by-word streaming
    words = complete_text.split(' ')

    for i, word in enumerate(words):
        # Add space back except for the first word
        if i > 0:
            word = ' ' + word

        # Yield a chunk
        yield f"data: {json.dumps({'text': word})}\n\n"

        # Short delay to simulate typing
        time.sleep(0.1)

    # Signal end of stream
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    temperature = float(data.get('temperature', 0))

    return Response(
        generate_stream(prompt, temperature),
        mimetype='text/event-stream'
    )


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.train.load_checkpoint == "":
        raise Exception("No checkpoint provided")
    global tiny_lm
    tiny_lm = TinyLM(cfg)

    app.run(debug=True)


if __name__ == '__main__':
    main()
