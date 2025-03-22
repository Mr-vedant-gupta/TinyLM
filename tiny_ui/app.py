from typing import Generator, Dict, Any

import json
import time
from functools import partial

import hydra
from flask import Flask, request, jsonify, render_template, Response
from omegaconf import DictConfig

from train import TinyLM

app = Flask(__name__)
tiny_lm = None  # Global model instance


def generate_stream(prompt: str, temperature: float) -> Generator[str, None, None]:
    """Generate text and yield it in chunks to simulate streaming.
    
    Args:
        prompt: Input text prompt
        temperature: Sampling temperature
        
    Yields:
        Server-sent events containing generated text chunks
    """
    # Generate complete text and remove prompt
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


@app.route('/')
def index() -> str:
    """Render the main page."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_text() -> Response:
    """Handle text generation requests.
    
    Returns:
        Server-sent events stream containing generated text
    """
    data = request.json
    prompt = data.get('prompt', '')
    temperature = float(data.get('temperature', 0))
    
    return Response(
        generate_stream(prompt, temperature),
        mimetype='text/event-stream'
    )


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Initialize and run the Flask application.
    
    Args:
        cfg: Configuration object containing model parameters
        
    Raises:
        ValueError: If no checkpoint is provided
    """
    if not cfg.train.load_checkpoint:
        raise ValueError("No checkpoint provided")
        
    global tiny_lm
    tiny_lm = TinyLM(cfg)
    
    app.run(debug=True)


if __name__ == '__main__':
    main()
