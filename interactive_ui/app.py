from flask import Flask, request, jsonify, render_template, Response
import json
import time
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
    complete_text = generate(prompt, temperature)

    # Then stream it chunk by chunk
    chunks = []
    total_sent = 0

    # Split by spaces to simulate word-by-word streaming
    words = complete_text.split(' ')

    for i, word in enumerate(words):
        # Add space back except for the first word
        if i > 0:
            word = ' ' + word

        # Yield a chunk
        yield f"data: {json.dumps({'text': word})}\n\n"

        # Short delay to simulate typing
        time.sleep(0.2)

    # Signal end of stream
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    temperature = float(data.get('temperature', 0.7))

    return Response(
        generate_stream(prompt, temperature),
        mimetype='text/event-stream'
    )


if __name__ == '__main__':
    app.run(debug=True)