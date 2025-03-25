import eventlet
eventlet.patcher.import_patched('requests.__init__')
from flask import Flask, render_template, redirect, url_for, request
import os
from flask_socketio import SocketIO, send, emit
import random
from rag import MistralRAGAgentRemote

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

eventlet.monkey_patch()

assistant = MistralRAGAgentRemote()

@app.route('/')
def index():
    return render_template("index.html")
@app.route("/chat")
def chat():
    assistant.prepare_all()
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.pdf'):
        file.save(os.path.join("uploads", file.filename))
        assistant.ingest_file(os.path.join("uploads", file.filename))
        os.remove(os.path.join("uploads", file.filename))
        return "File uploaded successfully", 200
    return "Invalid file type", 400
@socketio.on('start_chat')
def handle_message(msg):
    def stream_response():
        with app.app_context():
            """Iterates over the generator and emits tokens."""
            for token in assistant.respond(msg):
                socketio.emit("chat_message", token)  # Emit each token
                eventlet.sleep(0)  # Yield control to event loop
            socketio.emit("end_message", "")

    socketio.start_background_task(stream_response)  # Run the function in the background

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host="0.0.0.0", port="8080")