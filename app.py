from flask import Flask, request, jsonify
app = Flask(__name__)
from rag import RAG
import yaml
import os

from flask_cors import CORS
CORS(app)


configFile = '25650.yaml'
def load_config(path=f"configFiles/{configFile}"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
chunk_size = config['chunk_size']
overlap = config['overlap']
text_docs = config['text_docs']
load_model = config['load_model']
embedding_model = config['embedding_model']
retrieval_model = config['retrieval_model']
ranking_model = config['ranking_model']
rag = RAG(load_model,text_docs, chunk_size, overlap, embedding_model, ranking_model, retrieval_model)



@app.route('/')
def home():
    return '✅ Your Flask app is running on Render!'

@app.route('/', methods=["GET"])
def index():
    return "Hello from RAG backend!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    #question = data.get("question", "")
    question = "What is Joe's background"
    answer = rag.ask(question)
    return jsonify({
        "question": question,
        "answer": f"{answer}" #f"You asked: '{question}' — here's a simple placeholder response."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
