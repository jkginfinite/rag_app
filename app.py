from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/')
def home():
    return '✅ Your Flask app is running on Render!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    return jsonify({
        "question": question,
        "answer": f"You asked: '{question}' — here's a simple placeholder response."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
