# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Needed to allow requests from your web app (different origin)

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

@app.route('/reverse_string', methods=['POST'])
def reverse_string():
    # Get JSON data from the request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data['text']
    reversed_text = input_text[::-1] # Pythonic way to reverse a string

    return jsonify({"original": input_text, "reversed": reversed_text})

if __name__ == '__main__':
    # Run the Flask app on port 5000 (or any other available port)
    app.run(debug=True, port=5000)
