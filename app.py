from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy

app = Flask(__name__)
CORS(app)

nlp = spacy.load("model")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    return jsonify(entities)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)