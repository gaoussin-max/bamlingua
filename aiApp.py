from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Charger le modèle
model_path = "./bambara-translator"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Fonction de traduction
def traduire_texte(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API REST
@app.route('/traduire', methods=['POST'])
def traduire():
    data = request.json
    texte = data.get("texte")
    if not texte:
        return jsonify({"erreur": "Aucun texte fourni"}), 400

    traduction = traduire_texte(texte)
    return jsonify({"traduction": traduction})

if __name__ == '__main__':
    app.run(debug=True)