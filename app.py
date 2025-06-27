from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import os
import gdown

app = Flask(__name__)


MODEL_URL = "https://drive.google.com/uc?id=1dJzq9ixrvTL0eAtKzwrm1RfGqtknfoKV"
MODEL_PATH = "bambara-translator/model.safetensors"

# Télécharger le modèle s'il n'existe pas déjà
if not os.path.exists(MODEL_PATH):
    os.makedirs("bambara_model", exist_ok=True)
    print("🔽 Téléchargement du modèle depuis Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Ensuite charge ton modèle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bambara_model")
tokenizer = AutoTokenizer.from_pretrained("bambara_model")
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
    
