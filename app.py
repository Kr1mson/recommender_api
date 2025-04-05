from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from flask_cors import CORS
import re
import unicodedata
import string

def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = unicodedata.normalize('NFKC', text)

    text = ''.join(c for c in text if c in string.printable)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading data and model...")
df = pd.read_csv('data.csv')

model = SentenceTransformer("all-MiniLM-L6-v2")
loaded_index = faiss.read_index('index_file.index')

app = Flask(__name__)
CORS(app)
@app.route('/recommend', methods=['POST'])
def recommend_games():
    user_query = request.json.get("query")
    user_query = clean_text(user_query) if user_query else None
    top_n = request.json.get("top_n", 5)

    if not user_query:
        return jsonify({"error": "Query not provided."}), 400

    query_embedding = model.encode([user_query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    distances, indices = loaded_index.search(query_embedding, top_n)

    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        game_info = {
            "cover": df.iloc[idx]["cover"],
            "score": float(dist)
        }
        recommendations.append(game_info)

    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)