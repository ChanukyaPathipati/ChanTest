from sentence_transformers import SentenceTransformer

# Load the pretrained embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()