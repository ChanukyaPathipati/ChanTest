from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import get_embedding

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    try:
        embedding = get_embedding(data.text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}