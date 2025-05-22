
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import get_embedding

# Import prometheus middleware and metrics handler
from starlette_exporter import PrometheusMiddleware, handle_metrics

app = FastAPI()

# Add Prometheus middleware to collect metrics automatically
app.add_middleware(PrometheusMiddleware)

# Add a new route to expose metrics
@app.get("/metrics")
async def metrics():
    return await handle_metrics()

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



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from app.model import get_embedding

# app = FastAPI()

# class TextInput(BaseModel):
#     text: str

# @app.post("/predict")
# def predict(data: TextInput):
#     try:
#         embedding = get_embedding(data.text)
#         return {"embedding": embedding}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# def health():
#     return {"status": "ok"}
