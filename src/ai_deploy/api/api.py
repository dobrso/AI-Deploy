from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from ai_deploy.models.text_model import init_text_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.text_model = init_text_model()
    yield

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str

@app.get('/health')
def health():
    if app.state.text_model:
        return {'status': 'ok'}
    return {'status': 'fail'}

@app.post('/predict')
def predict(request: PredictRequest):
    result = app.state.text_model(request.text)
    return {'result': result}