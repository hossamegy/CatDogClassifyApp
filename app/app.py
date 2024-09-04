from fastapi import FastAPI
from routers import base_router, inference_router

app = FastAPI()

app.include_router(base_router)

app.include_router(inference_router)