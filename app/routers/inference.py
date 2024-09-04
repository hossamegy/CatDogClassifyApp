from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from controllers.inference import Inference

inference_router = APIRouter()

@inference_router.post('/inference')
async def get_inference(file: UploadFile = File(...)):
    inference = Inference()
    prediction = await inference.inference_func(file)
    return JSONResponse(
        content=prediction
    )
