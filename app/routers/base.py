from fastapi import APIRouter
from fastapi.responses import JSONResponse 

base_router = APIRouter()

@base_router.get('/')
def get_wellcom():

    return JSONResponse(
        content='hello, it work'
    )