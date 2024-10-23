from fastapi import APIRouter

router = APIRouter()

@router.post("/upload")
def upload():
    return {"message": "Service is available"}