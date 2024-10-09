from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .routers import health, posture_points

# from core.schemas.prediction import Prediction

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(title="Posture Points Identification API", debug=True,
              swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}, middleware=middleware)

app.include_router(health.router)
app.include_router(posture_points.router)