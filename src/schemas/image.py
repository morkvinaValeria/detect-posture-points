from pydantic import BaseModel


class Image(BaseModel):
    base64: str
