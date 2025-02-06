from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class CreateRequest(BaseModel):
    model_type: str
    model_name: str

class DeleteRequest(BaseModel):
    model_uuid: str


class Response(BaseModel):
    message: str
    uuid: str

class DictResponse(BaseModel):
    dict: dict

class ErrorResponse(BaseModel):
    message: str

