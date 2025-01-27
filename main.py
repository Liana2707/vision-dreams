import json
from typing import Annotated

from fastapi import Form
from fastapi import FastAPI

from builders.general_builder import GeneralBuilder

app = FastAPI()


@app.get("/get_models")
async def root():
    return { "models": GeneralBuilder().get_models()}

