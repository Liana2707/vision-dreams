from typing import List, Dict, Optional

from pydantic import BaseModel, ConfigDict


class XYWHN(BaseModel):
    xn: float
    yn: float
    wn: float
    hn: float


class TLBR(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class OutputItem(BaseModel):
    class_id: int
    class_name: str
    score: float
    xywhn: XYWHN
    tlbr: TLBR


class ModelDescription(BaseModel):
    id: str
    name: str
    type: str
    problem_type: str


class RootModel(BaseModel):
    model_description: ModelDescription
    model_output: List[List[OutputItem]]

