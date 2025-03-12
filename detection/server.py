import io
import json
import cv2
from fastapi import FastAPI, File, Form, Query, Response, UploadFile, HTTPException
from fastapi.concurrency import asynccontextmanager

from uuid import UUID
import httpx
import numpy as np

from schemas.output_detect_schema import RootModel
from schemas.schemas import CreateRequest, DeleteRequest, DictResponse, SimpleResponse
from models_config import models_config
from builders.general_builder import GeneralBuilder
from server_config import Settings
import asyncio
import gradio as gr
import PIL.Image as Image
from server_config import Settings
from api.predict import predict_router
from api.base_models import created_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    settings = Settings()
    global active_processes
    active_processes = asyncio.Semaphore(settings.num_cores - 1)  
    yield

app = FastAPI(lifespan=lifespan, title="Detection", description="API for your detection models")
 
app.include_router(predict_router, prefix="/predict")

# Получение всех текущих на данный момент созданных загруженных моделей
@app.get("/", response_model=DictResponse)
async def get_models():
    models_inform = {}
    for uuid, wrapper in created_models.items():
        models_inform[uuid] = (wrapper.model_type, wrapper.model_name)
    return DictResponse(dict=models_inform)

'''
@app.post("/", response_model=SimpleResponse)
async def create_model(create_request: CreateRequest):
    model_type = create_request.model_type
    model_name = create_request.model_name
    
    try:
        model_builder = GeneralBuilder(model_type, model_name)
        detector = model_builder.build()
        created_models[detector.uuid] = detector
    except KeyError:
         raise HTTPException(status_code=400, detail=f"Нельзя создать `{model_type}`, выберите что-то из {list(models_config.keys())}")
    
    return SimpleResponse(message=f"Модель {model_name} создана.", uuid=f"{detector.uuid}")

# Выгрузка модели из загруженных в текущий момент
@app.post("/unload", response_model=SimpleResponse)
async def delete_model(delete_request: DeleteRequest):
    model_uuid = delete_request.model_uuid
    print(created_models.keys())
    if model_uuid not in created_models.keys():
        raise HTTPException(status_code=404, detail=f"Модель {model_uuid} не загружена")
    
    del created_models[model_uuid]
    return SimpleResponse(message=f"Модель выгружена.", uuid=f"{model_uuid}")

# Сохранение модели в папку /models
@app.post("/save_model", response_model=SimpleResponse)
async def save_model(input_uuid: UUID = Query(...)):
    created_models[str(input_uuid)].save(settings.model_dir) 
    return SimpleResponse(message=f"Модель {created_models[str(input_uuid)].model_name} скачана.", uuid=f"{input_uuid}")

# Загрузка модели c сервера по имени
@app.post('/load_model', response_model=SimpleResponse)
async def load_model(create_request: CreateRequest):
    if len(created_models) >= settings.max_inference_models:
        raise HTTPException(status_code=400, detail="Достигнуто максимальное количество загруженных моделей")
    
    model_type = create_request.model_type
    model_name = create_request.model_name

    try:
        model_builder = GeneralBuilder(model_type, model_name)
        detector = model_builder.build().load()
        created_models[detector.uuid] = detector
    except KeyError:
         raise HTTPException(status_code=400, detail=f"Нельзя загрузить `{model_type}`, выберите что-то из {list(models_config.keys())}")
    
    return SimpleResponse(message=f"Модель {model_name} загружена.", uuid=f"{detector.uuid}")
        
# Запуск обучения 
@app.post("/train")
async def train(name: str = Query(...), input_uuid: UUID = Query(...), configdict = Query(...)):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.url}/runner/run/",
                params={"name": name, "input_uuid": str(input_uuid), "configdict": json.dumps(configdict)},
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"Ошибка:": response.status_code}

predict_iface = gr.Interface(
    fn=plot_results,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        'text',
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Predict",
    description="Upload images for inference.",
)       

app = gr.mount_gradio_app(app, predict_iface, path="/plotter")'
'''

