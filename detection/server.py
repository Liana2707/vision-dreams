import io
import json
from fastapi import FastAPI, File, Query, UploadFile
from uuid import UUID
from fastapi.responses import StreamingResponse
import httpx
import yaml
from wrappers.ssdlite import SSDLiteDetector
from wrappers.yolo_transformer import YOLOTransformerDetector
from wrappers.yolov11 import YOLODetector
from models_config import models_config
from builders.general_builder import GeneralBuilder
from schemas import CreateRequest, DeleteRequest, DictResponse, Response
from server_config import Settings
import asyncio
import os
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
import gradio as gr
import PIL.Image as Image
from server_config import Settings

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    settings = Settings()
    global active_processes
    active_processes = asyncio.Semaphore(settings.num_cores - 1)  
    yield

app = FastAPI(lifespan=lifespan)

# Выбор архитектуры модели и создание модели
created_models = {"554d3bd3-b0ef-4025-b95a-3f0e1a127d48": SSDLiteDetector('ssdlite'),
                  "554d3bd3-b0ef-4025-b95a-3f0e1a127d49": YOLODetector('yolo'),
                  "554d3bd3-b0ef-4025-b95a-3f0e1a127d43": YOLOTransformerDetector('transformer')}

@app.post("/create_model", response_model=Response)
async def create_model(create_request: CreateRequest):
    if len(created_models) >= settings.max_inference_models:
        raise HTTPException(status_code=400, detail="Достигнуто максимальное количество загруженных моделей")
    
    model_type = create_request.model_type
    model_name = create_request.model_name
    
    try:
        model_builder = GeneralBuilder(model_type, model_name)
        detector = model_builder.build()
        created_models[detector.uuid] = detector
    except KeyError:
         raise HTTPException(status_code=400, detail=f"Нельзя создать `{model_type}`, выберите что-то из {list(models_config.keys())}")
    
    return Response(message=f"Модель {model_name} загружена.", uuid=f"{detector.uuid}")

# Выгрузка модели из загруженных в текущий момент
@app.post("/delete_model", response_model=Response)
async def delete_model(delete_request: DeleteRequest):
    model_uuid = delete_request.model_uuid
    print(created_models.keys())
    if model_uuid not in created_models.keys():
        raise HTTPException(status_code=404, detail=f"Модель {model_uuid} не загружена")
    
    del created_models[model_uuid]
    return Response(message=f"Модель выгружена.", uuid=f"{model_uuid}")

# Получение всех текущих на данный момент созданных загруженных моделей
@app.get("/get_models", response_model=DictResponse)
async def get_models():
    models_inform = {}
    for uuid, wrapper in created_models.items():
        models_inform[uuid] = (wrapper.model_type, wrapper.model_name)
    return DictResponse(dict=models_inform)

# Загрузка датасета для обучения
@app.post("/upload_train_dataset")
async def upload_train_dataset(input_file: UploadFile):
    async with httpx.AsyncClient() as client:
        files = {'file': (input_file.filename, input_file.file.read())} 
        response = await client.post(
            f"{settings.url}/runner/upload_input_files/",
            files=files
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"Ошибка:": response.status_code}
        
# Распаковка датасета для обучения
@app.post("/unpack_train_dataset")
async def unpack_train_dataset(input_uuid: UUID = Query(...)):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.url}/runner/unpack_input_files/",
            params={"input_uuid": str(input_uuid)},
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"Ошибка:": response.status_code} 
        
# Сохранение модели в папку /models
@app.post("/save_model", response_model=Response)
async def save_model(input_uuid: UUID = Query(...)):
    created_models[str(input_uuid)].save(settings.model_dir) 
    return Response(message=f"Модель {created_models[str(input_uuid)].model_name} скачана.", uuid=f"{input_uuid}")

# Загрузка модели c сервера по имени
@app.post('/load_model', response_model=Response)
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
    
    return Response(message=f"Модель {model_name} загружена.", uuid=f"{detector.uuid}")


# Загрузка картинки для предсказания
@app.post("/upload_image")
async def upload_image(input_file: UploadFile):
    async with httpx.AsyncClient() as client:
        files = {'file': (input_file.filename, input_file.file.read())} 
        response = await client.post(
            f"{settings.url}/runner/upload_input_files/",
            files=files
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"Ошибка:": response.status_code}
        

# Распаковка картинки для предсказания
@app.post("/unpack_image")
async def unpack_image(input_uuid: UUID = Query(...)):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.url}/runner/unpack_input_files/",
            params={"input_uuid": str(input_uuid)},
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"Ошибка:": response.status_code} 
        
# Запуск модели на картинке
@app.post("/detect")
async def detect(img: UploadFile = File(...), input_uuid: UUID = Query(...), conf_threshold=0.7, iou_threshold=0.5):
    try:
        detector = created_models[str(input_uuid)]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Eror: {e}")
    
    try:
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        print(f"Не удалось сконвертировать картинку в PIL: {e}")
    
    results = detector.predict(
        image_path=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640)

    return results

        
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
    fn=detect,
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

app = gr.mount_gradio_app(app, predict_iface, path="/detect")
