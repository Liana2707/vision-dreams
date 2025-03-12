from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
import numpy as np
import PIL.Image as Image
import io
import json
import cv2
from uuid import UUID

from schemas.output_detect_schema import RootModel
from api.base_models import created_models


predict_router = APIRouter(tags=["predict"])

@predict_router.post("/json", response_model=RootModel, summary="Predict on an image and return json")
async def detect(img: UploadFile = File(...), input_uuid: UUID = Query(...), conf_threshold: float = 0.7, iou_threshold: float = 0.5):
    try:
        detector = created_models[str(input_uuid)]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Eror: {e}")
    
    contents = await img.read()
    img = Image.open(io.BytesIO(contents))
    
    results = detector.predict(
        image=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True)
    
    return results

@predict_router.post("/image", summary="Predict on an image and display results")
async def plot_results(img: UploadFile = File(...), input_uuid: UUID = Query(...), conf_threshold: float = 0.7, iou_threshold: float = 0.5):
    try:
        detector = created_models[str(input_uuid)]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Eror: {e}")
    
    contents = await img.read()
    img = Image.open(io.BytesIO(contents))
    
    result_img = detector.plot_image(img, conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True)
    
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    return Response(content=buffered.getvalue(), media_type="image/png")

@predict_router.post("/draw", summary="Draw detections from json on image")
async def visualise(detections: str = Form(...), image_request: UploadFile = File(...), boxes_format:str = 'tlbr'):
    try:
        image = Image.open(io.BytesIO(await image_request.read()))  
        image_np = np.array(image)
        detections = json.loads(detections)
        detections = RootModel(**detections)
        image_height, image_width, _ = image_np.shape

        for detection_list in detections.model_output:
            for detection in detection_list:
                if boxes_format == 'tlbr':
                    xmin = int(detection.tlbr.xmin)
                    ymin = int(detection.tlbr.ymin)
                    xmax = int(detection.tlbr.xmax)
                    ymax = int(detection.tlbr.ymax)
                elif boxes_format == 'xywhn':
                    xn = detection.xywhn.xn
                    yn = detection.xywhn.yn
                    wn = detection.xywhn.wn
                    hn = detection.xywhn.hn

                    xmin = int((xn - wn/2) * image_width)
                    ymin = int((yn - hn/2) * image_height)
                    xmax = int((xn + wn/2) * image_width)
                    ymax = int((yn + hn/2) * image_height)
                

                cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    image_np,
                    f"{detection.class_name} ({detection.score:.2f})",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".jpg", image_np)
        image_bytes = buffer.tobytes()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")