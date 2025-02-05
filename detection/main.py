import json

from builders.general_builder import GeneralBuilder
from models_config import models_config
import send_task
from tools import read_config, utils

import yaml

# ----- read parameters -----
args = read_config.read_input()

# ----- start logger -----
mainLogger = utils.createLogger('API',f'{args.output}/det_logs.log')
mainLogger.info(f'Using parameters: {args}')

model_builder = GeneralBuilder(args.method)

mainLogger.info("Detector is started.")

try:
    detector = model_builder.build()
    results = detector.predict(args.input_video, save=False)
    mainLogger.info("results =", results)
    '''
    with open('models/coco.yaml') as fh:
        class_names = yaml.load(fh, Loader=yaml.FullLoader)

    
    frame_id = 0
    for r in results:
        frame_id += 1
        cls = r.boxes.cls
        xywhn = r.boxes.xywhn
        conf = r.boxes.conf
        curr_det = []
        for i in range(len(cls)):
            curr_det.append({
            "class":  class_names['names'][int(cls[i])],
            "score": float(conf[i]),
            "bbox": xywhn[i].tolist(),
            })
        curr_frame = {"frame_id": frame_id, "detections": curr_det}

        if args.rabbit:
            command = json.dumps(curr_frame)
            send_task.interactive_shell(args.output_queue, args.output_host, command, mainLogger)
        '''
except KeyError as e:
    print(f"Такой модели нет, выберите что-то из {models_config.keys()}. Ошибка: {e}")

mainLogger.info('Detection finished successfully')












