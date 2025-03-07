import json

from builders.general_builder import GeneralBuilder
from models_config import models_config
import send_task
from tools import read_config, utils
from server import created_models
import yaml
from server import created_models

# ----- read parameters -----
args = read_config.read_input()

# ----- start logger -----
mainLogger = utils.createLogger('API',f'{args.output}/det_logs.log')
mainLogger.info(f'Using parameters: {args}')

try:
    detector = created_models[args.method]
    mainLogger.info("Detector is started.")
except KeyError as e:
        print(f"Такой модели нет, выберите что-то из {models_config.keys()}. Ошибка: {e}")

if args.model_mode == 'predict':
    results = detector.predict(args.input, save=True)
    mainLogger.info("results =", results)
    
    with open('models/coco.yaml') as fh:
        class_names = yaml.load(fh, Loader=yaml.FullLoader)

    cls = results["class"]
    xywhn = results["bbox"]
    conf = results["score"]
    curr_det = []
    for i in range(len(cls)):
        curr_det.append({
        "class":  class_names['names'][int(cls[i])],
        "score": float(conf[i]),
        "bbox": xywhn[i].tolist(),
        })
    curr_frame = {"frame_id": 0, "detections": curr_det}
    
    print(curr_frame)

    if args.rabbit:
        command = json.dumps(curr_frame)
        send_task.interactive_shell(args.output_queue, args.output_host, command, mainLogger)
    
elif args.model_mode == 'train':
    results = detector.train(args.input)
    mainLogger.info("results =", results)

mainLogger.info('Detection finished successfully')












