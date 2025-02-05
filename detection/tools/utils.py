import logging
import cv2


def createLogger(loggerName, fileName):
    logger = logging.getLogger(loggerName)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(fileName)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
    return logger

def get_frames(mov):
    video = cv2.VideoCapture(mov)
    ok, frame = video.read()
    count = 0
    while ok:
        cv2.imwrite(f'frames/frame{count}.jpg', frame)
        count += 1
        ok, frame = video.read()
    video.release()

def get_image_size(self):
    img = cv2.imread(self.name)
    return img.shape




