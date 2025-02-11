# vision-dreams

Vision Dreams - это модуль детекции объектов, разработанный на базе FastAPI.

## Установка

Чтобы запустить этот проект, вам потребуется установить необходимые зависимости.  Используйте следующие шаги:


1. **Клонируйте репозиторий:**

`git clone https://github.com/Liana2707/vision-dreams.git`
    
`cd vision-dreams`

`cd detection`

2. **Установка зависимостей:**

`pip install -r requirements.txt`

`pip install -q git+https://github.com/huggingface/transformers.git`

`pip install "fastapi[standart]"`

3. **Разработка:**

`fastapi dev server.py`

4. **Запуск демо:**

`python main.py -f test_config.json`

## Модели

Возможные встроенные архитектуры: 

 1. YOLO v11

 2. YOLOS

 3. SSDMobileNet

 ## Конфигурация сервера

 Конфигурационные переменные хранятся в файле `.env`

 `MODEL_DIR` - директория для сохранения моделей на сервере

 `NUM_CORES` - количество ядер для обучения (пока не используется)

 `MAX_INFERENCE_MODELS` - максимальное число моделей для инференса

 `URL="http://dev8123.ai-center.online"`

 ## Endpoints
 
 **/create_model** - Создание модели

 **/delete_model** - Выгрузка модели из загруженных в текущий момент

 **/get_models** - Получение всех текущих на данный момент созданных загруженных моделей

 **/upload_train_dataset** - Загрузка датасета для обучения
       
 **/unpack_train_dataset** - Распаковка датасета для обучения
  
 **/save_model** - Сохранение модели в папку /models

 **/load_model** - Загрузка модели c сервера по имени(пока только для yolo)

 **/upload_image** - Загрузка картинки для предсказания

 **/unpack_image** - Распаковка картинки для предсказания

 **/predict** - Запуск модели на картинке
 
 









 
