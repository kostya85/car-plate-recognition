## Оборудование
Тестирование производилось с использованием следующего оборудования:

i7-7700k CPU and Nvidia 1080TI GPU
OS Ubuntu 18.04
CUDA 10.1
cuDNN v7.6.5
TensorRT-6.0.1.5
Tensorflow-GPU 2.3.1

## Установка
Для начала склонируйте репозиторий и установите предтренировочные веса:
```
pip install -r ./requirements.txt
```

# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights
custom weights: https://drive.google.com/file/d/10Fj94jEgMc8Xc-uimr9F4Egxg-W0d9kp/view?usp=sharing
Кастомные веса надо положить в папку /checkpoints

## Датасет
https://github.com/RobertLucian/license-plate-dataset
Распакован в /custom_dataset

## В папке /custom_dataset находятся изображения для тренировки модели
Выбран именно такой датасет, так как у него были готовы изображения и размеченные xml файлы меток

## Обработка данных под формат YOLOv3
```
!python tools/XML_to_YOLOv3.py
```

`./yolov3/configs.py` конфиг настроен для тренировки модели.

## Тренировка модели
Тренировать можно через терминал
```
python train.py
tensorboard --logdir=log
```
А также через notebook
```
from train import *
tf.keras.backend.clear_session()
main()
```

## Оценка модели
## AP and precision/recall per class
97.698% = license-plate AP  
## mAP of all classes
mAP = 97.698%, 7.09 FPS
# Tensorboard
Tensorboard располагается по адресу http://localhost:6006/
<p align="center">
    <img width="100%" src="images/tensorboard-1.png" style="max-width:100%;"></a>
</p>
Выше приведены графики всех потерь, используемых в процессе обучения, наиболее важным является validate_loss/total_val:
<p align="center">
    <img width="100%" src="images/tensorboard-2.png" style="max-width:100%;"></a>
</p>
Чем меньше значение validate_loss, тем лучше модель. В нашем случае лучшее значение было равно 0.32

## Сравнение с YOLOv7
В качестве эксперимента было принято решение также обучить модуль на основе YOLOv7.
Для этого было создан notebook с тренировкой модели
Название файла - yolov7-train.ipynb
Лучшее значение метрики mAP было равно 0.868 за 55 эпох
<p align="center">
    <img width="100%" src="images/yolov7.PNG" style="max-width:100%;"></a>
</p>

## Использование кастомной модели
Далее натренированная модель используется для определения номерных рамок
```
# Create a new model instance
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights

# Plate detection
image_path   = "./images/3.jpg"
detect_data = detect_image(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
image = detect_data["image"]
bboxes = detect_data["bboxes"]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(30,15))
plt.imshow(image)
```

<p align="center">
    <img width="100%" src="images/plate-detection.PNG" style="max-width:100%;"></a>
</p>

Для распознавания самих символов используется CNN
```
# Create a new model instance
loaded_model = Sequential()
loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
loaded_model.add(Dropout(0.4))
loaded_model.add(Flatten())
loaded_model.add(Dense(128, activation='relu'))
loaded_model.add(Dense(36, activation='softmax'))

# Restore the weights
loaded_model.load_weights('checkpoints/my_checkpoint')
```

<p align="center">
    <img width="100%" src="images/plate-recognition.PNG" style="max-width:100%;"></a>
</p>


## Масштабирование проекта
Для того, чтобы масштабировать проект на несколько камер, был написан скрипт `detect.py`, на вход которому будут отправляться пути до изробржений с машинами, скрипт возвращает распознанный номер. Предполагаемая архитертура проекта выглядит следующий образом:

<p align="center">
    <img width="100%" src="images/structure.PNG" style="max-width:100%;"></a>
</p>