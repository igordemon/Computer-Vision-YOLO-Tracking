from ultralytics import YOLO

import torch
import cv2
import numpy as np
import math
import time

from collections import deque


# ============================================================
# НАСТРОЙКИ
# ============================================================

YOLO_MODEL = "yolo11n.pt"

TRACKER_CONFIG = "bytetrack.yaml"

CAMERA_ID = 0


# ------------------------------------------------------------
# Пороги MiDaS
#
# ВАЖНО:
# это НЕ метры.
#
# В нашем случае:
# больше depth -> объект ближе
#
# Эти значения потом можно откалибровать под конкретную камеру.
# ------------------------------------------------------------

DEPTH_FAR = 200
DEPTH_MEDIUM = 350
DEPTH_NEAR = 500


# ------------------------------------------------------------
# Минимальное изменение depth,
# которое считаем реальным приближением/удалением.
#
# Нужно для защиты от шума MiDaS.
# ------------------------------------------------------------

DEPTH_MOVEMENT_THRESHOLD = 8


# ------------------------------------------------------------
# Минимальное движение центра объекта,
# чтобы определять направление.
# ------------------------------------------------------------

DIRECTION_THRESHOLD = 3


# ------------------------------------------------------------
# Классы, которые потенциально важны
# для транспортной системы.
# ------------------------------------------------------------

DANGEROUS_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


# ============================================================
# ФУНКЦИИ
# ============================================================


def get_depth_state(depth):
    """
    Классификация относительной глубины.

    Чем больше MiDaS depth,
    тем объект ближе.
    """

    if depth >= DEPTH_NEAR:
        return "VERY CLOSE"

    elif depth >= DEPTH_MEDIUM:
        return "NEAR"

    elif depth >= DEPTH_FAR:
        return "MEDIUM"

    else:
        return "FAR"



def calculate_pixel_speed(points):
    """
    Условная скорость объекта.

    Единицы:
        pixels / frame

    Это НЕ м/с и НЕ км/ч.
    """

    if len(points) < 2:
        return 0.0

    x1, y1 = points[-2]
    x2, y2 = points[-1]

    dx = x2 - x1
    dy = y2 - y1

    return math.sqrt(dx ** 2 + dy ** 2)



def get_direction(points):
    """
    Определяем направление движения
    в плоскости изображения.
    """

    if len(points) < 2:
        return "STILL"

    x1, y1 = points[-2]
    x2, y2 = points[-1]

    dx = x2 - x1
    dy = y2 - y1


    # мелкие изменения считаем шумом

    if (
        abs(dx) < DIRECTION_THRESHOLD
        and
        abs(dy) < DIRECTION_THRESHOLD
    ):
        return "STILL"


    if abs(dx) > abs(dy):

        if dx > 0:
            return "RIGHT"

        else:
            return "LEFT"

    else:

        if dy > 0:
            return "DOWN"

        else:
            return "UP"



def get_depth_motion(depth_history):
    """
    Определяет:
        APPROACHING
        MOVING AWAY
        STABLE

    MiDaS у нас работает примерно как inverse depth:

        depth увеличивается -> объект приближается
        depth уменьшается -> объект удаляется

    Используем несколько кадров,
    чтобы уменьшить влияние шума.
    """

    if len(depth_history) < 6:
        return "UNKNOWN"


    values = list(depth_history)


    # среднее старых трех кадров

    previous = np.mean(
        values[-6:-3]
    )


    # среднее последних трех кадров

    current = np.mean(
        values[-3:]
    )


    delta = current - previous


    if delta > DEPTH_MOVEMENT_THRESHOLD:

        return "APPROACHING"


    elif delta < -DEPTH_MOVEMENT_THRESHOLD:

        return "MOVING AWAY"


    else:

        return "STABLE"



def calculate_risk(
    object_class,
    depth_state,
    depth_motion
):
    """
    Простая экспертная система Risk Assessment.

    Это НЕ отдельная нейросеть.

    Мы объединяем информацию:

        класс объекта
        +
        относительную глубину
        +
        приближение/удаление
    """


    if object_class not in DANGEROUS_CLASSES:

        return "LOW"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if (
        depth_state == "VERY CLOSE"
        and
        depth_motion == "APPROACHING"
    ):
        return "HIGH"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if (
        depth_state == "NEAR"
        and
        depth_motion == "APPROACHING"
    ):
        return "MEDIUM"


    if depth_state == "VERY CLOSE":

        return "MEDIUM"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"



def get_risk_color(risk):
    """
    OpenCV использует BGR.
    """

    if risk == "HIGH":

        return (0, 0, 255)      # красный

    elif risk == "MEDIUM":

        return (0, 255, 255)    # желтый

    else:

        return (0, 255, 0)      # зеленый



def draw_trajectory(frame, points, color):
    """
    Рисуем последние координаты объекта.
    """

    if len(points) < 2:
        return


    for i in range(1, len(points)):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            2
        )



def draw_object_depth_bar(
    frame,
    x2,
    y1,
    y2,
    depth,
    color
):
    """
    Небольшая depth-шкала возле объекта.
    """

    bar_width = 10


    # Высота bounding box

    box_height = max(
        20,
        y2 - y1
    )


    # Нормируем для визуализации.

    normalized = np.clip(
        depth / 650.0,
        0.0,
        1.0
    )


    filled_height = int(
        box_height * normalized
    )


    bar_x1 = x2 + 5

    bar_x2 = bar_x1 + bar_width


    # фон

    cv2.rectangle(
        frame,
        (bar_x1, y1),
        (bar_x2, y2),
        (60, 60, 60),
        -1
    )


    # заполнение

    cv2.rectangle(
        frame,
        (bar_x1, y2 - filled_height),
        (bar_x2, y2),
        color,
        -1
    )



def get_object_depth(
    depth_map,
    x1,
    y1,
    x2,
    y2
):
    """
    Получаем относительную глубину объекта.

    Не берем весь bounding box,
    потому что по краям часто находится фон.

    Используем центральные 50% области.
    Затем берем median вместо mean —
    она устойчивее к выбросам.
    """


    object_depth = depth_map[
        y1:y2,
        x1:x2
    ]


    if object_depth.size == 0:

        return None


    h, w = object_depth.shape


    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)

    cy1 = int(h * 0.25)
    cy2 = int(h * 0.75)


    center_depth = object_depth[
        cy1:cy2,
        cx1:cx2
    ]


    if center_depth.size == 0:

        return None


    # median обычно устойчивее,
    # чем обычное среднее

    depth_value = np.median(
        center_depth
    )


    return float(depth_value)


# ============================================================
# ЗАГРУЖАЕМ YOLO
# ============================================================

print("Loading YOLO...")

yolo = YOLO(
    YOLO_MODEL
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# ЗАГРУЖАЕМ MIDAS
# ============================================================

print("Loading MiDaS...")


midas = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small"
)


midas.to(device)

midas.eval()


midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)


transform = midas_transforms.small_transform


print("Models loaded.")


# ============================================================
# КАМЕРА
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)


if not cap.isOpened():

    raise RuntimeError(
        "Не удалось открыть камеру."
    )


# ============================================================
# ИСТОРИЯ ОБЪЕКТОВ
# ============================================================


# Координаты траекторий

tracks = {}


# История глубины

depth_histories = {}


# Для FPS

previous_time = time.time()


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

while True:


    ret, frame = cap.read()


    if not ret:

        print(
            "Не удалось получить кадр."
        )

        break


    frame_height, frame_width = frame.shape[:2]


    # ========================================================
    # 1. YOLO + BYTETRACK
    # ========================================================

    results = yolo.track(

        frame,

        persist=True,

        tracker=TRACKER_CONFIG,

        verbose=False

    )


    result = results[0]


    # Создаем копию оригинального изображения.

    # Bounding boxes будем рисовать сами,
    # чтобы полностью контролировать интерфейс.

    output = frame.copy()


    # ========================================================
    # 2. MIDAS
    # ========================================================


    image_rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    input_batch = transform(
        image_rgb
    ).to(device)


    with torch.no_grad():


        prediction = midas(
            input_batch
        )


        prediction = torch.nn.functional.interpolate(

            prediction.unsqueeze(1),

            size=image_rgb.shape[:2],

            mode="bicubic",

            align_corners=False

        ).squeeze()


    depth_map = (
        prediction
        .cpu()
        .numpy()
    )


    # ========================================================
    # 3. ОБРАБОТКА НАЙДЕННЫХ ОБЪЕКТОВ
    # ========================================================


    if (
        result.boxes is not None
        and
        result.boxes.id is not None
    ):


        for box, tracker_id in zip(

            result.boxes,

            result.boxes.id

        ):


            track_id = int(
                tracker_id.item()
            )


            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0].item()
            )


            object_class = yolo.names[
                class_id
            ]


            confidence = float(
                box.conf[0].item()
            )


            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            coords = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            x1, y1, x2, y2 = coords


            # Защита от выхода за изображение

            x1 = max(
                0,
                min(x1, frame_width - 1)
            )


            x2 = max(
                0,
                min(x2, frame_width)
            )


            y1 = max(
                0,
                min(y1, frame_height - 1)
            )


            y2 = max(
                0,
                min(y2, frame_height)
            )


            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------


            cx = int(
                (x1 + x2) / 2
            )


            cy = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # TRACK HISTORY
            # ------------------------------------------------


            if track_id not in tracks:

                tracks[track_id] = deque(
                    maxlen=30
                )


            tracks[track_id].append(
                (cx, cy)
            )


            # ------------------------------------------------
            # PIXEL MOTION
            # ------------------------------------------------


            pixel_speed = calculate_pixel_speed(

                tracks[track_id]

            )


            direction = get_direction(

                tracks[track_id]

            )


            # ------------------------------------------------
            # DEPTH OBJECT
            # ------------------------------------------------


            raw_depth = get_object_depth(

                depth_map,

                x1,
                y1,
                x2,
                y2

            )


            if raw_depth is None:

                continue


            # ------------------------------------------------
            # DEPTH HISTORY
            # ------------------------------------------------


            if track_id not in depth_histories:

                depth_histories[track_id] = deque(
                    maxlen=12
                )


            depth_histories[
                track_id
            ].append(
                raw_depth
            )


            # ------------------------------------------------
            # DEPTH SMOOTHING
            # ------------------------------------------------

            # Берем median последних значений.
            # Так цифра будет меньше прыгать.

            recent_depth = list(
                depth_histories[track_id]
            )[-5:]


            depth_value = float(
                np.median(
                    recent_depth
                )
            )


            # ------------------------------------------------
            # DEPTH STATE
            # ------------------------------------------------


            depth_state = get_depth_state(

                depth_value

            )


            # ------------------------------------------------
            # APPROACHING / MOVING AWAY
            # ------------------------------------------------


            depth_motion = get_depth_motion(

                depth_histories[
                    track_id
                ]

            )


            # ------------------------------------------------
            # RISK
            # ------------------------------------------------


            risk = calculate_risk(

                object_class,

                depth_state,

                depth_motion

            )


            color = get_risk_color(
                risk
            )


            # =================================================
            # ВИЗУАЛИЗАЦИЯ
            # =================================================


            # Bounding Box

            cv2.rectangle(

                output,

                (x1, y1),

                (x2, y2),

                color,

                2

            )


            # Центр объекта

            cv2.circle(

                output,

                (cx, cy),

                5,

                color,

                -1

            )


            # Траектория

            draw_trajectory(

                output,

                tracks[track_id],

                color

            )


            # Depth Bar

            draw_object_depth_bar(

                output,

                x2,
                y1,
                y2,

                depth_value,

                color

            )


            # -------------------------------------------------
            # ИНФОРМАЦИЯ ОБ ОБЪЕКТЕ
            # -------------------------------------------------


            label_1 = (

                f"{object_class.upper()} "

                f"ID:{track_id} "

                f"{confidence:.2f}"

            )


            label_2 = (

                f"Depth:{depth_value:.1f} "

                f"{depth_state}"

            )


            label_3 = (

                f"{depth_motion} "

                f"{direction} "

                f"{pixel_speed:.1f}px/f"

            )


            label_4 = (

                f"RISK: {risk}"

            )


            text_y = max(
                25,
                y1 - 65
            )


            cv2.putText(

                output,

                label_1,

                (x1, text_y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2

            )


            cv2.putText(

                output,

                label_2,

                (x1, text_y + 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2

            )


            cv2.putText(

                output,

                label_3,

                (x1, text_y + 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                color,

                2

            )


            cv2.putText(

                output,

                label_4,

                (x1, text_y + 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                color,

                2

            )


    # ========================================================
    # FPS
    # ========================================================


    current_time = time.time()


    delta_time = (
        current_time
        -
        previous_time
    )


    if delta_time > 0:

        fps = 1.0 / delta_time

    else:

        fps = 0


    previous_time = current_time


    cv2.putText(

        output,

        f"FPS: {fps:.1f}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )


    # ========================================================
    # ПОДПИСЬ
    # ========================================================


    cv2.putText(

        output,

        "Relative depth - NOT meters",

        (20, frame_height - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2

    )


    # ========================================================
    # OUTPUT
    # ========================================================


    cv2.imshow(

        "Autonomous Perception Prototype",

        output

    )


    # ESC

    if cv2.waitKey(1) & 0xFF == 27:

        break


# ============================================================
# ЗАВЕРШЕНИЕ
# ============================================================

cap.release()

cv2.destroyAllWindows()