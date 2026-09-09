from ultralytics import YOLO
import cv2


# Загружаем модель
model = YOLO("yolo11n.pt")


# Открываем камеру
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Ошибка: камера не найдена")
    exit()


while True:

    # Получаем кадр
    ret, frame = cap.read()

    if not ret:
        break


    # Передаем кадр в YOLO
    results = model(frame)


    # Рисуем найденные объекты
    annotated_frame = results[0].plot()


    # Показываем результат
    cv2.imshow(
        "YOLO Real-Time Detection",
        annotated_frame
    )


    # ESC для выхода
    if cv2.waitKey(1) == 27:
        break


# Освобождаем камеру
cap.release()

cv2.destroyAllWindows()