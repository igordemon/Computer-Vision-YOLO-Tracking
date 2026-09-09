from ultralytics import YOLO
import cv2


# Загружаем модель
model = YOLO("yolo11n.pt")


# Открываем камеру
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )


    result = results[0]


    # Получаем информацию об объектах

    if result.boxes.id is not None:

        boxes = result.boxes


        for box, track_id in zip(
            boxes,
            boxes.id
        ):

            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            confidence = float(
                box.conf[0]
            )


            print(
                f"ID: {int(track_id)} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )


    # Рисуем результат

    annotated_frame = result.plot()


    cv2.imshow(
        "YOLO ByteTrack",
        annotated_frame
    )


    # ESC выход

    if cv2.waitKey(1) == 27:
        break


cap.release()

cv2.destroyAllWindows()