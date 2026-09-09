from ultralytics import YOLO


model = YOLO("yolo11n.pt")


results = model(
    "test.jpg"
)


result = results[0]


for box in result.boxes:


    # класс

    cls_id = int(box.cls[0])


    name = model.names[cls_id]


    # уверенность

    confidence = float(box.conf[0])


    # координаты

    coordinates = box.xyxy[0].tolist()


    print(
        f"""
        Object: {name}
        Confidence: {confidence:.2f}
        Coordinates: {coordinates}
        """
    )


result.show()