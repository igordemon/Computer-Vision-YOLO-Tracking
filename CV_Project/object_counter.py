from ultralytics import YOLO
import cv2


model = YOLO("yolo11n.pt")


cap = cv2.VideoCapture(0)


# координата линии

LINE_Y = 300


# история объектов

tracks = {}


# счетчик

counter = 0


while True:


    ret, frame = cap.read()


    if not ret:
        break



    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )


    result = results[0]


    frame = result.plot()



    # рисуем линию


    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0,0,255),
        3
    )



    if result.boxes.id is not None:


        for box, track_id in zip(
            result.boxes,
            result.boxes.id
        ):


            track_id = int(track_id)



            x1,y1,x2,y2 = map(
                int,
                box.xyxy[0]
            )


            # центр объекта

            cx = int(
                (x1+x2)/2
            )

            cy = int(
                (y1+y2)/2
            )



            # первый раз видим объект

            if track_id not in tracks:


                tracks[track_id] = {

                    "previous_y": cy,

                    "counted": False

                }



            else:


                previous_y = tracks[track_id]["previous_y"]



                # проверка пересечения


                if (
                    previous_y < LINE_Y
                    and
                    cy >= LINE_Y
                    and
                    not tracks[track_id]["counted"]
                ):


                    counter += 1


                    tracks[track_id]["counted"] = True



                tracks[track_id]["previous_y"] = cy




            cv2.putText(

                frame,

                f"ID:{track_id}",

                (cx,cy),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255,255,255),

                2

            )



    cv2.putText(

        frame,

        f"Count: {counter}",

        (30,50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.5,

        (0,255,0),

        3

    )



    cv2.imshow(
        "Object Counter",
        frame
    )



    if cv2.waitKey(1)==27:
        break



cap.release()

cv2.destroyAllWindows()