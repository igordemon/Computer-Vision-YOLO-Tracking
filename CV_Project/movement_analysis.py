from ultralytics import YOLO
import cv2
import math



def get_direction(points):

    if len(points) < 2:
        return ""


    x1,y1 = points[-2]
    x2,y2 = points[-1]


    dx = x2-x1
    dy = y2-y1


    if abs(dx) > abs(dy):

        return "RIGHT" if dx > 0 else "LEFT"

    else:

        return "DOWN" if dy > 0 else "UP"



def calculate_speed(points):

    if len(points)<2:
        return 0


    x1,y1 = points[-2]
    x2,y2 = points[-1]


    dx=x2-x1
    dy=y2-y1


    return math.sqrt(
        dx**2+dy**2
    )



model = YOLO("yolo11n.pt")


cap=cv2.VideoCapture(0)


tracks={}



while True:


    ret,frame=cap.read()


    if not ret:
        break



    results=model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )


    result=results[0]


    frame=result.plot()



    if result.boxes.id is not None:


        for box,track_id in zip(
            result.boxes,
            result.boxes.id
        ):


            track_id=int(track_id)


            x1,y1,x2,y2=map(
                int,
                box.xyxy[0]
            )


            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)



            if track_id not in tracks:

                tracks[track_id]=[]


            tracks[track_id].append(
                (cx,cy)
            )


            if len(tracks[track_id])>50:

                tracks[track_id].pop(0)



            # скорость

            speed=calculate_speed(
                tracks[track_id]
            )


            direction=get_direction(
                tracks[track_id]
            )



            cv2.putText(
                frame,
                f"ID:{track_id} {direction} {speed:.1f}",
                (cx,cy-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )



            # траектория


            points=tracks[track_id]


            for i in range(1,len(points)):

                cv2.line(
                    frame,
                    points[i-1],
                    points[i],
                    (0,255,0),
                    2
                )



    cv2.imshow(
        "Movement Analysis",
        frame
    )


    if cv2.waitKey(1)==27:
        break



cap.release()

cv2.destroyAllWindows()