import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone


model = YOLO('Models/Counting_Car.pt')
logo = cv2.imread('Logo/Techwalnut_logo.jpg', -1)
classnames = []
with open('Classes/classes.txt', 'r') as f:
    classnames = f.read().splitlines()
tracker = Sort(max_age=20)


# Resolution
def set_resolution(resolution):
    switcher = {
        "360":  (640,360),
        "480":  (940, 480),
        "540":  (960,540),
        "720":  (1280, 720),
        "1080": (1920, 1080),
        "2k":   (2048, 1080),
        "4k":   (3840, 2160),
    }
    return switcher.get(resolution.lower(), (1920, 1080))  # Default to 1080p

def calculate_total_count(counter_in, counter_out):
    return len(counter_in) - len(counter_out)

def calculate_empty_space(total_count, total_spaces):
    empty_space = total_spaces - total_count
    return empty_space

def main_loop_1080p(cap_1080, model, logo, classnames, tracker, total_spaces):
    line = [20, 700, 1900, 700]
    counter_in = set()
    counter_out = set()
    prev_positions = {}

    while True:
        ret, frame_1080 = cap_1080.read()
        if not ret:
            cap_1080.release()
            cv2.destroyAllWindows()
            break
        detections = np.empty((0, 5))
        result = model(frame_1080, stream=1)
        for info in result:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                classindex = box.cls[0]
                conf = math.ceil(conf * 100)
                classindex = int(classindex)
                objectdetect = classnames[classindex]

                if objectdetect in ['car', 'bus', 'truck'] and conf > 60:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    new_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_detections))

                    cv2.rectangle(frame_1080, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cvzone.putTextRect(frame_1080, f'{objectdetect} {conf}%',
                                       [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        track_result = tracker.update(detections)
        cv2.line(frame_1080, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 7)

        for results in track_result:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cv2.circle(frame_1080, (cx, cy), 6, (0, 0, 255), -1)
            cv2.rectangle(frame_1080, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cvzone.putTextRect(frame_1080, f'{id}',
                               [x1 + 8, y1 - 12], thickness=1, scale=1.5)

            if id in prev_positions:
                prev_cx, prev_cy = prev_positions[id]
                if cy < prev_cy:
                    if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
                        cv2.line(frame_1080, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 15)
                        counter_in.discard(id)
                        counter_out.add(id)
                elif cy > prev_cy:
                    if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
                        cv2.line(frame_1080, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 15)
                        counter_out.discard(id)
                        counter_in.add(id)
            prev_positions[id] = (cx, cy)
        
        total_count = calculate_total_count(counter_in, counter_out)
        empty_space = calculate_empty_space(total_count, total_spaces)

        logo_width = 130
        logo_height = 40
        logo_resized = cv2.resize(logo, (logo_width, logo_height))

        logo_x = 0
        logo_y = 0

        frame_1080[logo_y:logo_y + logo_height, logo_x:logo_x + logo_width] = logo_resized

        #cv2.rectangle(frame, (0, 1050), (400, 1080), (255, 255, 255), -1)
        cv2.putText(frame_1080, f'Parking Management System By Techwalnut Innovations', (10, 1040), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),thickness=2)

        cv2.rectangle(frame_1080, (1400, 0), (1920, 200), (255, 255, 255), -1)
        cv2.putText(frame, f'Incoming Vehicles: {len(counter_in)}', (1410, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),thickness=2)

        cv2.putText(frame_1080, f'Outgoing Vehicles: {len(counter_out)}', (1410, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),thickness=2)
        
        cv2.putText(frame_1080, f'Occupied Space: {total_count}', (1410, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),thickness=2)

        cv2.putText(frame_1080, f'Empty Space: {empty_space}', (1410, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),thickness=2)

        cv2.imshow('frame_1080', frame_1080)

        # To Exit press 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
def main():
    #camera_url = "rtsp://admin:L2AE329A@192.168.1.116:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    camera_url_1080 = "Example_Videos/Example2_1080p.mp4"
    cap_1080 = cv2.VideoCapture(camera_url_1080)

    resolution = "1080"
    width, height = set_resolution(resolution)
    cap_1080.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_1080.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    main_loop_1080p(cap_1080, model, logo, classnames, tracker)
    total_spaces = 60
    

if __name__ == "__main__":
    main()