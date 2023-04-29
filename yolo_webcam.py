from ultralytics import YOLO
import cv2
import cvzone
import math

videos = "http://192.168.1.33:8080/video"
cap = cv2.VideoCapture(0)
cap.open(videos)
cap.set(3,640)
cap.set(4,480)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("../yolo-weights/yolov8n.pt")
while True:
    success , img = cap.read()
    resized = cv2.resize(img, (500,500))
    predict = model(resized,stream=True)
    for r in predict:
        boxes = r.boxes
        for box in boxes:
           
           #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # print( "============",x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            w,h = x2-x1, y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(resized,(x1,y1,w,h))
            #confidance
            conf = math.ceil(box.conf[0]*100)/100
            #class
            cls = int(box.cls[0])

            
            cvzone.putTextRect(resized,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale = 2.5)
    cv2.imshow("Image",resized)
    cv2.waitKey(1)
