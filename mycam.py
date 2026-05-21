from ultralytics import YOLO
import cv2

model = YOLO(r"yolo11n-seg.pt")
results = model(
    source=0,
    stream=True,
)
#下面的代码用来解决内存占用问题
for result in results:
    plotted = result.plot()
    cv2.imshow("yolo inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break