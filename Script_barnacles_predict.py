import cv2

from ultralytics import YOLO

model = YOLO("runs/barnacles/para1_mixenh_phase2_full/weights/best.pt")
results = model.predict(
    source=r"video_predict/barnacles2.mp4",  # 视频路径
    save=False,  # 结果的保存
    show=False,  # 直接查看结果
    conf=0.45,  # 置信度阈值
    iou=0.45,  # NMS 的 IoU 阈值
    stream=True,
)
for result in results:
    plotted = result.plot()
    cv2.imshow("yolo inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
