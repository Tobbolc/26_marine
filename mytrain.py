from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"coco8.yaml",
        epochs=40,  # 训练次数
        imgsz=640,  # 图片尺寸
        batch=2,  # 批数
        cache=False,
        workers=0,
        val=False,  # 单纯训练不验证
    )
