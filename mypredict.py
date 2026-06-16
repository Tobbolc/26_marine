from ultralytics import YOLO

model = YOLO(r"yolo11s.pt")  # 使用的模型
model.predict(
    source=r"ultralytics\assets",  # 被预测数据来源
    save=True,  # 结果的保存
    show=False,  # 直接查看结果
    # line_width = 8,
    visualize=True,  # 保存中间特征图
)
