from ultralytics import YOLO

model = YOLO(r"yolo11s.pt")
print(model.task)#输出任务类型
print(model.names)#输出模型支持的类型
print(sum(p.numel() for p in model.parameters()))#打印出所有参数的数量

"""
目标检测detect
旋转目标检测obb
姿态估计pose
实例分割segment
图像分类classify
"""