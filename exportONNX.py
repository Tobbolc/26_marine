from ultralytics import YOLO


def main():
    BEST_PT = "runs/barnacles/y11n_phase2_full/weights/best.pt"
    model = YOLO(BEST_PT)

    # 导出 ONNX（用于通用验证/以及给 TensorRT 构建 engine）
    model.export(format="onnx", opset=12, simplify=True, imgsz=640)


if __name__ == "__main__":
    main()

# import onnx
# import sys
# print("onnx version:", onnx.__version__)
# print("python:", sys.executable)
