from ultralytics import YOLO

# train 204 / val 26 / test 25
# 目标：yolo11n + 强化水下增强 + 两阶段（先冻后放）避免小数据过拟合

def main():
    DATA_YAML = "barnacles_mix_enh.yaml"
    IMG_SIZE = 640  # 与 Jetson 端部署对齐，先用 640 稳定跑到 15-30fps
    DEVICE = 0      # RTX 4060

    # 用 test/val 集跑一次评估
    model3 = YOLO("runs/barnacles/para1_mixenh_phase2_full/weights/best.pt")
    model3.val(data=DATA_YAML, split="val", imgsz=IMG_SIZE, device=DEVICE,project="runs/barnacles",name="val_enh_para1")


if __name__ == "__main__":
    main()