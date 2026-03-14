from ultralytics import YOLO

# train 204 / val 26 / test 25
# 目标：yolo11n + 强化水下增强 + 两阶段（先冻后放）避免小数据过拟合

def main():
    DATA_YAML = "barnacles.yaml"
    IMG_SIZE = 640  # 与 Jetson 端部署对齐，先用 640 稳定跑到 15-30fps
    DEVICE = 0      # RTX 4060
    # Windows 若 DataLoader 报错，把 workers 改成 0
    WORKERS = 0
    BATCH = 16      # 4060 16GB 通常足够跑 yolo11n@640；如OOM改 8/4 或 batch=-1 自动

    # -------------------------
    # Phase 1：冻住 backbone，先让 head 适应藤壶/海生物
    # -------------------------
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=60,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=True,
        pretrained=True,

        # 迁移学习：先冻一部分层（数值可在 5~15 间试）
        freeze=10,

        # 训练策略
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.01,
        patience=50,
        cos_lr=True,
        warmup_epochs=3,
        seed=42,
        amp=True,

        # -------------------------
        # 水下/低能见度针对性增强（颜色衰减/光照不均/畸变）
        # -------------------------
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.6,
        degrees=5.0,
        translate=0.10,
        scale=0.60,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.1,
        mosaic=1.0,
        mixup=0.10,
        close_mosaic=15,

        project="runs/barnacles",
        name="y11n_phase1_frozen",
        plots=True,
    )

    # -------------------------
    # Phase 2：全量解冻微调，让 backbone 适应水下域差异
    # -------------------------
    model2 = YOLO("runs/barnacles/y11n_phase1_frozen/weights/best.pt")
    model2.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=200,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=True,
        pretrained=True,

        freeze=0,

        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        weight_decay=0.01,
        patience=60,
        cos_lr=True,
        warmup_epochs=2,
        seed=42,
        amp=True,

        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.6,
        degrees=5.0,
        translate=0.10,
        scale=0.60,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.1,
        mosaic=0.8,
        mixup=0.05,
        close_mosaic=20,

        project="runs/barnacles",
        name="y11n_phase2_full",
        plots=True,
    )

    # （可选）用 test 集跑一次评估
    model3 = YOLO("runs/barnacles/y11n_phase2_full/weights/best.pt")
    model3.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, device=DEVICE)


if __name__ == "__main__":
    main()