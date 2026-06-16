from __future__ import annotations

"""
藤壶检测训练 / 验证 / 测试脚本（无数据集生成版）
==========================================

用途
----
1) 直接使用你已经准备好的“混合训练集”进行训练：
   - train: 原图 + 增强图
   - val/test: 保持原图

2) 保留你增强版脚本中已经验证过的训练参数：
   - Phase 1: 冻结部分 backbone，先让检测头适应任务
   - Phase 2: 全量解冻微调
   - Phase 2 中适当减弱增强强度，避免后期训练不稳定

3) 提供训练、验证、测试三个入口，便于后续复用。

说明
----
- 这个脚本已经把“构建增强数据集”的代码全部删除。
- 你现在只需要把 DATA_YAML 指向你已经生成好的混合数据集 yaml。
- 建议用 val 集选模型、选阈值；test 集尽量只做最终确认，不要反复拿它调参。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# =========================
# 1) 默认参数（可在命令行覆盖）
# =========================
DEFAULT_DATA_YAML = "barnacles_mix_enh/dataset_mix_enh.yaml"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_PROJECT = "runs/barnacles"

IMG_SIZE = 640
DEVICE: int | str = 0  # 有 GPU 时一般为 0；无 GPU 可改成 "cpu"
WORKERS = 0  # Windows 常用 0；Linux 可按机器情况增大
BATCH = 16
SEED = 42


# =========================
# 2) 训练部分
# =========================
def train_two_stage(
    data_yaml: str,
    *,
    base_model: str = DEFAULT_MODEL,
    project: str = DEFAULT_PROJECT,
    phase1_name: str = "y11n_mixenh_phase1_frozen",
    phase2_name: str = "y11n_mixenh_phase2_full",
    imgsz: int = IMG_SIZE,
    batch: int = BATCH,
    device: int | str = DEVICE,
    workers: int = WORKERS,
    seed: int = SEED,
) -> Path:
    """两阶段训练。.

    Phase 1：
        - 冻结 backbone 的前一部分层
        - 让 head 先快速适应“藤壶检测”这个任务
        - 增强强度略高一点，提高小样本条件下的鲁棒性

    Phase 2：
        - 全量解冻
        - 在更温和的增强配置下微调
        - 目的是把特征表达进一步压到你的数据分布上

    返回值：
        Phase 2 的 best.pt 路径
    """
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml_path}")

    # -------------------------
    # Phase 1：冻结训练
    # -------------------------
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml_path),
        imgsz=imgsz,
        epochs=50,
        batch=batch,
        device=device,
        workers=workers,
        cache=True,
        pretrained=True,
        seed=seed,
        deterministic=True,
        amp=True,
        freeze=10,  # 冻结部分主干层
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        patience=25,
        cos_lr=True,
        warmup_epochs=3,
        hsv_h=0.01,
        hsv_s=0.40,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        shear=1.5,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.50,
        mixup=0.05,
        close_mosaic=10,
        project=project,
        name=phase1_name,
        plots=True,
    )

    phase1_best = Path(project) / phase1_name / "weights" / "best.pt"
    if not phase1_best.exists():
        raise FileNotFoundError(
            f"Phase 1 best weights not found: {phase1_best}\n请检查训练是否正常结束，或 project/name 是否被你改过。"
        )

    # -------------------------
    # Phase 2：全量解冻微调
    # -------------------------
    model2 = YOLO(str(phase1_best))
    model2.train(
        data=str(data_yaml_path),
        imgsz=imgsz,
        epochs=120,
        batch=batch,
        device=device,
        workers=workers,
        cache=True,
        seed=seed,
        deterministic=True,
        amp=True,
        freeze=0,  # 全量解冻
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        patience=35,
        cos_lr=True,
        warmup_epochs=2,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.20,
        degrees=2.0,
        translate=0.05,
        scale=0.25,
        shear=1.0,
        perspective=0.0002,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.20,
        mixup=0.0,
        close_mosaic=15,
        project=project,
        name=phase2_name,
        plots=True,
    )

    phase2_best = Path(project) / phase2_name / "weights" / "best.pt"
    if not phase2_best.exists():
        raise FileNotFoundError(f"Phase 2 best weights not found: {phase2_best}\n请检查 Phase 2 训练是否正常结束。")

    return phase2_best


# =========================
# 3) 验证 / 测试部分
# =========================
def evaluate_model(
    weights: str,
    data_yaml: str,
    *,
    split: str = "val",
    imgsz: int = IMG_SIZE,
    batch: int = BATCH,
    device: int | str = DEVICE,
    workers: int = WORKERS,
    project: str = DEFAULT_PROJECT,
    name: str = "eval",
) -> None:
    """统一的评估函数。.

    split 可选：
    - "val"  : 验证集
    - "test" : 测试集

    说明：
    - 建议先用 val 集看结果、选参数、选阈值。
    - test 集尽量作为最终一次性确认，避免反复“看 test 调参”。
    """
    weights_path = Path(weights)
    data_yaml_path = Path(data_yaml)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml_path}")

    model = YOLO(str(weights_path))
    model.val(
        data=str(data_yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
        project=project,
        name=name,
    )


# =========================
# 4) 命令行接口
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barnacles training / validation / testing script")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="运行模式：train / val / test / all",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_YAML,
        help="已经生成好的混合数据集 yaml 路径，例如 barnacles_mix_enh/dataset_mix_enh.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="评估时使用的权重路径；mode=train 时可留空；mode=val/test 时必须给出",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL,
        help="训练起点模型，例如 yolo11n.pt",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="训练/评估输出目录",
    )
    parser.add_argument(
        "--phase1-name",
        type=str,
        default="y11n_mixenh_phase1_frozen",
        help="第一阶段训练的 run 名称",
    )
    parser.add_argument(
        "--phase2-name",
        type=str,
        default="y11n_mixenh_phase2_full",
        help="第二阶段训练的 run 名称",
    )
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="输入分辨率")
    parser.add_argument("--batch", type=int, default=BATCH, help="batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE),
        help='设备，例如 "0" 或 "cpu"',
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    return parser.parse_args()


def normalize_device(device_text: str) -> int | str:
    """命令行字符串转成 Ultralytics 可接受的 device 格式。."""
    if device_text.isdigit():
        return int(device_text)
    return device_text


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)

    if args.mode == "train":
        best_phase2 = train_two_stage(
            data_yaml=args.data,
            base_model=args.base_model,
            project=args.project,
            phase1_name=args.phase1_name,
            phase2_name=args.phase2_name,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            seed=args.seed,
        )
        print(f"[INFO] Phase 2 best weights: {best_phase2}")

    elif args.mode == "val":
        if not args.weights:
            raise ValueError("mode=val 时，必须通过 --weights 指定待评估的 best.pt 路径。")
        evaluate_model(
            weights=args.weights,
            data_yaml=args.data,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=args.project,
            name="val_eval",
        )

    elif args.mode == "test":
        if not args.weights:
            raise ValueError("mode=test 时，必须通过 --weights 指定待评估的 best.pt 路径。")
        evaluate_model(
            weights=args.weights,
            data_yaml=args.data,
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=args.project,
            name="test_eval",
        )

    elif args.mode == "all":
        best_phase2 = train_two_stage(
            data_yaml=args.data,
            base_model=args.base_model,
            project=args.project,
            phase1_name=args.phase1_name,
            phase2_name=args.phase2_name,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            seed=args.seed,
        )
        print(f"[INFO] Phase 2 best weights: {best_phase2}")

        # 训练结束后，先看 val 再看 test
        evaluate_model(
            weights=str(best_phase2),
            data_yaml=args.data,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=args.project,
            name="val_eval_after_train",
        )
        evaluate_model(
            weights=str(best_phase2),
            data_yaml=args.data,
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=args.project,
            name="test_eval_after_train",
        )


if __name__ == "__main__":
    main()
