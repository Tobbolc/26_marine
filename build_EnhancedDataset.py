"""
build_underwater_enhanced_dataset.py.

用途：
1. 读取现有 YOLO 检测数据集（目录式数据集 + dataset.yaml）
2. 对训练集图像执行水下图像增强
3. 构建新的“混合训练集”：
   - train: 原图 + 增强图
   - val: 原图
   - test: 原图
4. 输出新的 dataset yaml，供后续多类别/单类别训练直接使用

说明：
- 这个脚本只保留“水下图像增强 + 生成新数据集”的部分
- 不包含训练、验证、测试代码
- 适用于后续构建三类海生物混合数据集，只要你的标签已经统一成 YOLO 格式即可

依赖：
    pip install opencv-python pyyaml numpy

示例：
    python build_underwater_enhanced_dataset.py \
        --data-yaml "D:/datasets/marine_3cls/dataset.yaml" \
        --out-root "marine_3cls_mix_enh"

输出：
    out-root/
      ├─ images/
      │   ├─ train/   (原图 + *_enh 增强图)
      │   ├─ val/     (原图)
      │   └─ test/    (原图)
      ├─ labels/
      │   ├─ train/   (与图像对应的标签)
      │   ├─ val/
      │   └─ test/
      └─ dataset_mix_enh.yaml
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# 1) 水下图像增强模块
#    这部分可以直接复用到你后续的单帧推理预处理里
# ============================================================
def adaptive_white_balance_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """自适应白平衡 / 通道补偿（BGR 输入）.

    作用：
    - 缓解水下图像常见的偏蓝、偏绿问题
    - 对红通道给予更大的补偿范围，因为水下红光衰减通常更明显

    输入：
        img_bgr: OpenCV 读取的 BGR 图像，uint8

    输出：
        增强后的 BGR 图像，uint8
    """
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6  # B, G, R
    target = float(means.mean())

    gains = np.clip(target / means, 0.85, 1.8)
    gains[2] = np.clip(max(gains[2], 1.0), 1.0, 2.2)  # 红通道允许更强补偿

    out = img * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_on_l_channel(img_bgr: np.ndarray) -> np.ndarray:
    """在 LAB 空间的 L 通道上做 CLAHE（自适应直方图均衡）.

    作用：
    - 提升暗部和局部区域的可见性
    - 比直接在 RGB/BGR 通道上拉伸更稳

    参数设置较保守，避免过增强。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def soft_dehaze_like(img_bgr: np.ndarray) -> np.ndarray:
    """轻量“去雾式”局部对比增强.

    说明：
    - 这里不是严格意义上的物理去雾
    - 而是通过“原图 - 低频模糊”来抬升局部对比度
    - 强度刻意设得较轻，减少光晕和过冲

    更适合工程快速落地，而不是追求复杂复原。
    """
    img = img_bgr.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)
    out = img + 0.20 * (img - blur)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def edge_preserving_refine(img_bgr: np.ndarray) -> np.ndarray:
    """边缘保留 + 轻微锐化.

    作用：
    - bilateralFilter 在降噪时尽量保留边缘
    - addWeighted 做轻微锐化，让壳体边缘更清楚

    注意：
    - 这是整条增强链里相对较耗时的一步
    - 如果后续实时性不够，可以优先考虑把这一步设成可开关
    """
    smooth = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=40, sigmaSpace=9)
    sharp = cv2.addWeighted(img_bgr, 1.12, smooth, -0.12, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_underwater_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """完整水下增强主函数.

    顺序：
    1) 自适应白平衡 / 通道补偿
    2) CLAHE
    3) 轻量局部对比增强
    4) 边缘保留 + 轻微锐化
    """
    out = adaptive_white_balance_bgr(img_bgr)
    out = clahe_on_l_channel(out)
    out = soft_dehaze_like(out)
    out = edge_preserving_refine(out)
    return out


# ============================================================
# 2) YOLO 数据集处理工具
# ============================================================
def load_dataset_cfg(yaml_path: Path) -> dict:
    """读取原始数据集的 yaml 配置."""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid dataset yaml: {yaml_path}")
    return cfg


def resolve_entry(dataset_root: Path, entry: str | None) -> Path | None:
    """将 train/val/test 在 yaml 里的相对路径解析成绝对路径."""
    if entry is None:
        return None
    p = Path(entry)
    if not p.is_absolute():
        p = (dataset_root / p).resolve()
    return p


def replace_images_with_labels(path_obj: Path) -> Path:
    """将 .../images/... 路径替换成 .../labels/... 适用于标准 YOLO 目录式数据集.
    """
    parts = list(path_obj.parts)
    if "images" not in parts:
        raise ValueError(f"Path does not contain 'images': {path_obj}")
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts)


def ensure_clean_dir(path_obj: Path) -> None:
    """确保输出目录为空目录： - 若已存在，则删除重建 - 若不存在，则直接创建.

    注意： 这个操作会清空 out_root，请不要把 out_root 设到重要目录。
    """
    if path_obj.exists():
        shutil.rmtree(path_obj)
    path_obj.mkdir(parents=True, exist_ok=True)


def copy_label(src_txt: Path, dst_txt: Path) -> None:
    """拷贝对应标签文件；如果原图无标签，则写空 txt 这样可以避免 YOLO 扫描数据集时报错.
    """
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    if src_txt.exists():
        shutil.copy2(src_txt, dst_txt)
    else:
        dst_txt.write_text("", encoding="utf-8")


def iter_images(dir_path: Path) -> Iterator[Path]:
    """递归遍历目录中的所有图像文件."""
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


# ============================================================
# 3) 构建“混合增强数据集”
#    - train: 原图 + 增强图
#    - val: 原图
#    - test: 原图
# ============================================================
def build_mixed_dataset(raw_yaml: str, out_root: str) -> str:
    """根据原始 YOLO 数据集构建新的混合数据集.

    参数：
        raw_yaml: 原始数据集 yaml 路径
        out_root: 新数据集输出目录

    返回：
        新生成的 dataset_mix_enh.yaml 的路径（字符串）
    """
    raw_yaml_path = Path(raw_yaml).resolve()
    cfg = load_dataset_cfg(raw_yaml_path)

    # 如果 yaml 里写了 path，就以它为根目录；否则默认取 yaml 所在目录
    dataset_root = Path(cfg.get("path", raw_yaml_path.parent)).resolve()
    out_root_path = Path(out_root).resolve()
    ensure_clean_dir(out_root_path)

    split_map = {
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "test": cfg.get("test"),
    }

    # 写入新的数据集 yaml
    new_cfg: dict[str, object] = {
        "path": str(out_root_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }

    # 优先保留 names；若没有 names，则保留 nc
    if "names" in cfg:
        new_cfg["names"] = cfg["names"]
    if "nc" in cfg and "names" not in cfg:
        new_cfg["nc"] = cfg["nc"]

    for split, entry in split_map.items():
        if entry is None:
            continue

        in_img_dir = resolve_entry(dataset_root, entry)
        if in_img_dir is None or not in_img_dir.exists() or not in_img_dir.is_dir():
            raise ValueError(f"当前脚本假设 YOLO 数据集采用目录式组织。split='{split}' 解析后路径为: {in_img_dir}")

        in_lbl_dir = replace_images_with_labels(in_img_dir)
        out_img_dir = out_root_path / "images" / split
        out_lbl_dir = out_root_path / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        num_raw = 0
        num_enh = 0

        for img_path in iter_images(in_img_dir):
            rel = img_path.relative_to(in_img_dir)

            # 1) 复制原图
            raw_dst = out_img_dir / rel
            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, raw_dst)

            # 2) 复制原图对应标签
            label_src = (in_lbl_dir / rel).with_suffix(".txt")
            label_dst = (out_lbl_dir / rel).with_suffix(".txt")
            copy_label(label_src, label_dst)
            num_raw += 1

            # 3) 仅在训练集额外生成增强图副本
            if split == "train":
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {img_path}")

                img_enh = enhance_underwater_bgr(img)

                enh_name = rel.with_name(f"{rel.stem}_enh{rel.suffix}")
                enh_img_dst = out_img_dir / enh_name
                enh_img_dst.parent.mkdir(parents=True, exist_ok=True)

                ok = cv2.imwrite(str(enh_img_dst), img_enh)
                if not ok:
                    raise RuntimeError(f"Failed to write image: {enh_img_dst}")

                enh_label_dst = (out_lbl_dir / enh_name).with_suffix(".txt")
                copy_label(label_src, enh_label_dst)
                num_enh += 1

        print(f"[{split}] raw={num_raw}, enhanced_added={num_enh}")

    new_yaml_path = out_root_path / "dataset_mix_enh.yaml"
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"New dataset yaml written to: {new_yaml_path}")
    return str(new_yaml_path)


# ============================================================
# 4) 给后续主程序复用的单帧预处理接口
# ============================================================
def preprocess_frame_for_inference(frame_bgr: np.ndarray) -> np.ndarray:
    """后续如果你在实时视频主程序中做单帧增强，可以直接调用这个函数。.

    典型接法（伪代码）：
        cap = cv2.VideoCapture(0)  # 或 RTSP / 视频文件
        ret, frame = cap.read()
        frame_pre = preprocess_frame_for_inference(frame)
        # 然后再送入 YOLO 推理

    后续你还可以在主程序里加入：
    - 模糊帧过滤（拉普拉斯方差）
    - 过曝/欠曝检测
    - 去畸变
    - ROI 后处理
    - 跟踪与时序平滑
    """
    return enhance_underwater_bgr(frame_bgr)


# ============================================================
# 5) 命令行入口
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 YOLO 用的水下增强混合数据集（train=原图+增强图，val/test=原图）")
    parser.add_argument(
        "--data-yaml",
        type=str,
        required=True,
        help="原始 YOLO 数据集 yaml 路径",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="新数据集输出目录，例如 marine_3cls_mix_enh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_mixed_dataset(args.data_yaml, args.out_root)


if __name__ == "__main__":
    main()
