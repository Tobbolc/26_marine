#!/usr/bin/env python3

"""
重新划分 YOLO 目标检测数据集（train / val / test）.

适用场景：
- 现有目录结构类似：
    dataset/
      train/images
      train/labels
      val/images        (可选)
      val/labels        (可选)
      test/images       (可选)
      test/labels       (可选)
- 标签格式为 YOLO 检测格式：
    class_id x_center y_center width height
- 需要把现有样本重新划分为新的 train/val/test
- 文件名杂乱无章也没关系：脚本按“图片相对路径/主文件名”匹配对应 label

默认策略：
1. 把现有 train/val/test（存在的都会纳入）合并为一个样本池
2. 按 85% / 10% / 5% 重新划分（可改）
3. 尽量保持类别分布稳定（基于目标框数量的贪心近似分层）
4. 把“完全相同内容”的图片自动分到同一组，避免重复样本泄漏到不同集合
5. 图片和对应 label 一起复制到新的输出目录
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BG_CLASS = "__background__"


@dataclass
class Item:
    source_split: str
    image_path: Path
    label_path: Path | None
    rel_path: Path
    stem: str
    image_hash: str
    box_counts: Counter = field(default_factory=Counter)
    num_boxes: int = 0
    class_set: set = field(default_factory=set)


@dataclass
class Group:
    group_id: str
    items: list[Item] = field(default_factory=list)
    box_counts: Counter = field(default_factory=Counter)

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def class_set(self) -> set:
        s = set()
        for item in self.items:
            s.update(item.class_set)
        return s


def parse_args():
    parser = argparse.ArgumentParser(description="重新划分 YOLO 检测数据集")
    parser.add_argument("--input", required=True, help="输入数据集根目录")
    parser.add_argument("--output", required=True, help="输出数据集根目录（建议新目录）")
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["train", "val", "test"],
        help="参与重划分的源 split，默认: train val test",
    )
    parser.add_argument("--train-ratio", type=float, default=0.85, help="训练集比例，默认 0.85")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="验证集比例，默认 0.10")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="测试集比例，默认 0.05")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="输出时复制还是移动文件，默认 copy",
    )
    parser.add_argument(
        "--group-mode",
        choices=["auto", "none", "hash", "parent"],
        default="auto",
        help=(
            "防止相似来源样本泄漏的分组策略："
            "auto=若 images 下有子目录则按子目录，否则按完全相同图片内容哈希；"
            "none=每张图独立；hash=完全相同内容归一组；parent=按相对父目录归组"
        ),
    )
    parser.add_argument(
        "--allow-missing-label",
        action="store_true",
        help="允许图片没有对应 label.txt；若缺失，则按空目标图像处理",
    )
    parser.add_argument("--min-val-images", type=int, default=1, help="验证集最少图片数，默认 1")
    parser.add_argument("--min-test-images", type=int, default=1, help="测试集最少图片数，默认 1")
    return parser.parse_args()


def ensure_ratio_valid(train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {total:.6f}")


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def parse_yolo_label(label_path: Path) -> tuple[Counter, int, set]:
    box_counts = Counter()
    class_set = set()
    num_boxes = 0

    with open(label_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{label_path} 第 {line_no} 行格式错误：{raw!r}")
            try:
                cls = int(float(parts[0]))
            except ValueError as e:
                raise ValueError(f"{label_path} 第 {line_no} 行类别编号无法解析：{raw!r}") from e
            box_counts[cls] += 1
            class_set.add(cls)
            num_boxes += 1

    if num_boxes == 0:
        box_counts[BG_CLASS] += 1
        class_set.add(BG_CLASS)

    return box_counts, num_boxes, class_set


def find_items(input_root: Path, source_splits: list[str], allow_missing_label: bool) -> list[Item]:
    items: list[Item] = []

    for split in source_splits:
        images_dir = input_root / split / "images"
        labels_dir = input_root / split / "labels"

        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            rel_path = image_path.relative_to(images_dir)
            label_path = labels_dir / rel_path.with_suffix(".txt")

            if not label_path.exists():
                if not allow_missing_label:
                    raise FileNotFoundError(
                        f"找不到对应标签文件：{label_path}\n"
                        f"图片为：{image_path}\n"
                        f"如需允许无标签图片，请加 --allow-missing-label"
                    )
                label_path = None

            if label_path is not None:
                box_counts, num_boxes, class_set = parse_yolo_label(label_path)
            else:
                box_counts = Counter({BG_CLASS: 1})
                num_boxes = 0
                class_set = {BG_CLASS}

            items.append(
                Item(
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path,
                    rel_path=rel_path,
                    stem=image_path.stem,
                    image_hash=file_md5(image_path),
                    box_counts=box_counts,
                    num_boxes=num_boxes,
                    class_set=class_set,
                )
            )

    if not items:
        raise RuntimeError("没有找到任何图片，请检查输入目录和 source-splits。")

    return items


def choose_group_key(item: Item, mode: str) -> str:
    parent_key = str(item.rel_path.parent).replace("\\", "/")
    if mode == "none":
        return f"single::{item.source_split}::{item.rel_path.as_posix()}::{item.image_hash}"
    if mode == "hash":
        return f"hash::{item.image_hash}"
    if mode == "parent":
        return f"parent::{parent_key}"
    if parent_key not in ("", "."):
        return f"parent::{parent_key}"
    return f"hash::{item.image_hash}"


def build_groups(items: list[Item], group_mode: str) -> list[Group]:
    groups_map: dict[str, Group] = {}
    for item in items:
        gid = choose_group_key(item, group_mode)
        groups_map.setdefault(gid, Group(group_id=gid))
        groups_map[gid].items.append(item)
        groups_map[gid].box_counts.update(item.box_counts)
    return list(groups_map.values())


def compute_targets(
    total_images: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_val_images: int,
    min_test_images: int,
):
    val_target = max(min_val_images, round(total_images * val_ratio))
    test_target = max(min_test_images, round(total_images * test_ratio))
    if val_target + test_target >= total_images:
        raise ValueError("val/test 数量太大，导致 train 没有剩余样本。")
    train_target = total_images - val_target - test_target
    return {"train": train_target, "val": val_target, "test": test_target}


def class_weights_from_counts(total_box_counts: Counter) -> dict[object, float]:
    return {cls: 1.0 / math.sqrt(max(cnt, 1)) for cls, cnt in total_box_counts.items()}


def group_priority(group: Group, global_box_counts: Counter):
    rarity = 0.0
    for cls in group.class_set:
        rarity += 1.0 / max(global_box_counts.get(cls, 1), 1)
    return (-rarity, -len(group.class_set), -group.size)


def incremental_cost(
    current_images: int,
    target_images: int,
    current_box_counts: Counter,
    target_box_counts: dict[object, float],
    group: Group,
    class_weights: dict[object, float],
) -> float:
    before_img_gap = abs(current_images - target_images) / max(target_images, 1)
    after_images = current_images + group.size
    after_img_gap = abs(after_images - target_images) / max(target_images, 1)

    overshoot_penalty = 0.0
    if after_images > target_images:
        overshoot_penalty = 3.0 * (after_images - target_images) / max(target_images, 1)

    before_cls_gap = 0.0
    after_cls_gap = 0.0
    for cls, target in target_box_counts.items():
        w = class_weights.get(cls, 1.0)
        before = current_box_counts.get(cls, 0)
        after = before + group.box_counts.get(cls, 0)
        norm = max(target, 1.0)
        before_cls_gap += w * abs(before - target) / norm
        after_cls_gap += w * abs(after - target) / norm

    delta_img = after_img_gap - before_img_gap
    delta_cls = after_cls_gap - before_cls_gap
    size_penalty = 0.15 * group.size / max(target_images, 1)
    return (1.0 * delta_img) + (2.0 * delta_cls) + overshoot_penalty + size_penalty


def select_groups_for_split(
    candidates: list[Group], target_images: int, split_ratio: float, global_box_counts: Counter, rng: random.Random
):
    if target_images <= 0:
        return [], candidates[:]

    target_box_counts = {cls: cnt * split_ratio for cls, cnt in global_box_counts.items()}
    class_weights = class_weights_from_counts(global_box_counts)

    remaining = candidates[:]
    remaining.sort(key=lambda g: group_priority(g, global_box_counts))

    chosen = []
    current_images = 0
    current_box_counts = Counter()

    uncovered = {cls for cls, cnt in global_box_counts.items() if cnt > 0}
    while remaining and current_images < target_images and uncovered:
        best_idx = None
        best_score = None
        for idx, group in enumerate(remaining):
            covers = uncovered.intersection(group.class_set)
            if not covers:
                continue
            cost = incremental_cost(
                current_images=current_images,
                target_images=target_images,
                current_box_counts=current_box_counts,
                target_box_counts=target_box_counts,
                group=group,
                class_weights=class_weights,
            )
            cover_bonus = sum(class_weights.get(cls, 1.0) for cls in covers)
            score = cost - 2.0 * cover_bonus
            if (best_score is None) or (score < best_score):
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        group = remaining.pop(best_idx)
        chosen.append(group)
        current_images += group.size
        current_box_counts.update(group.box_counts)
        uncovered -= group.class_set

    while remaining and current_images < target_images:
        scored = []
        for idx, group in enumerate(remaining):
            cost = incremental_cost(
                current_images=current_images,
                target_images=target_images,
                current_box_counts=current_box_counts,
                target_box_counts=target_box_counts,
                group=group,
                class_weights=class_weights,
            )
            scored.append((cost + rng.random() * 1e-6, idx))

        scored.sort(key=lambda x: x[0])
        best_idx = scored[0][1]
        group = remaining.pop(best_idx)
        chosen.append(group)
        current_images += group.size
        current_box_counts.update(group.box_counts)

    return chosen, remaining


def assign_splits(groups: list[Group], targets: dict[str, int], rng: random.Random):
    total_box_counts = Counter()
    for g in groups:
        total_box_counts.update(g.box_counts)

    candidates = groups[:]
    rng.shuffle(candidates)

    total_images = sum(targets.values())
    val_groups, remaining_after_val = select_groups_for_split(
        candidates=candidates,
        target_images=targets["val"],
        split_ratio=targets["val"] / total_images,
        global_box_counts=total_box_counts,
        rng=rng,
    )

    test_groups, train_groups = select_groups_for_split(
        candidates=remaining_after_val,
        target_images=targets["test"],
        split_ratio=targets["test"] / total_images,
        global_box_counts=total_box_counts,
        rng=rng,
    )

    return {"train": train_groups, "val": val_groups, "test": test_groups}


def flatten_assignments(assignments: dict[str, list[Group]]) -> dict[str, list[Item]]:
    out = {"train": [], "val": [], "test": []}
    for split, groups in assignments.items():
        for group in groups:
            out[split].extend(group.items)
    return out


def unique_rel_path(dest_dir: Path, rel_path: Path, source_marker: str) -> Path:
    candidate = rel_path
    if not (dest_dir / candidate).exists():
        return candidate

    stem = rel_path.stem
    suffix = rel_path.suffix
    parent = rel_path.parent
    safe_marker = hashlib.md5(source_marker.encode("utf-8")).hexdigest()[:8]
    return parent / f"{stem}_{safe_marker}{suffix}"


def copy_or_move_file(src: Path, dst: Path, action: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if action == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def copy_metadata_files(input_root: Path, output_root: Path):
    for name in ["data.yaml", "dataset.yaml", "classes.txt"]:
        src = input_root / name
        if src.exists() and src.is_file():
            shutil.copy2(src, output_root / name)


def summarize_items(name: str, items: list[Item]) -> str:
    img_count = len(items)
    box_counts = Counter()
    empty_images = 0
    for item in items:
        box_counts.update(item.box_counts)
        if item.num_boxes == 0:
            empty_images += 1
    class_part = ", ".join(f"{cls}:{cnt}" for cls, cnt in sorted(box_counts.items(), key=lambda x: str(x[0])))
    return f"[{name}] 图片数={img_count}, 空标签图片={empty_images}, 类别框统计={{ {class_part} }}"


def main():
    args = parse_args()
    ensure_ratio_valid(args.train_ratio, args.val_ratio, args.test_ratio)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"输出目录已存在且非空：{output_root}\n请换一个新的输出目录。")

    rng = random.Random(args.seed)

    items = find_items(
        input_root=input_root,
        source_splits=args.source_splits,
        allow_missing_label=args.allow_missing_label,
    )
    total_images = len(items)
    targets = compute_targets(
        total_images=total_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_val_images=args.min_val_images,
        min_test_images=args.min_test_images,
    )

    groups = build_groups(items, args.group_mode)
    assignments = assign_splits(groups, targets, rng)
    split_items = flatten_assignments(assignments)

    print("=" * 80)
    print(f"总图片数: {total_images}")
    print(f"目标划分: train={targets['train']}, val={targets['val']}, test={targets['test']}")
    print(f"group_mode={args.group_mode}, 分组总数={len(groups)}")
    print(summarize_items("train", split_items["train"]))
    print(summarize_items("val", split_items["val"]))
    print(summarize_items("test", split_items["test"]))
    print("=" * 80)

    used_rel_paths = {"train": set(), "val": set(), "test": set()}
    for split, items_in_split in split_items.items():
        images_dst_root = output_root / split / "images"
        labels_dst_root = output_root / split / "labels"

        for item in items_in_split:
            source_marker = f"{item.source_split}::{item.rel_path.as_posix()}::{item.image_hash}"
            img_rel = unique_rel_path(images_dst_root, item.rel_path, source_marker)
            while img_rel.as_posix() in used_rel_paths[split]:
                img_rel = img_rel.with_name(f"{img_rel.stem}_dup{img_rel.suffix}")
            used_rel_paths[split].add(img_rel.as_posix())

            img_dst = images_dst_root / img_rel
            copy_or_move_file(item.image_path, img_dst, args.action)

            if item.label_path is not None and item.label_path.exists():
                label_dst = labels_dst_root / img_rel.with_suffix(".txt")
                copy_or_move_file(item.label_path, label_dst, args.action)

    output_root.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(input_root, output_root)

    print(f"重划分完成，输出目录：{output_root}")


if __name__ == "__main__":
    main()
