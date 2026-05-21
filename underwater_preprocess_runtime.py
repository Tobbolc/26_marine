from __future__ import annotations

"""
水下图像预处理模块（运行时版本）
================================

用途
----
1) 复用你增强版训练脚本里已经验证过的水下增强链路：
   - 自适应白平衡 / 通道补偿
   - LAB-L 通道 CLAHE
   - 轻量“去雾式”局部对比增强
   - 边缘保留 + 轻微锐化

2) 作为主程序中的“单帧预处理函数”直接调用：
       frame_pre = preprocess_frame_for_inference(frame_bgr)

3) 给后续接入摄像头链路时提供一个清晰的代码骨架：
   收流 -> 帧质量检查 -> 预处理 -> YOLO 推理 -> 后处理 -> ROI/坐标输出

设计原则
--------
- 这个版本已经把“只用于生成增强训练集”的代码完全删除。
- 保留可直接复用到部署端的图像增强逻辑。
- 增加了帧质量评估和一个最基础的运行示例，方便你后续接相机/视频流。
- 尽量只依赖 OpenCV + NumPy，便于在你的主程序里集成。

注意
----
1) 这套增强链路从工程角度是“温和增强”，不是物理模型意义上的严格水下复原。
2) 它适合做检测前预处理，但建议在真实视频上保留开关，不要一开始就强制所有帧都做完整增强。
3) 如果你后续发现 CPU 占用偏高，优先检查 bilateralFilter 这一段。
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


# =========================
# 1) 基础图像增强函数
# =========================
def adaptive_white_balance_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    自适应白平衡 / 通道增益补偿（BGR 输入）

    思路：
    - 统计 B/G/R 三个通道的全局均值
    - 用“各通道均值向全局平均值靠拢”的方式计算增益
    - 水下常见问题是红色衰减更强，因此给 R 通道更宽的补偿上限

    优点：
    - 简单、稳定、速度快
    - 对水下偏蓝偏绿画面有明显修正作用

    风险：
    - 如果画面本身颜色非常偏，或者有大面积单色区域，可能会有轻微偏色
    """
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6  # B, G, R
    target = float(means.mean())

    gains = np.clip(target / means, 0.85, 1.8)
    gains[2] = np.clip(max(gains[2], 1.0), 1.0, 2.2)  # R 通道更强一点

    out = img * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_on_l_channel(
    img_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    在 LAB 色彩空间的 L 通道上做 CLAHE。

    作用：
    - 提升局部对比度
    - 尤其适合水下低对比度、轻度浑浊、局部发灰的画面

    参数：
    - clip_limit 越大，对比度提升越明显，但也越容易把噪声抬起来
    - tile_grid_size 越小，局部增强越细；越大，增强更平滑
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def soft_dehaze_like(
    img_bgr: np.ndarray,
    sigma: float = 15.0,
    amount: float = 0.20,
) -> np.ndarray:
    """
    轻量“去雾式”局部对比增强。

    这不是严格意义上的水下去雾/去散射算法，
    更接近“低频背景抑制 + 局部对比提升”。

    数学形式近似：
        out = img + amount * (img - blur(img))

    作用：
    - 让目标边界和纹理略微突出
    - 对轻度浑浊和发灰画面有效

    风险：
    - amount 太大时容易产生光晕、过冲
    - 对强噪声或大量悬浮颗粒会有放大效应

    这里 deliberately 设得比较轻，属于工程上更稳的版本。
    """
    img = img_bgr.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = img + amount * (img - blur)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def edge_preserving_refine(
    img_bgr: np.ndarray,
    d: int = 7,
    sigma_color: float = 40.0,
    sigma_space: float = 9.0,
    sharpen_gain: float = 0.12,
) -> np.ndarray:
    """
    边缘保留平滑 + 轻微锐化。

    过程：
    1) 先做双边滤波，抑制噪声，但尽量保留边缘
    2) 再用 addWeighted 做温和锐化

    说明：
    - 这是整条链路里相对更“吃算力”的一步
    - 如果后续实时性不够，先从这里优化

    实际部署建议：
    - 若画面分辨率高（例如 1920x1080）且 PC 端实时性吃紧，
      可以只对“送入检测器的分辨率”做增强，而不是对原始全分辨率做增强。
    """
    smooth = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    sharp = cv2.addWeighted(img_bgr, 1.0 + sharpen_gain, smooth, -sharpen_gain, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_underwater_bgr(
    img_bgr: np.ndarray,
    *,
    enable_awb: bool = True,
    enable_clahe: bool = True,
    enable_dehaze_like: bool = True,
    enable_refine: bool = True,
) -> np.ndarray:
    """
    按顺序执行整条水下增强链路。

    推荐顺序：
        白平衡 -> CLAHE -> 局部对比增强 -> 边缘保留轻锐化

    这也是你训练脚本里使用的增强顺序。
    """
    out = img_bgr
    if enable_awb:
        out = adaptive_white_balance_bgr(out)
    if enable_clahe:
        out = clahe_on_l_channel(out)
    if enable_dehaze_like:
        out = soft_dehaze_like(out)
    if enable_refine:
        out = edge_preserving_refine(out)
    return out


# =========================
# 2) 帧质量评估：供主程序判断是否跳帧 / 是否启用增强
# =========================
@dataclass
class FrameQuality:
    """
    简单的帧质量指标。

    这些指标不是“绝对标准”，而是工程上的启发式量：
    - brightness: 平均亮度
    - contrast: 灰度标准差
    - focus: 拉普拉斯方差，常用于模糊检测
    """
    brightness: float
    contrast: float
    focus: float


def estimate_frame_quality(frame_bgr: np.ndarray) -> FrameQuality:
    """
    估计当前帧的亮度、对比度、清晰度。
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FrameQuality(brightness=brightness, contrast=contrast, focus=focus)


def is_bad_frame(
    frame_bgr: np.ndarray,
    *,
    min_brightness: float = 20.0,
    max_brightness: float = 245.0,
    min_focus: float = 20.0,
) -> bool:
    """
    粗略判断是否为“异常帧”。

    可用于：
    - 严重过暗 / 过曝
    - 严重模糊
    - 需要直接丢弃或做特殊处理的帧

    说明：
    - 这些阈值一定要结合你的真实视频再调
    - 不建议直接把这个函数的输出当成最终决策
    - 更建议先打印日志，观察分布，再决定是否真的丢帧
    """
    q = estimate_frame_quality(frame_bgr)
    if q.brightness < min_brightness:
        return True
    if q.brightness > max_brightness:
        return True
    if q.focus < min_focus:
        return True
    return False


# =========================
# 3) 部署时复用的统一入口
# =========================
def preprocess_frame_for_inference(
    frame_bgr: np.ndarray,
    *,
    use_quality_gate: bool = False,
    force_skip_on_bad_frame: bool = False,
    enable_awb: bool = True,
    enable_clahe: bool = True,
    enable_dehaze_like: bool = True,
    enable_refine: bool = True,
) -> np.ndarray:
    """
    给检测主程序调用的单帧预处理函数。

    参数设计说明：
    ------------
    1) use_quality_gate:
       - False: 无条件执行完整增强
       - True : 先做帧质量分析，再决定增强策略

    2) force_skip_on_bad_frame:
       - 若为 True，且判断为异常帧，则直接返回原帧
       - 适合你想把“是否丢帧”逻辑放到更上层控制时使用

    推荐工程用法：
    -------------
    初版联调时，先用最简单稳定的方式：
        frame_pre = preprocess_frame_for_inference(frame, use_quality_gate=False)

    当你拿到真实视频、开始追求稳态效果时，再改成：
        frame_pre = preprocess_frame_for_inference(frame, use_quality_gate=True)

    这里的质量门控策略只是一个保守示例：
    - 如果帧已经很清晰、对比度也够，可以考虑减弱增强
    - 如果帧很差，则执行完整增强
    """
    if not use_quality_gate:
        return enhance_underwater_bgr(
            frame_bgr,
            enable_awb=enable_awb,
            enable_clahe=enable_clahe,
            enable_dehaze_like=enable_dehaze_like,
            enable_refine=enable_refine,
        )

    q = estimate_frame_quality(frame_bgr)
    bad = is_bad_frame(frame_bgr)

    if bad and force_skip_on_bad_frame:
        return frame_bgr

    # 一个较保守的门控示例：
    # - 画面已经较亮、对比度也不错时，少做一步“局部去雾式增强”
    # - 画面较差时，走完整链路
    if q.brightness > 70 and q.contrast > 35 and q.focus > 120:
        return enhance_underwater_bgr(
            frame_bgr,
            enable_awb=enable_awb,
            enable_clahe=enable_clahe,
            enable_dehaze_like=False,
            enable_refine=enable_refine,
        )

    return enhance_underwater_bgr(
        frame_bgr,
        enable_awb=enable_awb,
        enable_clahe=enable_clahe,
        enable_dehaze_like=enable_dehaze_like,
        enable_refine=enable_refine,
    )


# =========================
# 4) 给主程序接摄像头/视频流的示例骨架
# =========================
def open_video_source(source: Union[int, str]) -> cv2.VideoCapture:
    """
    打开视频源。

    source 可以是：
    - 0 / 1 / 2 ...           本地 USB 摄像头
    - "test.mp4"              本地视频文件
    - "rtsp://..."            RTSP 视频流
    - "http://..."            某些网络视频流
    - 采集卡对应的设备号

    如果你们水下链路最终是“岸上 PC 通过某个采集卡/解码器拿到视频”，
    通常这里会落到两种接法：
    1) OpenCV 直接 VideoCapture(设备号)
    2) OpenCV 直接 VideoCapture(RTSP URL)

    如果以后必须走厂家 SDK，那么也没有关系：
    - 你只需要把 SDK 回调里拿到的单帧图像转成 BGR ndarray
    - 然后传给 preprocess_frame_for_inference(frame_bgr)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def run_preprocess_demo(
    source: Union[int, str],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    use_quality_gate: bool = False,
    skip_bad_frames: bool = False,
) -> None:
    """
    预处理演示函数。

    这个函数的作用不是替代你的正式主程序，而是帮你快速做三件事：
    1) 检查视频流是否能正常进入 PC
    2) 检查这套预处理在真实视频上的观感
    3) 观察实时性大概是否够用

    你后续正式主程序建议改造成：
        while True:
            收帧
            必要的时间戳/帧号记录
            预处理
            YOLO 推理
            后处理（NMS 后的业务逻辑、ROI 合并、时序平滑）
            向控制模块输出坐标/valid/conf
    """
    cap = open_video_source(source)

    writer = None
    win_name = "underwater preprocess demo"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # 这里是上层的异常帧判断示例：
            # 如果你后续发现确实需要“严重异常帧直接跳过”，
            # 可以在这里 continue，而不是在增强函数内部决定。
            if skip_bad_frames and is_bad_frame(frame):
                continue

            t0 = time.perf_counter()
            frame_pre = preprocess_frame_for_inference(
                frame,
                use_quality_gate=use_quality_gate,
                force_skip_on_bad_frame=False,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # 为了方便观察，演示里把原图和增强图拼接显示
            vis = np.hstack([frame, frame_pre])
            q = estimate_frame_quality(frame)
            text = (
                f"brightness={q.brightness:.1f}  "
                f"contrast={q.contrast:.1f}  "
                f"focus={q.focus:.1f}  "
                f"preprocess={dt_ms:.1f}ms"
            )
            cv2.putText(vis, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            if writer is None and save_path is not None:
                save_path_obj = Path(save_path)
                save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                h, w = vis.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 1e-6:
                    fps = 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(save_path_obj), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create video writer: {save_path}")

            if writer is not None:
                writer.write(vis)

            if show:
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


# =========================
# 5) 你后续真正主程序里，推荐这样接入
# =========================
def example_runtime_pipeline_note() -> None:
    """
    这里只写说明，不执行任何逻辑。

    推荐你的视觉主程序框架大致如下：

        1. 从相机/RTSP/采集卡收帧
        2. 给每帧加 frame_id、timestamp
        3. 可选：异常帧检测（过曝、掉帧、严重模糊）
        4. 调 preprocess_frame_for_inference(frame)
        5. 调 YOLO 模型推理
        6. 业务后处理：
            - 置信度筛选
            - 多框合并 / 选主目标
            - ROI 扩张
            - 坐标换算
            - 时序平滑（EMA/Kalman）
            - 连续 N 帧命中才 valid=1
        7. 将 cx, cy, w, h, conf, valid 通过串口 / UDP / TCP 发给控制模块
        8. 记录日志：原始帧耗时、预处理耗时、推理耗时、输出状态

    如果你后续还要加其他预处理，建议插入位置如下：

        A. 镜头畸变校正（cv2.undistort）
           -> 放在所有增强之前
           -> 前提是你已经标定过相机

        B. 去噪 / 去模糊
           -> 放在白平衡之前或之后都可以试
           -> 但不要一开始就堆太重的算法，先看真实视频是否真的需要

        C. resize / letterbox
           -> 若你直接调用 Ultralytics 的 model.predict / model(frame)，
              它内部会处理 resize / letterbox
           -> 所以大多数情况下，你这里不用自己再做一遍

        D. ROI 裁剪
           -> 如果控制策略允许，只处理图像中心区域或历史 ROI 附近，
              可以明显降低计算量

    结论：
        这个模块建议你先作为“默认预处理基线”使用，
        但保留开关，便于和 raw 输入做 A/B 对比。
    """
    pass


# =========================
# 6) 命令行入口
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Underwater preprocessing runtime module")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源：摄像头编号(如 0)、视频文件路径、RTSP 地址等",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="可选：保存拼接后的演示视频路径，例如 outputs/preprocess_demo.mp4",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹窗显示，只进行处理/保存",
    )
    parser.add_argument(
        "--quality-gate",
        action="store_true",
        help="启用基于帧质量的增强门控策略",
    )
    parser.add_argument(
        "--skip-bad-frames",
        action="store_true",
        help="对明显异常帧直接跳过（演示用，正式主程序请按业务需要修改）",
    )
    return parser.parse_args()


def parse_source(source_text: str) -> Union[int, str]:
    """
    命令行里如果写的是 "0"，这里会转成整数 0；
    如果是文件路径或 RTSP URL，则保持字符串。
    """
    if source_text.isdigit():
        return int(source_text)
    return source_text


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    save_path = args.save if args.save else None

    run_preprocess_demo(
        source=source,
        save_path=save_path,
        show=not args.no_show,
        use_quality_gate=args.quality_gate,
        skip_bad_frames=args.skip_bad_frames,
    )


if __name__ == "__main__":
    main()
