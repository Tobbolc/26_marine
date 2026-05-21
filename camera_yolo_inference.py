# -*- coding: utf-8 -*-
"""
海康网络相机 RTSP 取流 + YOLO 藤壶检测实时推理（低延迟版）

延迟优化策略:
  1. UDP 传输 — 网线直连/PLC 下比 TCP 延迟低（无重传等待）
  2. 后台取帧线程 — 持续取帧但只保留最新一帧，丢弃堆积帧
  3. FFmpeg 零缓冲参数 — nobuffer + probesize 最小化 + max_delay
  4. 相机端 GOP 调小 — 减少 I 帧间隔（需在相机 Web 页面配置）

用法:
    python camera_yolo_inference.py

按 'q' 退出 / 's' 截图 / 'p' 暂停
"""
import cv2
import time
import sys
import os
import threading
from collections import deque
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
RTSP_URL = "rtsp://admin:Xdj20060318@@192.168.1.64:554/Streaming/Channels/101"

# 部署链路切换到 PLC 后，如果丢包严重，建议改用子码流（带宽低，抗丢包）:
# RTSP_URL = "rtsp://admin:Xdj20060318@@192.168.1.64:554/Streaming/Channels/102"

# 传输协议: "tcp" 可靠清晰, "udp" 低延迟但可能丢包花屏
STREAM_TRANSPORT = "tcp"

MODEL_PATH = "runs/barnacles/para1_mixenh_phase2_full/weights/best.pt"

INFERENCE_SIZE = 640
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
DISPLAY_WIDTH = 960

WINDOW_NAME = "Barnacle Detection - Q:quit S:save P:pause"

# ============================================================
# 低延迟取帧器
# ============================================================

class LowLatencyCapture:
    """后台线程持续取帧，只保留最新帧，丢弃 OpenCV 内部缓冲"""

    def __init__(self, url, transport="udp"):
        self.url = url
        self.transport = transport
        self.latest_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        # 设置 FFmpeg 低延迟环境变量
        # 关键参数说明:
        #   rtsp_transport: udp/tcp — 传输协议
        #   fflags nobuffer — 禁用 FFmpeg 输入缓冲
        #   flags low_delay — 启用低延迟模式
        #   probesize 32 — 最小化流探测数据量
        #   analyzeduration 0 — 跳过流分析
        #   max_delay 100000 — 最大封包延迟 100ms (单位: 微秒)
        #   avioflags direct — 减少 avio 缓冲
        ffmpeg_opts = (
            f"rtsp_transport;{self.transport}"
            "|fflags;nobuffer"
            "|flags;low_delay"
            "|probesize;32"
            "|analyzeduration;0"
            "|max_delay;100000"
            "|avioflags;direct"
        )
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts
        print(f"FFmpeg opts: {ffmpeg_opts}")

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

        timeout = 10
        while timeout > 0:
            with self.lock:
                if self.latest_frame is not None:
                    print("First frame received")
                    return True
            time.sleep(0.1)
            timeout -= 0.1
        return False

    def _grab_loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # 最小化 OpenCV 内部缓冲队列
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            # 尽可能快的读取并丢弃旧帧，只保留最新的
            grabbed = False
            latest = None
            # 一次性把缓冲里的帧全读出来，只留最后一帧
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    latest = frame
                    grabbed = True
                else:
                    break

            if not grabbed:
                # 断流重连
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            with self.lock:
                self.latest_frame = latest

        cap.release()

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.stopped = True


# ============================================================
# 主程序
# ============================================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {model.names}")

    print(f"Opening: {RTSP_URL} (transport={STREAM_TRANSPORT})")
    cap = LowLatencyCapture(RTSP_URL, transport=STREAM_TRANSPORT)
    if not cap.start():
        print("ERROR: Stream timeout")
        sys.exit(1)

    _, test = cap.read()
    h, w = test.shape[:2]
    print(f"Resolution: {w}x{h}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, int(DISPLAY_WIDTH * h / w))

    paused = False
    infer_times = deque(maxlen=30)

    print("\nRunning. Q=quit P=pause S=screenshot")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        if not paused:
            t0 = time.time()
            results = model.predict(
                frame, imgsz=INFERENCE_SIZE,
                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                verbose=False,
            )
            dt = time.time() - t0
            infer_times.append(dt)
            plotted = results[0].plot()
        else:
            plotted = frame.copy()
            cv2.putText(plotted, "PAUSED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if infer_times:
            avg = sum(infer_times) / len(infer_times)
            cv2.putText(plotted, f"Infer: {1/avg:.1f} FPS ({avg*1000:.0f}ms)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status_color = (0, 255, 0) if not paused else (0, 0, 255)
        cv2.circle(plotted, (plotted.shape[1] - 20, 20), 8, status_color, -1)
        cv2.imshow(WINDOW_NAME, plotted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, plotted)
            print(f"Saved: {fname}")
        elif key == ord('p'):
            paused = not paused
            print(f"Paused: {paused}")

    cap.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
