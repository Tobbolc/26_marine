"""
海康网络相机 RTSP 实时预览脚本
用法:
    python camera_rtsp_preview.py.

按 'q' 退出
按 's' 截图保存到当前目录
"""

import sys
import time

import cv2

# --- 配置 ---
# 主码流 1920x1080（用于YOLO推理），子码流 640x360（低延迟预览）
RTSP_URL = "rtsp://admin:Xdj20060318@@192.168.1.64:554/Streaming/Channels/101"
# 如果用子码流，改为:
# RTSP_URL = "rtsp://admin:Xdj20060318@@192.168.1.64:554/Streaming/Channels/102"

WINDOW_NAME = "Hikvision Camera (RTSP) - Press Q to quit, S to screenshot"

# ffmpeg 后端参数（Windows 上优化延迟）
# CAP_FFMPEG 使用 -fflags nobuffer 减少缓冲
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open RTSP stream: {RTSP_URL}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream opened: {w}x{h} @ {fps:.1f} FPS (nominal)")
    print(f"URL: {RTSP_URL}")
    print("Press 'q' to quit, 's' to save screenshot")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # 缩小显示窗口（1080p 全屏太大）
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream disconnected, retrying...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - t_start
            fps_display = 30 / elapsed
            t_start = time.time()

        # 在画面上显示FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
