"""
RTSP 连接测试脚本
依次尝试常见的海康网络摄像机 RTSP 地址，
找到能用的后打印出来并尝试显示一帧画面。.
"""

import cv2

CAMERA_IP = "192.168.1.64"
RTSP_PORT = "554"

# 常见海康 RTSP URL 模板
# 你需要填入相机用户名/密码（如果相机设置了认证）
# 常见默认用户名密码组合，按实际情况修改
USERS = ["admin"]
PASSWORDS = ["Xdj20060318@"]

# RTSP 路径模板 (user:pass@ip:port/path)
PATH_TEMPLATES = [
    "rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/101",  # Hikvision 主码流
    "rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/102",  # Hikvision 子码流
    "rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/1",  # 另一种格式
    "rtsp://{ip}:{port}/Streaming/Channels/101",  # 无认证
    "rtsp://{ip}:{port}/Streaming/Channels/102",  # 无认证
    "rtsp://{ip}/Streaming/Channels/101",  # 默认554端口
    "rtsp://{user}:{pw}@{ip}:{port}/h264/ch1/main/av_stream",  # 通用RTSP
    "rtsp://{user}:{pw}@{ip}:{port}/h265/ch1/main/av_stream",  # H.265
    "rtsp://{user}:{pw}@{ip}:{port}/live",  # 另一种
    "rtsp://{ip}:{port}/live",
    "rtsp://{ip}:8554/live",
    "rtsp://{user}:{pw}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
    "rtsp://{user}:{pw}@{ip}:{port}/ISAPI/Streaming/Channels/101",
]

urls_to_try = []
for user in USERS:
    for pw in PASSWORDS:
        for tmpl in PATH_TEMPLATES:
            try:
                url = tmpl.format(user=user, pw=pw, ip=CAMERA_IP, port=RTSP_PORT)
                if url not in urls_to_try:
                    urls_to_try.append(url)
            except KeyError:
                pass

# 去重
urls_to_try = list(dict.fromkeys(urls_to_try))

print(f"Testing {len(urls_to_try)} RTSP URLs on {CAMERA_IP}...\n")

working_urls = []
for url in urls_to_try:
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"  [OK] {url}")
            print(f"       Frame size: {frame.shape}")
            working_urls.append(url)
        else:
            print(f"  [FAIL] {url}")
    except Exception as e:
        print(f"  [ERR] {url} - {e}")

print(f"\n{'=' * 60}")
if working_urls:
    print(f"\nFound {len(working_urls)} working RTSP URLs:")
    for u in working_urls:
        print(f"  {u}")
    print(f'\nTry: python camera_rtsp_preview.py "{working_urls[0]}"')
else:
    print("\nNo working RTSP URL found.")
    print("\n可能的原因:")
    print("  1. 相机需要用户名/密码认证 — 请查看相机说明书或联系厂商")
    print("  2. 相机RTSP端口不是554 — 用nmap扫描: nmap -p 554,8554,80,443,8000 192.168.1.64")
    print("  3. 相机可能使用ONVIF或其他协议")
    print("\n  Try scanning ports first:")
    print(f"    nmap -p 1-65535 {CAMERA_IP}  # or use Hikvision SADP tool")
