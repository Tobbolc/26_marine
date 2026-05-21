# 用于调试module

import sys
print("python:", sys.executable)

import ultralytics
print("ultralytics:", ultralytics.__file__)

from ultralytics import YOLO
print("YOLO object:", YOLO, "type:", type(YOLO))

import onnx
print("onnx:", onnx.__file__, "type:", type(onnx))

import onnxruntime as ort
print("onnxruntime:", ort.__file__, "type:", type(ort))

import torch
print("torch:", torch.__file__, "type:", type(torch))
