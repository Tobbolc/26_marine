# -*- coding: utf-8 -*-
import os
import sys
import platform
from ctypes import *
from ctypes import wintypes

print("Python:", sys.executable)
print("Python arch:", platform.architecture())

mvimport_path = os.path.join(os.path.dirname(__file__), "MVSdevelopment", "Samples", "Python", "MvImport")
sys.path.insert(0, mvimport_path)

from MvCameraControl_class import *

kernel32 = WinDLL("kernel32", use_last_error=True)

GetModuleHandleW = kernel32.GetModuleHandleW
GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
GetModuleHandleW.restype = wintypes.HMODULE

GetModuleFileNameW = kernel32.GetModuleFileNameW
GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
GetModuleFileNameW.restype = wintypes.DWORD

h = GetModuleHandleW("MvCameraControl.dll")
if h:
    buf = create_unicode_buffer(1024)
    GetModuleFileNameW(h, buf, 1024)
    print("Loaded MvCameraControl.dll:", buf.value)
else:
    print("MvCameraControl.dll not loaded yet")

MvCamera.MV_CC_Initialize()
h = GetModuleHandleW("MvCameraControl.dll")
if h:
    buf = create_unicode_buffer(1024)
    GetModuleFileNameW(h, buf, 1024)
    print("Loaded MvCameraControl.dll after init:", buf.value)
else:
    print("MvCameraControl.dll still not loaded")

MvCamera.MV_CC_Finalize()