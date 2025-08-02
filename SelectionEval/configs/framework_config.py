#!/usr/bin/env python3
"""
Mobile-Selection-Eval framework core configuration file
"""

from dataclasses import dataclass

# shopping
SHOPPING_CONFIG = {
    "host": "http://localhost:5173/"
}

# ADB configuration
ADB_CONFIG = {
    "adb_path": "/usr/bin/adb",
    "device_id": "your-device-id",  # Use 'adb devices' to find your device ID
    "screenshot_path": "/sdcard/screenshot.png",
    "local_screenshot_dir": "./screenshot/"
}
