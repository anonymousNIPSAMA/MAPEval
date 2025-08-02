import json
import os
import random
import subprocess
import time
from SelectionEval.configs.framework_config import ADB_CONFIG,SHOPPING_CONFIG, AttackConfig, AttackType


def reset(
    app: str = "shopping", test_case: str = None
):  
    adb_path = ADB_CONFIG.get("adb_path", "adb")
    device = ADB_CONFIG.get("device_id", None)
    if app in ["shopping","meituan"]:
        test_case = f"{app}_{test_case}"
        if app == "shopping":
            url = SHOPPING_CONFIG.get("host", "http://localhost:5173/")
        if app == "meituan":
            url = "http://localhost:8080/"
        
        
        command = "adb shell am force-stop org.mozilla.firefox"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(1)
        
        command = f'{adb_path} shell am start -a android.intent.action.VIEW -d "{url}" org.mozilla.firefox'

        # if device:
        #     command = f'{adb_path} -s {device} shell am start -a android.intent.action.VIEW -d "{url}'
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(6)
        return test_case
    # elif app == "google":
    #     kill_browser_cmd = f"{adb_path} shell am force-stop  com.google.android.googlequicksearchbox"
    #     subprocess.run(kill_browser_cmd, capture_output=True, text=True, shell=True)
    #     time.sleep(2)

    #     start_cmd = f"{adb_path} -s {device} shell  start -a android.intent.action.WEB_SEARCH"
    #     subprocess.run(start_cmd, capture_output=True, text=True, shell=True)
    #     time.sleep(5)

    elif app in ["amazon","foodpanda","booking","tripadvisor","maps","walmart",'imdb','tradingview','google','travel']:
        if app == "amazon":
            package = "com.amazon.mShop.android.shopping"
        elif app == "foodpanda":
            package = "com.global.foodpanda.android"
        elif app == "booking":
            package = "com.booking"
        elif app == "tripadvisor":
            package = "com.tripadvisor.tripadvisor" 
        elif app == "maps":
            package = "com.google.android.apps.maps"
        elif app == "walmart":
            package = "com.walmart.android"
        elif app == "imdb":
            package = "com.imdb.mobile"
        elif app == "tradingview":
            package = "com.tradingview.tradingviewapp"
        elif app == "google":
            package = "com.google.android.googlequicksearchbox"
        elif app == "travel":
            package = "com.alivakili.travel"
            
        stop_cmd = f"{adb_path} shell am force-stop {package}"
        start_cmd = f"{adb_path} shell monkey -p {package} -c android.intent.category.LAUNCHER 1"

        

        if device:
            stop_cmd = f"{adb_path} -s {device} shell am force-stop {package}"
            start_cmd = f"{adb_path} -s {device} shell monkey -p {package} -c android.intent.category.LAUNCHER 1"

        subprocess.run(stop_cmd, capture_output=True, text=True, shell=True)
        time.sleep(1.5)
        subprocess.run(start_cmd, capture_output=True, text=True, shell=True)
        time.sleep(2.5)
        
        time.sleep(5)



    else:
        raise ValueError(f"Unsupported app: {app}. Supported apps: shopping.")