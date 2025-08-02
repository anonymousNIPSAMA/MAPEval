#!/usr/bin/env python3
"""
Android后端协议 - 负责与Android设备通信和Web服务器管理
"""

import subprocess
import time
import os
import json
from typing import Optional, Dict, Any
import shutil

from ..configs.framework_config import WEB_CONFIG, ADB_CONFIG


class AndroidBackend:
    """Android设备后端管理器"""
    
    def __init__(self, adb_path: str = "adb"):
        self.adb_path = adb_path
        self.device_id = ADB_CONFIG.get("device_id")
        self.screenshot_path = ADB_CONFIG.get("screenshot_path", "/sdcard/screenshot.png")
        self.local_screenshot_dir = ADB_CONFIG.get("local_screenshot_dir", "./screenshots/")
        
        # 确保本地截图目录存在
        os.makedirs(self.local_screenshot_dir, exist_ok=True)
    
    def _run_adb_command(self, command: str) -> subprocess.CompletedProcess:
        """执行ADB命令"""
        if self.device_id:
            full_command = f"{self.adb_path} -s {self.device_id} {command}"
        else:
            full_command = f"{self.adb_path} {command}"
        
        print(f"执行ADB命令: {full_command}")
        return subprocess.run(full_command, capture_output=True, text=True, shell=True)
    
    def check_device_connection(self) -> bool:
        """检查设备连接状态"""
        try:
            result = self._run_adb_command("devices")
            return result.returncode == 0 and "device" in result.stdout
        except Exception as e:
            print(f"检查设备连接失败: {e}")
            return False
    
    def open_url(self, url: str, test_case: Optional[str] = None) -> bool:
        """在Android设备上打开指定URL"""
        try:
            if test_case:
                full_url = f"{url}?test_case={test_case}"
            else:
                full_url = url
            
            command = f'shell am start -a android.intent.action.VIEW -d "{full_url}"'
            result = self._run_adb_command(command)
            
            if result.returncode == 0:
                print(f"成功打开URL: {full_url}")
                time.sleep(3)  # 等待页面加载
                return True
            else:
                print(f"打开URL失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"打开URL异常: {e}")
            return False
    
    def take_screenshot(self, local_filename: Optional[str] = None) -> Optional[str]:
        """截取设备屏幕截图"""
        try:
            # 在设备上截图
            result = self._run_adb_command(f"shell screencap -p {self.screenshot_path}")
            if result.returncode != 0:
                print(f"设备截图失败: {result.stderr}")
                return None
            
            # 拉取截图到本地
            if local_filename is None:
                local_filename = f"screenshot_{int(time.time())}.png"
            
            time.sleep(1)  # 等待截图文件生成
            result = self._run_adb_command(f"pull {self.screenshot_path} {local_filename}")
            
            if result.returncode == 0:
                print(f"截图保存到: {local_filename}")
                return local_filename
            else:
                print(f"拉取截图失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"截图异常: {e}")
            return None
    
    def click_coordinates(self, x: int, y: int) -> bool:
        """点击指定坐标"""
        try:
            command = f"shell input tap {x} {y}"
            result = self._run_adb_command(command)
            
            if result.returncode == 0:
                print(f"点击坐标: ({x}, {y})")
                time.sleep(1)  # 等待响应
                return True
            else:
                print(f"点击失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"点击异常: {e}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 500) -> bool:
        """滑动屏幕"""
        try:
            command = f"shell input swipe {x1} {y1} {x2} {y2} {duration}"            
            # command = f"shell input touchscreen {x1} {y1} {x2} {y2} {duration}"

            result = self._run_adb_command(command)
            
            if result.returncode == 0:
                print(f"滑动: ({x1}, {y1}) -> ({x2}, {y2})")
                time.sleep(1)
                return True
            else:
                print(f"滑动失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"滑动异常: {e}")
            return False
    
    def input_text(self, text: str) -> bool:
        """输入文本"""
        try:
            # 转义特殊字符
            escaped_text = text.replace(" ", "%s").replace("&", "\\&")
            command = f'shell input text "{escaped_text}"'
            result = self._run_adb_command(command)
            
            if result.returncode == 0:
                print(f"输入文本: {text}")
                time.sleep(1)
                return True
            else:
                print(f"输入文本失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"输入文本异常: {e}")
            return False
    
    def go_back(self) -> bool:
        """返回上一页"""
        try:
            command = "shell input keyevent KEYCODE_BACK"
            result = self._run_adb_command(command)
            
            if result.returncode == 0:
                print("执行返回操作")
                time.sleep(1)
                return True
            else:
                print(f"返回操作失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"返回操作异常: {e}")
            return False


class WebServerManager:
    """Web服务器管理器"""
    
    def __init__(self):
        self.host = WEB_CONFIG["host"]
        self.port = WEB_CONFIG["port"]
        self.products_template_path = WEB_CONFIG["products_template_path"]
        self.products_output_path = WEB_CONFIG["products_output_path"]
    
    def get_base_url(self) -> str:
        """获取基础URL"""
        return f"http://{self.host}:{self.port}"
    
    def reset_products(self) -> bool:
        """重置商品文件到模板状态"""
        try:
            if os.path.exists(self.products_template_path):
                shutil.copy2(self.products_template_path, self.products_output_path)
                print("商品文件已重置到模板状态")
                return True
            else:
                print(f"商品模板文件不存在: {self.products_template_path}")
                return False
        except Exception as e:
            print(f"重置商品文件失败: {e}")
            return False
    
    def load_products_template(self) -> Optional[Dict[str, Any]]:
        """加载商品模板"""
        try:
            with open(self.products_template_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载商品模板失败: {e}")
            return None
    
    def save_products(self, products_data: Dict[str, Any]) -> bool:
        """保存商品数据"""
        try:
            with open(self.products_output_path, "w", encoding="utf-8") as f:
                json.dump(products_data, f, ensure_ascii=False, indent=2)
            print("商品数据已保存")
            return True
        except Exception as e:
            print(f"保存商品数据失败: {e}")
            return False
    
    def update_product(self, product_id: int, updates: Dict[str, Any]) -> bool:
        """更新指定商品的信息"""
        try:
            # 加载当前商品数据
            products_data = self.load_products_template()
            if not products_data:
                return False
            
            products = products_data.get("products", [])
            
            # 找到并更新目标商品
            for product in products:
                if product.get("id") == product_id:
                    product.update(updates)
                    print(f"更新商品ID {product_id}: {updates}")
                    break
            else:
                print(f"未找到商品ID: {product_id}")
                return False
            
            return self.save_products(products_data)
            
        except Exception as e:
            return False


def reset_environment(android_backend: AndroidBackend, web_manager: WebServerManager, 
                     test_case: Optional[str] = None) -> bool:
    try:
        web_manager.reset_products()
        
        base_url = web_manager.get_base_url()
        success = android_backend.open_url(base_url, test_case)
        
    
        
        return success
        
    except Exception as e:
        return False
