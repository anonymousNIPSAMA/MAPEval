
import os
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from .android_banckend_protocol.android_backend import AndroidBackend, WebServerManager, reset_environment
from .mobile_agent_worflow.workflow_manager import WorkflowManager
from .datasets.evaluation_statistics import EvaluationStatistics


class MobileSelectionEvaluator:
    
    def __init__(self, log_root: str = "logs/mobile_selection_eval"):
        self.log_root = log_root
        self.android_backend = AndroidBackend()
        self.web_manager = WebServerManager()
        self.workflow_manager = WorkflowManager()
        self.statistics = EvaluationStatistics()
        
        os.makedirs(self.log_root, exist_ok=True)
        
        self.session_log_dir = os.path.join(
            self.log_root, 
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.session_log_dir, exist_ok=True)
        
        print(f"Mobile-Selection-Eval initialization completed")
        print(f"Session log directory: {self.session_log_dir}")
    
    def check_environment(self) -> bool:

        if not self.android_backend.check_device_connection():
            return False
        
        if not os.path.exists(self.web_manager.products_template_path):
            return False
        
        return True

