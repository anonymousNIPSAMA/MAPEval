
import argparse
import sys
import os
import json
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SelectionEval.configs.framework_config import (
    AttackType, MobileAgentType, EvaluationConfig, AttackConfig
)
from SelectionEval.mobile_agent_worflow.workflow_manager import WorkflowManager
from SelectionEval.android_banckend_protocol.andriod import apply_attack

class MobileAgentEvaluator:
    
    def __init__(self, app_type: str = "shopping", agent_type:str ='v2', mode="run",epoch = 0):
        self.app_type = app_type
        self.agent_type = agent_type
        self.workflow_manager = WorkflowManager()
        self.config = None
        
        self.log_base_dir = Path(__file__).parent / "logs"
        self.session_id = epoch
        self.session_log_dir = self.log_base_dir / agent_type / app_type / f"{epoch}"
        self.mode = mode
        

    
    def setup_evaluation_config(self, max_iterations: int = 5, temperature: float = 0.0, run_iter = 1):
        self.config = EvaluationConfig(
            agent_type=self.agent_type,
            max_iterations=max_iterations,
            temperature=temperature,
            seed=1234,
            max_consecutive_failures=3,
            max_repetitive_actions=3,
            log_dir=str(self.session_log_dir) + f"/{run_iter}",
            evaluation_rounds=1,
            task_timeout=300,
            experiment_name=f"{self.app_type}_evaluation",
            dataset_name="ecommerce_shopping_tasks"
        )
        if self.mode == "run":
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        return self.config
    
    def initialize_agent(self):
        print(f"🚀 {self.agent_type}...")
        
        success = self.workflow_manager.initialize_agent(self.agent_type, self.config)
        
            
        return success