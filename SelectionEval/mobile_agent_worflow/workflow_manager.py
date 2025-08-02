
import time
import os
import json
import sys
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from analysize import check_is_list_page, check_is_select
from edit import apply_image_replace

from ..configs.framework_config import  MobileAgentType, EvaluationConfig
from benchmark.attack import AttackConfig
from ..android_banckend_protocol.andriod import reset
from PIL import Image

@dataclass
class AgentExecutionResult:
    success: bool
    selected_product_id: str = ""
    test_case: str = ""
    execution_time: float = 0.0
    steps_taken: int = 0
    error_message: Optional[str] = None
    log_data: Optional[Dict[str, Any]] = None
    instruction: str = ""  # 任务指令




class BaseMobileAgent(ABC):
    """Mobile Agent基类"""
    
    def __init__(self, agent_type: MobileAgentType):
        self.agent_type = agent_type
        self.config = None
        self.attack_setting:AttackConfig  = None
    
    @abstractmethod
    def initialize(self, config: EvaluationConfig) -> bool:
        pass
    
    @abstractmethod
    def execute_task(self, task_instruction: str, test_case: Optional[str] = None) -> AgentExecutionResult:
        pass
    
    
    def cleanup(self):
        pass

    def _capture_step_screenshot(self):
        
        print(self.log_root)
        if not self.log_root:
            return False
            
        try:
            screenshot_filename = f"{self.log_root}/{self.step_count}.jpg"
            
            screenshot_path = self.android_backend.take_screenshot(screenshot_filename)

            # image_path = "./screenshot/screenshot.png"
            save_path = "./screenshot/screenshot.jpg"

            read_path = apply_image_replace(screenshot_path, self.app, self.attack_setting,screenshot_path)

            image = Image.open(screenshot_path)
            image.convert("RGB").save(save_path, "JPEG")


            self.step_count += 1

            return check_is_select(self.log_root)
            
        except Exception as e:
            self.step_count += 1

            return False

        # ...existing code...
    def _setup_screenshot_capture(self, task_id: str=""):
        try:
            from ..android_banckend_protocol.android_backend import AndroidBackend
            from ..configs.framework_config import ADB_CONFIG
                      
            adb_path = ADB_CONFIG.get("adb_path", "adb")
            self.android_backend = AndroidBackend(adb_path=adb_path)
            
            self.step_count = 1
                        
        except Exception as e:
            self.android_backend = None

class MobileAgentEWorkflow(BaseMobileAgent):
    
    def __init__(self):
        super().__init__(MobileAgentType.MOBILE_AGENT_E)
        self.perceptor = None
        self.log_root = None
        self.run_name = None
        self.default_perception_args = None
        self.mobile_agent_e_path = None
        self._inference_module = None
        self.selection_id = ""
        
        self.step_count = 0
        self.screenshot_dir = None
        self.android_backend = None
        
    def _load_inference_module(self):
        if self._inference_module is not None:
            return self._inference_module
            
        try:
            import inference_agent_E
            self._inference_module = inference_agent_E
            return self._inference_module
        except ImportError:
            pass
        
        try:
            spec = importlib.util.spec_from_file_location(
                "inference_agent_E", 
                os.path.join(self.mobile_agent_e_path, "inference_agent_E.py")
            )
            self._inference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._inference_module)
            return self._inference_module
        except Exception as e:
            raise ImportError(f"无法加载inference_agent_E模块: {e}")
        
    def initialize(self, config: EvaluationConfig) -> bool:
        try:
            import torch
            
            self.mobile_agent_e_path = os.path.join(
                os.path.dirname(__file__), 
                "MobileAgent", 
                "Mobile-Agent-E"
            )
            if self.mobile_agent_e_path not in sys.path:
                sys.path.insert(0, self.mobile_agent_e_path)
            
            inference_module = self._load_inference_module()
            
            self.config = config
            self.default_perception_args = inference_module.DEFAULT_PERCEPTION_ARGS
            
            self.log_root = config.log_dir
            
            self.perceptor = inference_module.Perceptor(
                adb_path=os.environ.get("ADB_PATH", "adb"),
                perception_args=self.default_perception_args
            )
            
            torch.manual_seed(config.seed or 1234)
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False


    def select_hook_function(self,*args,**kwargs) -> bool:
        test_case = self.test_case
        app = self.app
        return self._capture_step_screenshot()
    
    async def simulation(self, *args, **kwargs) -> List[str]:
        
        return await self._inference_module.run_agent_e_simulation(*args,**kwargs)
        
    def execute_task(self, task_instruction: str, test_case: Optional[str] = None,app = "shopping") -> AgentExecutionResult:
        start_time = time.time()
        try:
            inference_module = self._load_inference_module()
            run_single_task = inference_module.run_single_task
            task_id = test_case or f"task_{int(time.time())}"
            self._setup_screenshot_capture(task_id)

            self.test_case  = reset(app,test_case)
            self.app = app
            
            run_single_task(
                instruction=task_instruction,
                future_tasks=[],  
                run_name=self.run_name,
                log_root=self.log_root,
                task_id=task_id,
                tips_path=None,  
                shortcuts_path=None,  
                persistent_tips_path=None, 
                persistent_shortcuts_path=None,  
                perceptor=self.perceptor,
                perception_args=self.default_perception_args,
                max_itr=self.config.max_iterations or 40,
                max_consecutive_failures=self.config.max_consecutive_failures or 5,
                max_repetitive_actions=self.config.max_repetitive_actions or 5,
                overwrite_log_dir=True,
                err_to_manager_thresh=2,
                enable_experience_retriever=False,
                temperature=self.config.temperature or 0.0,
                screenrecord=False,
                select_hook_function = self.select_hook_function
            )
            
            execution_time = time.time() - start_time
            return AgentExecutionResult(
                success=True,
                selected_product_id=self.selection_id,
                execution_time=execution_time,
                test_case=self.test_case,
                instruction=task_instruction,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            traceback.print_exc()
            
            return AgentExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    

class MobileAgentV2Workflow(BaseMobileAgent):
    
    def __init__(self):
        super().__init__(MobileAgentType.MOBILE_AGENT_V2)
        self.v2_module  = None
        self._module_loaded = False
    
    def initialize(self, config: EvaluationConfig) -> bool:
        try:
            self.config = config
            
            self._load_v2_module()
            
            return True
        except Exception as e:
            return False
    
    def _load_v2_module(self):
        if self._module_loaded and self.v2_module is not None:
            return self.v2_module
            
        try:
            import importlib.util
            import sys
            import os
            
            v2_path = os.path.join(os.path.dirname(__file__), "MobileAgent", "Mobile-Agent-v2", "run.py")
            
            if not os.path.exists(v2_path):
                raise FileNotFoundError(f"Mobile-Agent-V2 run.py 文件不存在: {v2_path}")
            
            spec = importlib.util.spec_from_file_location("run_v2", v2_path)
            self.v2_module = importlib.util.module_from_spec(spec)
            
            v2_dir = os.path.dirname(v2_path)
            if v2_dir not in sys.path:
                sys.path.insert(0, v2_dir)
            
            spec.loader.exec_module(self.v2_module)
            
            if not hasattr(self.v2_module, 'run_v2_workflow'):
                raise AttributeError("Mobile-Agent-V2模块中未找到run_v2_workflow函数")
            
            self._module_loaded = True
            return self.v2_module
            
        except Exception as e:
            raise ImportError(f"无法加载Mobile-Agent-V2模块: {e}")
        

    def select_hook_function(self,*args,**kwargs) -> bool:
        return self._capture_step_screenshot()

    async def simulation(self,*args,**kwargs) -> List[str]:

        return await self.v2_module.run_v2_simulation(*args,**kwargs)


    def execute_task(self, task_instruction: str, test_case: Optional[str] = None, app: str = "shopping") -> AgentExecutionResult:
        start_time = time.time()
        self.app = app

        self.log_root = self.config.log_dir

        self._setup_screenshot_capture(test_case)
        reset(app,test_case)

        
        try:
            if not self._module_loaded or self.v2_module is None:
                raise RuntimeError("Mobile-Agent-V2模块未正确加载，请先调用initialize方法")

    
            
            self.v2_module.run_v2_workflow(task_instruction,
                                           max_iterations=self.config.max_iterations or 40,
                                            select_hook_function = self.select_hook_function,
                                            log_path=self.log_root
                                           )
            
            return AgentExecutionResult(
                success=True,
                selected_product_id="",
                execution_time=0,
                steps_taken=5, 
                instruction=task_instruction,
                test_case=test_case or ""
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            traceback.print_exc()
            
            return AgentExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                instruction=task_instruction,
                test_case=test_case or ""
            )
    
    
    def cleanup(self):
        try:
            if self.v2_module:
                self.v2_module = None
                self._module_loaded = False
        except Exception as e:
            print(f"❌ Mobile-Agent-V2资源清理失败: {e}")
        finally:
            super().cleanup()


class AppAgentWorkflow(BaseMobileAgent):
    
    def __init__(self):
        super().__init__(MobileAgentType.APPAGENT)
        self.appagent_module = None
        self._module_loaded = False
        self.log_root = None
        self.app = "shopping"
        self.test_case = None
        
        self.step_count = 0
        self.screenshot_dir = None
        self.android_backend = None
        
    def initialize(self, config: EvaluationConfig) -> bool:
        try:
            self.config = config
            
            self._load_appagent_module()
            
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
    
    def _load_appagent_module(self):
        if self._module_loaded and self.appagent_module is not None:
            return self.appagent_module
            
        try:
            import importlib.util
            import sys
            import os
            
            appagent_path = os.path.join(
                os.path.dirname(__file__), 
                "AppAgent", 
                "scripts", 
                "task_executor.py"
            )
            
            if not os.path.exists(appagent_path):
                raise FileNotFoundError(f"AppAgent task_executor.py 文件不存在: {appagent_path}")
            
            appagent_scripts_dir = os.path.dirname(appagent_path)  # scripts目录
            appagent_root = os.path.dirname(appagent_scripts_dir)  # AppAgent根目录
            
            original_cwd = os.getcwd()
            
            try:
                modules_to_clear = ['utils', 'and_controller', 'config', 'model', 'prompts']
                for module_name in modules_to_clear:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                
                if appagent_scripts_dir not in sys.path:
                    sys.path.insert(0, appagent_scripts_dir)
                if appagent_root not in sys.path:
                    sys.path.insert(0, appagent_root)
                
             
                os.chdir(appagent_root)
                
                utils_spec = importlib.util.spec_from_file_location(
                    "appagent_utils", 
                    os.path.join(appagent_scripts_dir, "utils.py")
                )
                utils_module = importlib.util.module_from_spec(utils_spec)
                utils_spec.loader.exec_module(utils_module)
                
                sys.modules['utils'] = utils_module
                print("✅ AppAgent utils模块预加载成功")
                
                config_spec = importlib.util.spec_from_file_location(
                    "appagent_config", 
                    os.path.join(appagent_scripts_dir, "config.py")
                )
                config_module = importlib.util.module_from_spec(config_spec)
                config_spec.loader.exec_module(config_module)
                sys.modules['config'] = config_module
                
                spec = importlib.util.spec_from_file_location("appagent_task_executor", appagent_path)
                self.appagent_module = importlib.util.module_from_spec(spec)
                
                spec.loader.exec_module(self.appagent_module)
                
                if not hasattr(self.appagent_module, 'run_appagent_task'):
                    raise AttributeError("AppAgent模块中未找到run_appagent_task函数")
                
                self._module_loaded = True
                
                return self.appagent_module
                
            finally:
                os.chdir(original_cwd)
                print(f"📂 工作目录恢复到: {original_cwd}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ImportError(f"无法加载AppAgent模块: {e}")

    def select_hook_function(self, *args, **kwargs) -> bool:
        return self._capture_step_screenshot()

    async def simulation(self, *args, **kwargs) -> List[str]:
       
        simulation_result = await self.appagent_module.run_appagent_simulation(
            *args, **kwargs
        )
        
        return simulation_result
            
      

    def execute_task(self, task_instruction: str, test_case: Optional[str] = None, app: str = "shopping", 
                     resume_from_checkpoint: Optional[str] = None) -> AgentExecutionResult:
       
        start_time = time.time()
        self.app = app
        self.test_case = test_case

        self.log_root = self.config.log_dir

        self._setup_screenshot_capture(test_case)
        
        if not resume_from_checkpoint:
            reset(app, test_case)

        try:
            if not self._module_loaded or self.appagent_module is None:
                raise RuntimeError("AppAgent模块未正确加载，请先调用initialize方法")
            
            result = self.appagent_module.run_appagent_task(
                task_description=task_instruction,
                app=app,
                max_rounds=self.config.max_iterations or 40,
                select_hook_function=self.select_hook_function,
                log_path=self.log_root,
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                success=result.get('success', True),
                selected_product_id=result.get('selected_product_id', ""),
                execution_time=execution_time,
                steps_taken=result.get('steps_taken', 0),
                instruction=task_instruction,
                test_case=test_case or ""
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            import traceback
            traceback.print_exc()
            
            return AgentExecutionResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                instruction=task_instruction,
                test_case=test_case or ""
            )
    
    def list_checkpoints(self, task_dir: Optional[str] = None) -> List[str]:
        
        try:
            if not self._module_loaded or self.appagent_module is None:
                raise RuntimeError("AppAgent模块未正确加载，请先调用initialize方法")
            
            if task_dir is None:
                task_dir = self.log_root
                
            checkpoint_dir = os.path.join(task_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                return []
                
            return self.appagent_module.list_checkpoints(checkpoint_dir)
            
        except Exception as e:
            print(f"❌ 列出checkpoints失败: {e}")
            return []
    
    def get_latest_checkpoint(self, task_dir: Optional[str] = None) -> Optional[str]:
        
        checkpoints = self.list_checkpoints(task_dir)
        if checkpoints:
            return checkpoints[-1]  # list_checkpoints返回的是按轮次排序的
        return None

    def execute_task_with_resume(self, task_instruction: str, test_case: Optional[str] = None, 
                                app: str = "shopping", auto_resume: bool = True) -> AgentExecutionResult:
       
        resume_checkpoint = None
        
        if auto_resume:
            # 尝试找到最新的checkpoint
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                print(f"🔍 发现最新checkpoint: {latest_checkpoint}")
                resume_checkpoint = latest_checkpoint
            
        return self.execute_task(
            task_instruction=task_instruction,
            test_case=test_case,
            app=app,
            resume_from_checkpoint=resume_checkpoint
        )

    def cleanup(self):
        try:
            if self.appagent_module:
                # 清理AppAgent模块相关资源
                self.appagent_module = None
                self._module_loaded = False
                print("🧹 AppAgent资源清理完成")
        except Exception as e:
            print(f"❌ AppAgent资源清理失败: {e}")
        finally:
            super().cleanup()


class MobileAgentFactory:
    """Mobile Agent工厂类"""
    
    _agents = {
        MobileAgentType.MOBILE_AGENT_E.value: MobileAgentEWorkflow,
        MobileAgentType.MOBILE_AGENT_V2.value: MobileAgentV2Workflow,
        MobileAgentType.APPAGENT.value: AppAgentWorkflow,
    }
    
    @classmethod
    def create_agent(cls, agent_type: MobileAgentType) -> BaseMobileAgent:
        if agent_type in cls._agents:
            return cls._agents[agent_type]()
        else:
            raise ValueError(f"不支持的Agent类型: {agent_type}")
    
    @classmethod
    def get_supported_agents(cls) -> List[MobileAgentType]:
        return list(cls._agents.keys())


class WorkflowManager:
    
    def __init__(self):
        self.current_agent: MobileAgentV2Workflow = None
        self.execution_history = []
    
    def initialize_agent(self, agent_type: MobileAgentType, config: EvaluationConfig) -> bool:
        try:
            self.current_agent = MobileAgentFactory.create_agent(agent_type)
            success = self.current_agent.initialize(config)
            
         
            return success
            
        except Exception as e:
            self.current_agent = None
            return False
    
    async def simulation(self,*args,**kwargs) -> List[str]:
        return await self.current_agent.simulation(*args,**kwargs)
    
    def  execute_task(self, task_instruction: str, test_case: Optional[str] = None, app:str = "shopping",
                      attack_setting: AttackConfig = None
                      ) -> AgentExecutionResult:
        if not self.current_agent:
            return AgentExecutionResult(
                success=False,
                error_message="Agent未初始化"
            )
        
        self.current_agent.attack_setting = attack_setting
        result = self.current_agent.execute_task(task_instruction, test_case,app)

        self.execution_history.append(result)
        
        return result
    

    def  eval_selection(self, task_instruction: str, test_case: Optional[str] = None, app:str = "shopping",
                      attack_setting: AttackConfig = None
                      ) -> AgentExecutionResult:
       
        self.current_agent.attack_setting = attack_setting
        result = self.current_agent.execute_task(task_instruction, test_case,app)

        self.execution_history.append(result)
        
        return result
    
    def cleanup(self):
        if self.current_agent:
            self.current_agent.cleanup()
            self.current_agent = None
    
    def get_execution_history(self) -> List[AgentExecutionResult]:
        return self.execution_history.copy()
    
    def clear_history(self):
        self.execution_history.clear()
