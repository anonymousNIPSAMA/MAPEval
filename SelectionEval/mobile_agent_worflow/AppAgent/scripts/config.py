import os
import yaml


def load_config(config_path=None):
    configs = dict(os.environ)
    
    # 如果没有指定config_path，则自动寻找配置文件
    if config_path is None:
        # 首先尝试当前目录
        if os.path.exists("./config.yaml"):
            config_path = "./config.yaml"
        else:
            # 如果当前目录没有，尝试从脚本所在目录向上查找
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)  # AppAgent根目录
            parent_config = os.path.join(parent_dir, "config.yaml")
            if os.path.exists(parent_config):
                config_path = parent_config
            else:
                # 如果都找不到，使用默认路径
                config_path = "./config.yaml"
    
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs
