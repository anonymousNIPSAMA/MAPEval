import argparse
import ast
import datetime
import json
import os
import re
import sys
import time
import asyncio
from typing import List

import prompts
from config import load_config
from and_controller import list_all_devices, AndroidController, traverse_tree
from model import parse_explore_rsp, parse_grid_rsp, OpenAIModel, QwenModel
from utils import print_with_color, draw_bbox_multi, draw_grid


def save_checkpoint(task_description, configs, app, round_count, last_act, task_complete, 
                   grid_on, rows, cols, width, height, device, task_dir, log_file_path,
                   no_doc, docs_dir, checkpoint_dir=None):
    """
    保存AppAgent执行状态的checkpoint
    
    Args:
        task_description: 任务描述
        configs: 配置信息
        app: 应用名称
        round_count: 当前轮次
        last_act: 上一个动作
        task_complete: 任务是否完成
        grid_on: 是否开启网格模式
        rows, cols: 网格行列数
        width, height: 屏幕尺寸
        device: 设备ID
        task_dir: 任务目录
        log_file_path: 日志文件路径
        no_doc: 是否无文档
        docs_dir: 文档目录
        checkpoint_dir: checkpoint保存目录
    """
    # checkpoint 直接保存在任务目录下，不在子文件夹中
    if checkpoint_dir is None:
        checkpoint_dir = task_dir
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建状态字典
    state_data = {
        'task_description': task_description,
        'configs': configs,
        'app': app,
        'round_count': round_count,
        'last_act': last_act,
        'task_complete': task_complete,
        'grid_on': grid_on,
        'rows': rows,
        'cols': cols,
        'width': width,
        'height': height,
        'device': device,
        'task_dir': task_dir,
        'log_file_path': log_file_path,
        'no_doc': no_doc,
        'docs_dir': docs_dir,
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint_version': '1.0'
    }
    
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{round_count}.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, ensure_ascii=False, indent=4)
    
    print_with_color(f"Checkpoint saved at {checkpoint_file}", "green")
    return checkpoint_file


def load_checkpoint(checkpoint_file):
    """
    从checkpoint文件加载AppAgent执行状态
    
    Args:
        checkpoint_file: checkpoint文件路径
        
    Returns:
        dict: 包含所有状态信息的字典
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_file}")
    
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    
    print_with_color(f"Checkpoint loaded from {checkpoint_file}", "green")
    return state_data


def list_checkpoints(checkpoint_dir):
    """
    列出指定目录下的所有checkpoint文件
    
    Args:
        checkpoint_dir: checkpoint目录
        
    Returns:
        list: checkpoint文件列表，按轮次排序
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_iter_") and filename.endswith(".json"):
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    # 按轮次排序
    checkpoint_files.sort(key=lambda x: int(re.search(r'checkpoint_iter_(\d+)\.json', x).group(1)))
    return checkpoint_files


def run_appagent_task(task_description: str, app: str = "foodpanda", max_rounds: int = None, 
                      select_hook_function=None, log_path: str = None, 
                      resume_from_checkpoint: str = None) -> dict:
    """
    统一的AppAgent任务执行接口
    
    Args:
        task_description: 任务描述
        app: 应用名称
        max_rounds: 最大轮数
        select_hook_function: 选择钩子函数
        log_path: 日志路径
        resume_from_checkpoint: 要恢复的checkpoint文件路径
        
    Returns:
        dict: 包含执行结果的字典
    """
    try:
        # 如果指定了从checkpoint恢复
        if resume_from_checkpoint:
            print_with_color(f"正在从checkpoint恢复: {resume_from_checkpoint}", "yellow")
            state_data = load_checkpoint(resume_from_checkpoint)
            
            # 从checkpoint恢复状态
            task_description = state_data['task_description']
            app = state_data['app']
            configs = state_data['configs']
            round_count = state_data['round_count']
            last_act = state_data['last_act']
            task_complete = state_data['task_complete']
            grid_on = state_data['grid_on']
            rows = state_data['rows']
            cols = state_data['cols']
            width = state_data['width']
            height = state_data['height']
            device = state_data['device']
            task_dir = state_data['task_dir']
            log_file_path = state_data['log_file_path']
            no_doc = state_data['no_doc']
            docs_dir = state_data['docs_dir']
            
            print_with_color(f"从轮次 {round_count} 恢复执行", "green")
            
            # 重新初始化必要的组件
            controller = AndroidController(device)
            
            # 初始化模型
            if configs["MODEL"] == "OpenAI":
                mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                                   api_key=configs["OPENAI_API_KEY"],
                                   model=configs["OPENAI_API_MODEL"],
                                   temperature=configs["TEMPERATURE"],
                                   max_tokens=configs["MAX_TOKENS"])
            elif configs["MODEL"] == "Qwen":
                mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                                 model=configs["QWEN_MODEL"])
            else:
                raise ValueError(f"不支持的模型类型: {configs['MODEL']}")
                
        else:
            # 正常初始化流程
            # 重新加载配置
            configs = load_config()
            
            # 设置最大轮数
            if max_rounds:
                configs["MAX_ROUNDS"] = max_rounds
            
            # 初始化模型
            if configs["MODEL"] == "OpenAI":
                mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                                   api_key=configs["OPENAI_API_KEY"],
                                   model=configs["OPENAI_API_MODEL"],
                                   temperature=configs["TEMPERATURE"],
                                   max_tokens=configs["MAX_TOKENS"])
            elif configs["MODEL"] == "Qwen":
                mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                                 model=configs["QWEN_MODEL"])
            else:
                raise ValueError(f"不支持的模型类型: {configs['MODEL']}")
            
            # 设置应用和目录
            root_dir = "./"
            app_dir = os.path.join(os.path.join(root_dir, "apps"), app)
            work_dir = os.path.join(root_dir, "tasks")
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)
            
            auto_docs_dir = os.path.join(app_dir, "auto_docs")
            demo_docs_dir = os.path.join(app_dir, "demo_docs")
            task_timestamp = int(time.time())
            dir_name = datetime.datetime.fromtimestamp(task_timestamp).strftime(f"task_{app}_%Y-%m-%d_%H-%M-%S")
            task_dir = os.path.join(work_dir, dir_name)
            
            # 如果指定了log_path，使用指定的路径
            if log_path:
                task_dir = log_path
                os.makedirs(task_dir, exist_ok=True)
            else:
                os.mkdir(task_dir)
                
            # 创建temp子文件夹存放日志和XML文件
            temp_dir = os.path.join(task_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            log_file_path = os.path.join(temp_dir, f"log_{app}_{dir_name}.txt")

            # 检查文档
            no_doc = False
            if not os.path.exists(auto_docs_dir) and not os.path.exists(demo_docs_dir):
                no_doc = True
                docs_dir = None
            elif os.path.exists(auto_docs_dir):
                docs_dir = auto_docs_dir
            else:
                docs_dir = demo_docs_dir

            # 获取设备
            device_list = list_all_devices()
            if not device_list:
                raise RuntimeError("没有找到设备")
            
            device = device_list[0]  # 使用第一个设备
            controller = AndroidController(device)
            width, height = controller.get_device_size()
            if not width and not height:
                raise RuntimeError("无效的设备尺寸")

            # 初始化执行状态
            round_count = 0
            last_act = "None"
            task_complete = False
            grid_on = False
            rows, cols = 0, 0

        def area_to_xy(area, subarea):
            area -= 1
            row, col = area // cols, area % cols
            x_0, y_0 = col * (width // cols), row * (height // rows)
            if subarea == "top-left":
                x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 4
            elif subarea == "top":
                x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 4
            elif subarea == "top-right":
                x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 4
            elif subarea == "left":
                x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 2
            elif subarea == "right":
                x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 2
            elif subarea == "bottom-left":
                x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) * 3 // 4
            elif subarea == "bottom":
                x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) * 3 // 4
            elif subarea == "bottom-right":
                x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) * 3 // 4
            else:
                x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 2
            return x, y

        while round_count < configs["MAX_ROUNDS"]:
            round_count += 1
            print_with_color(f"Round {round_count}", "yellow")
            
            # 保存checkpoint
            try:
                save_checkpoint(
                    task_description, configs, app, round_count, last_act, task_complete,
                    grid_on, rows, cols, width, height, device, task_dir, log_file_path,
                    no_doc, docs_dir
                )
            except Exception as e:
                print_with_color(f"保存checkpoint失败: {e}", "red")
            
            # 调用选择钩子函数
            if select_hook_function:
                try:
                    select_hook_function()
                except Exception as e:
                    print_with_color(f"选择钩子函数执行失败: {e}", "red")
            
            # 使用temp目录存放截图和XML文件
            temp_dir = os.path.join(task_dir, "temp")
            screenshot_path = controller.get_screenshot(f"{dir_name}_{round_count}", temp_dir)
            xml_path = controller.get_xml(f"{dir_name}_{round_count}", temp_dir)
            if screenshot_path == "ERROR" or xml_path == "ERROR":
                break
                
            if grid_on:
                rows, cols = draw_grid(screenshot_path, os.path.join(temp_dir, f"{dir_name}_{round_count}_grid.png"))
                image = os.path.join(temp_dir, f"{dir_name}_{round_count}_grid.png")
                prompt = prompts.task_template_grid
            else:
                clickable_list = []
                focusable_list = []
                traverse_tree(xml_path, clickable_list, "clickable", True)
                traverse_tree(xml_path, focusable_list, "focusable", True)
                elem_list = clickable_list.copy()
                for elem in focusable_list:
                    bbox = elem.bbox
                    center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                    close = False
                    for e in clickable_list:
                        bbox = e.bbox
                        center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                        dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
                        if dist <= configs["MIN_DIST"]:
                            close = True
                            break
                    if not close:
                        elem_list.append(elem)
                draw_bbox_multi(screenshot_path, os.path.join(temp_dir, f"{dir_name}_{round_count}_labeled.png"), elem_list,
                                dark_mode=configs["DARK_MODE"])
                image = os.path.join(temp_dir, f"{dir_name}_{round_count}_labeled.png")
                if no_doc:
                    prompt = re.sub(r"<ui_document>", "", prompts.task_template)
                else:
                    ui_doc = ""
                    for i, elem in enumerate(elem_list):
                        doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")
                        if not os.path.exists(doc_path):
                            continue
                        ui_doc += f"Documentation of UI element labeled with the numeric tag '{i + 1}':\n"
                        doc_content = ast.literal_eval(open(doc_path, "r").read())
                        if doc_content["tap"]:
                            ui_doc += f"This UI element is clickable. {doc_content['tap']}\n\n"
                        if doc_content["text"]:
                            ui_doc += f"This UI element can receive text input. The text input is used for the following " \
                                      f"purposes: {doc_content['text']}\n\n"
                        if doc_content["long_press"]:
                            ui_doc += f"This UI element is long clickable. {doc_content['long_press']}\n\n"
                        if doc_content["v_swipe"]:
                            ui_doc += f"This element can be swiped directly without tapping. You can swipe vertically on " \
                                      f"this UI element. {doc_content['v_swipe']}\n\n"
                        if doc_content["h_swipe"]:
                            ui_doc += f"This element can be swiped directly without tapping. You can swipe horizontally on " \
                                      f"this UI element. {doc_content['h_swipe']}\n\n"
                    print_with_color(f"Documentations retrieved for the current interface:\n{ui_doc}", "magenta")
                    ui_doc = """
                    You also have access to the following documentations that describes the functionalities of UI 
                    elements you can interact on the screen. These docs are crucial for you to determine the target of your 
                    next action. You should always prioritize these documented elements for interaction:""" + ui_doc
                    prompt = re.sub(r"<ui_document>", ui_doc, prompts.task_template)
            
            prompt = re.sub(r"<task_description>", task_description, prompt)
            prompt = re.sub(r"<last_act>", last_act, prompt)
            print_with_color("Thinking about what to do in the next step...", "yellow")
            status, rsp = mllm.get_model_response(prompt, [image])

            if status:
                with open(log_file_path, "a") as logfile:
                    log_item = {"step": round_count, "prompt": prompt, "image": f"{dir_name}_{round_count}_labeled.png",
                                "response": rsp}
                    logfile.write(json.dumps(log_item) + "\n")
                if grid_on:
                    res = parse_grid_rsp(rsp)
                else:
                    res = parse_explore_rsp(rsp)
                act_name = res[0]
                if act_name == "FINISH":
                    task_complete = True
                    break
                if act_name == "ERROR":
                    break
                last_act = res[-1]
                res = res[:-1]
                if act_name == "tap":
                    _, area = res
                    tl, br = elem_list[area - 1].bbox
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.tap(x, y)
                    if ret == "ERROR":
                        print_with_color("ERROR: tap execution failed", "red")
                        break
                elif act_name == "text":
                    _, input_str = res
                    ret = controller.text(input_str)
                    if ret == "ERROR":
                        print_with_color("ERROR: text execution failed", "red")
                        break
                elif act_name == "long_press":
                    _, area = res
                    tl, br = elem_list[area - 1].bbox
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.long_press(x, y)
                    if ret == "ERROR":
                        print_with_color("ERROR: long press execution failed", "red")
                        break
                elif act_name == "swipe":
                    _, area, swipe_dir, dist = res
                    tl, br = elem_list[area - 1].bbox
                    x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
                    ret = controller.swipe(x, y, swipe_dir, dist)
                    if ret == "ERROR":
                        print_with_color("ERROR: swipe execution failed", "red")
                        break
                elif act_name == "grid":
                    grid_on = True
                elif act_name == "tap_grid" or act_name == "long_press_grid":
                    _, area, subarea = res
                    x, y = area_to_xy(area, subarea)
                    if act_name == "tap_grid":
                        ret = controller.tap(x, y)
                        if ret == "ERROR":
                            print_with_color("ERROR: tap execution failed", "red")
                            break
                    else:
                        ret = controller.long_press(x, y)
                        if ret == "ERROR":
                            print_with_color("ERROR: tap execution failed", "red")
                            break
                elif act_name == "swipe_grid":
                    _, start_area, start_subarea, end_area, end_subarea = res
                    start_x, start_y = area_to_xy(start_area, start_subarea)
                    end_x, end_y = area_to_xy(end_area, end_subarea)
                    ret = controller.swipe_precise((start_x, start_y), (end_x, end_y))
                    if ret == "ERROR":
                        print_with_color("ERROR: tap execution failed", "red")
                        break
                if act_name != "grid":
                    grid_on = False
                time.sleep(configs["REQUEST_INTERVAL"])
            else:
                print_with_color(rsp, "red")
                break

        # 返回执行结果
        if task_complete:
            print_with_color("Task completed successfully", "yellow")
            return {
                'success': True,
                'steps_taken': round_count,
                'selected_product_id': "",  # AppAgent可能需要额外逻辑来提取选择的产品ID
                'task_complete': True
            }
        elif round_count == configs["MAX_ROUNDS"]:
            print_with_color("Task finished due to reaching max rounds", "yellow")
            return {
                'success': False,
                'steps_taken': round_count,
                'selected_product_id': "",
                'task_complete': False,
                'error_message': "达到最大轮数限制"
            }
        else:
            print_with_color("Task finished unexpectedly", "red")
            return {
                'success': False,
                'steps_taken': round_count,
                'selected_product_id': "",
                'task_complete': False,
                'error_message': "任务意外结束"
            }
            
    except Exception as e:
        print_with_color(f"AppAgent任务执行失败: {e}", "red")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'steps_taken': 0,
            'selected_product_id': "",
            'task_complete': False,
            'error_message': str(e)
        }


async def run_appagent_simulation(state_file: str, screenshot_file: str = None, 
                                 repeat_num: int = 1, temperature: float = 0.2,
                                 model_name="",temp_file="",
                                 ) -> List[str]:
    """
    基于checkpoint状态文件模拟AppAgent的执行过程，支持并发批量推理
    
    Args:
        state_file: checkpoint状态文件路径
        screenshot_file: 新的截图文件路径（可选，如果不提供则使用状态文件中的截图）
        repeat_num: 重复模拟次数
        temperature: 模型温度参数
        
    Returns:
        List[str]: 所有无error的完整模拟响应
    """
    try:
        # 加载checkpoint状态
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"状态文件不存在: {state_file}")
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        print_with_color(f"加载checkpoint状态: {state_file}", "green")
        
        # 从状态中恢复关键信息
        task_description = state_data.get('task_description', '')
        configs = state_data.get('configs', {})
        app = state_data.get('app', 'foodpanda')
        round_count = state_data.get('round_count', 0)
        last_act = state_data.get('last_act', 'None')
        task_complete = state_data.get('task_complete', False)
        grid_on = state_data.get('grid_on', False)
        rows = state_data.get('rows', 0)
        cols = state_data.get('cols', 0)
        width = state_data.get('width', 1080)
        height = state_data.get('height', 1920)
        device = state_data.get('device', '')
        task_dir = state_data.get('task_dir', '')
        no_doc = state_data.get('no_doc', False)
        docs_dir = state_data.get('docs_dir', None)
        
        # 确定截图文件 - 如果没有提供新截图，尝试获取当前截图
        if screenshot_file is None:
            temp_dir = os.path.join(task_dir, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            # 生成新的截图文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_file = os.path.join(temp_dir, f"simulation_screenshot_{timestamp}.png")
            
            # 如果有设备连接，尝试截图
            if device:
                try:
                    controller = AndroidController(device)
                    controller.get_screenshot(f"simulation_{timestamp}", temp_dir)
                except Exception as e:
                    print_with_color(f"获取截图失败，无法进行准确模拟: {e}", "yellow")
                    # 没有截图就无法进行准确的模拟
                    return [f"simulation_error: 无法获取截图进行模拟"]
        

        base_url = configs.get("OPENAI_API_BASE", "")
        if "qwen" in model_name:
            base_url = "http://localhost:9999/v1/chat/completions"
            
        mllm = OpenAIModel(
                base_url=base_url,
                api_key=configs.get("OPENAI_API_KEY", ""),
                model=model_name,
                temperature=temperature,
                max_tokens=configs.get("MAX_TOKENS", 1000)
            )
        
        # 验证截图文件是否存在
        if not os.path.exists(screenshot_file):
            print_with_color(f"截图文件不存在: {screenshot_file}", "red")
            return [f"simulation_error: 截图文件不存在"]
        
        # 准备prompt（与run_appagent_task完全相同的逻辑）
        image = screenshot_file
        prompt = re.sub(r"<ui_document>", "", prompts.task_template)
        prompt = re.sub(r"<task_description>", task_description, prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)
                
        # 定义单次模拟任务
        async def single_simulation(sim_index: int):
                
            status, rsp = await mllm.get_model_response_async(prompt, [image])
            return {
                            'success': True,
                            'index': sim_index,
                            'response': rsp,
                            'action': "predicted_action",
                            'round': round_count + 1
                        }
                
               
        
        # 并发执行所有模拟任务
        tasks = [single_simulation(i) for i in range(repeat_num)]
        simulation_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤并返回所有无error的完整响应
        successful_responses = []
        for result in simulation_results:
            if isinstance(result, dict) and result.get('success', False):
                # 只返回成功的完整响应
                successful_responses.append(result['response'])
            
        return successful_responses
        
    except Exception as e:
        print_with_color(f"AppAgent模拟执行失败: {e}", "red")
        import traceback
        traceback.print_exc()
        return []



if __name__ == "__main__":
    # 原有的命令行交互式执行逻辑
    arg_desc = "AppAgent Executor"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
    parser.add_argument("--app", default="foodpanda", help="应用名称")
    parser.add_argument("--root_dir", default="./")
    args = vars(parser.parse_args())

    configs = load_config()

    if configs["MODEL"] == "OpenAI":
        mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                           api_key=configs["OPENAI_API_KEY"],
                           model=configs["OPENAI_API_MODEL"],
                           temperature=configs["TEMPERATURE"],
                           max_tokens=configs["MAX_TOKENS"])
    elif configs["MODEL"] == "Qwen":
        mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                         model=configs["QWEN_MODEL"])
    else:
        print_with_color(f"ERROR: Unsupported model type {configs['MODEL']}!", "red")
        sys.exit()

    app = args["app"]
    root_dir = args["root_dir"]

    if not app:
        print_with_color("What is the name of the app you want me to operate?", "blue")
        app = input()
        app = app.replace(" ", "")

    app_dir = os.path.join(os.path.join(root_dir, "apps"), app)
    work_dir = os.path.join(root_dir, "tasks")
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    auto_docs_dir = os.path.join(app_dir, "auto_docs")
    demo_docs_dir = os.path.join(app_dir, "demo_docs")
    task_timestamp = int(time.time())
    dir_name = datetime.datetime.fromtimestamp(task_timestamp).strftime(f"task_{app}_%Y-%m-%d_%H-%M-%S")
    task_dir = os.path.join(work_dir, dir_name)
    os.mkdir(task_dir)
    log_path = os.path.join(task_dir, f"log_{app}_{dir_name}.txt")

    no_doc = False
    if not os.path.exists(auto_docs_dir) and not os.path.exists(demo_docs_dir):
        print_with_color(f"No documentations found for the app {app}. Do you want to proceed with no docs? Enter y or n",
                         "red")
        user_input = ""
        while user_input != "y" and user_input != "n":
            user_input = input().lower()
        if user_input == "y":
            no_doc = True
        else:
            sys.exit()
    elif os.path.exists(auto_docs_dir) and os.path.exists(demo_docs_dir):
        print_with_color(f"The app {app} has documentations generated from both autonomous exploration and human "
                         f"demonstration. Which one do you want to use? Type 1 or 2.\n1. Autonomous exploration\n2. Human "
                         f"Demonstration",
                         "blue")
        user_input = ""
        while user_input != "1" and user_input != "2":
            user_input = input()
        if user_input == "1":
            docs_dir = auto_docs_dir
        else:
            docs_dir = demo_docs_dir
    elif os.path.exists(auto_docs_dir):
        print_with_color(f"Documentations generated from autonomous exploration were found for the app {app}. The doc base "
                         f"is selected automatically.", "yellow")
        docs_dir = auto_docs_dir
    else:
        print_with_color(f"Documentations generated from human demonstration were found for the app {app}. The doc base is "
                         f"selected automatically.", "yellow")
        docs_dir = demo_docs_dir

    device_list = list_all_devices()
    if not device_list:
        print_with_color("ERROR: No device found!", "red")
        sys.exit()
    print_with_color(f"List of devices attached:\n{str(device_list)}", "yellow")
    if len(device_list) == 1:
        device = device_list[0]
        print_with_color(f"Device selected: {device}", "yellow")
    else:
        print_with_color("Please choose the Android device to start demo by entering its ID:", "blue")
        device = input()
    controller = AndroidController(device)
    width, height = controller.get_device_size()
    if not width and not height:
        print_with_color("ERROR: Invalid device size!", "red")
        sys.exit()
    print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

    print_with_color("Please enter the description of the task you want me to complete in a few sentences:", "blue")
    task_desc = input()

    # 使用新的统一接口执行任务
    result = run_appagent_task(task_desc, app)
    
    if result['success']:
        print_with_color("任务执行成功!", "green")
    else:
        print_with_color(f"任务执行失败: {result.get('error_message', '未知错误')}", "red")
