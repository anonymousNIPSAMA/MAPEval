import asyncio
import os
import time
import copy
import torch
import shutil
import json
from datetime import datetime
from PIL import Image, ImageDraw
import tqdm

from MobileAgent.api import cache, inference_achat, inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot, tap, slide, type, back, home
from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent

####################################### Edit your Setting #########################################
# Your ADB path
adb_path = ""
adb_path = "/usr/bin/adb -s 192.168.3.38:5555"

# Your instruction
instruction = ""

# Your GPT-4o API URL
API_url = ""
API_url = ""

# Your GPT-4o API Token
token = ""
token= ""

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint
caption_call_method = "api"

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
caption_model = "qwen-vl-plus"

# If you choose the api caption call method, input your Qwen api here
qwen_api = ""
qwen_api = ""

# You can add operational knowledge to help Agent operate more accurately.
add_info = "The app is already opened."

# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
reflection_switch = True

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = True
###################################################################################################


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i,temp_file):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"{temp_file}/{i}.jpg")


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response

@cache
def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image
            },
            {
                'text': query
            },
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)
    
    try:
        response = response['output']['choices'][0]['message']['content'][0]["text"]
    except:
        response = "This is an icon."
    
    return response


def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


def merge_text_blocks(text_list, coordinates_list):
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(range(len(coordinates_list)), key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue
        
        anchor = i
        
        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i+1, num_blocks):
            if merge[j]:
                continue

            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
            sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
            abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates


def get_perception_infos(adb_path, screenshot_file, temp_file):
    # get_screenshot(adb_path)

            
    if not os.path.exists(temp_file):
        os.mkdir(temp_file)
    
    width, height = Image.open(screenshot_file).size
    
    text, coordinates = ocr(screenshot_file, ocr_detection, ocr_recognition)
    text, coordinates = merge_text_blocks(text, coordinates)
    
    center_list = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coordinates]
    draw_coordinates_on_image(screenshot_file, center_list)
    
    perception_infos = []
    for i in range(len(coordinates)):
        perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
        perception_infos.append(perception_info)
        
    coordinates = det(screenshot_file, "icon", groundingdino_model)
    
    for i in range(len(coordinates)):
        perception_info = {"text": "icon", "coordinates": coordinates[i]}
        perception_infos.append(perception_info)
        
    image_box = []
    image_id = []
    for i in range(len(perception_infos)):
        if perception_infos[i]['text'] == 'icon':
            image_box.append(perception_infos[i]['coordinates'])
            image_id.append(i)

    for i in range(len(image_box)):
        crop(screenshot_file, image_box[i], image_id[i],temp_file)

    images = get_all_files_in_folder(temp_file)
    if len(images) > 0:
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
        icon_map = {}
        prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
        if caption_call_method == "local":
            for i in range(len(images)):
                image_path = os.path.join(temp_file, images[i])
                icon_width, icon_height = Image.open(image_path).size
                if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                    des = "None"
                else:
                    des = generate_local(tokenizer, model, image_path, prompt)
                icon_map[i+1] = des
        else:
            for i in range(len(images)):
                images[i] = os.path.join(temp_file, images[i])
            icon_map = generate_api(images, prompt)
        for i, j in zip(image_id, range(1, len(image_id)+1)):
            if icon_map.get(j):
                perception_infos[i]['text'] = "icon: " + icon_map[j]

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
    
    #删除temp 文件夹
    if os.path.exists(temp_file):
        shutil.rmtree(temp_file)

    return perception_infos, width, height

### Load caption model ###
device = "cuda"
torch.manual_seed(1234)
if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    elif caption_model == "qwen-vl-chat-int4":
        qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
        model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True,use_safetensors=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
    else:
        print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
elif caption_call_method == "api":
    pass
else:
    print("You must choose the caption model call function from \"local\" and \"api\"")
    exit(0)

### Load ocr and icon detection model ###
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')


def generate_state_file(state_file,image_file):
    pass
    


async def run_v2_simulation(state_file,screenshot_file=None,repeat_num=1,temperature=0.2,model_name="gpt-4o",temp_file="temp"):
    
    if not os.path.exists(temp_file):
        os.mkdir(temp_file)
        
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    if not screenshot_file:
        screenshot_file = state['screenshot_file']
        perception_infos = state['perception_infos']
        width = state['width']
        height = state['height']

    else:
        perception_infos, width, height = get_perception_infos(adb_path, screenshot_file, temp_file)


    instruction = state['instruction']
    keyboard = state['keyboard']
    summary_history = state['summary_history']
    action_history = state['action_history']
    summary = state['summary']
    action = state['action']
    add_info = state['add_info']
    error_flag = state['error_flag']
    completed_requirements = state['completed_requirements']
    memory = state['memory']

    prompt_action = get_action_prompt(instruction, perception_infos, width, height, keyboard, summary_history, action_history, summary, action, add_info, error_flag, completed_requirements, memory)
    chat_action = init_action_chat()
    chat_action = add_response("user", prompt_action, chat_action, screenshot_file)

    res = []
    continues = []
    for i in (range(repeat_num)):
        continues.append(inference_achat(chat_action, model_name, API_url, token,temperature=temperature))

    res = await asyncio.gather(*continues)

    return res


def run_v2_workflow(instruction,max_iterations=20, select_hook_function = None,
                    log_path = "logs/",
                    ):
    """Mobile-Agent-V2 主执行函数"""
    thought_history = []
    summary_history = []
    action_history = []
    summary = ""
    action = ""
    completed_requirements = ""
    memory = ""
    insight = ""
    screenshot = "screenshot"
    
    temp_file="./temp"
    
    if not os.path.exists(temp_file):
        os.mkdir(temp_file)
    else:
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)
    if not os.path.exists(screenshot):
        os.mkdir(screenshot)
    error_flag = False


    iter = 0
    while iter < max_iterations:
        iter += 1
        if iter == 1:
            screenshot_file = "./screenshot/screenshot.jpg"

            
            select_hook_function and select_hook_function()
            perception_infos, width, height = get_perception_infos(adb_path, screenshot_file, temp_file)
            if os.path.exists(temp_file):
                shutil.rmtree(temp_file)
            os.mkdir(temp_file)
            
            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    keyboard = True
                    break

        prompt_action = get_action_prompt(instruction, perception_infos, width, height, keyboard, summary_history, action_history, summary, action, add_info, error_flag, completed_requirements, memory)
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, screenshot_file)
        # print(chat_action)

        save_checkpoint(instruction, perception_infos, width, height, keyboard, summary_history, action_history, summary, action, add_info, error_flag, completed_requirements, memory, screenshot_file, iter=iter,log_path=log_path)
        output_action = inference_chat(chat_action, 'gpt-4o', API_url, token)


        thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
        summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        chat_action = add_response("assistant", output_action, chat_action)
        status = "#" * 50 + " Decision " + "#" * 50
        print(status)
        print(output_action)
        print('#' * len(status))
        
        if memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(chat_action, 'gpt-4o', API_url, token)
            
            chat_action = add_response("assistant", output_memory, chat_action)

            status = "#" * 50 + " Memory " + "#" * 50
            print(status)
            print(output_memory)
            print('#' * len(status))
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory
        
        if "Open app" in action:
            app_name = action.split("(")[-1].split(")")[0]
            text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
            tap_coordinate = [0, 0]
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [int((coordinate[ti][0] + coordinate[ti][2])/2), int((coordinate[ti][1] + coordinate[ti][3])/2)]
                    tap(adb_path, name_coordinate[0], name_coordinate[1]- int(coordinate[ti][3] - coordinate[ti][1]))# 
                    break
        
        elif "Tap" in action:
            coordinate = action.split("(")[-1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])
            tap(adb_path, x, y)
        
        elif "Swipe" in action:
            coordinate1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
            coordinate2 = action.split("), (")[-1].split(")")[0].split(", ")
            x1, y1 = int(coordinate1[0]), int(coordinate1[1])
            x2, y2 = int(coordinate2[0]), int(coordinate2[1])
            slide(adb_path, x1, y1, x2, y2)
            
        elif "Type" in action:
            if "(text)" not in action:
                text = action.split("(")[-1].split(")")[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]
            type(adb_path, text)
        
        elif "Back" in action:
            back(adb_path)
        
        elif "Home" in action:
            home(adb_path)
            
        elif "Stop" in action:
            break
        
        time.sleep(5)
        
        last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)
        

        if select_hook_function:
            if select_hook_function():
                break
        perception_infos, width, height = get_perception_infos(adb_path, screenshot_file, temp_file)
        if os.path.exists(temp_file):
            shutil.rmtree(temp_file)
        os.mkdir(temp_file)
        
        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break
        
        if reflection_switch:
            prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, last_keyboard, keyboard, summary, action, add_info)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect, [last_screenshot_file, screenshot_file])


            output_reflect = inference_chat(chat_reflect, 'gpt-4o', API_url, token)
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflcetion " + "#" * 50
            print(status)
            print(output_reflect)
            print('#' * len(status))
        
            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)
                
                prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
                chat_planning = add_response("assistant", output_planning, chat_planning)
                status = "#" * 50 + " Planning " + "#" * 50
                print(status)
                print(output_planning)
                print('#' * len(status))
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
                
                error_flag = False
            
            elif 'B' in reflect:
                error_flag = True
                back(adb_path)
                
            elif 'C' in reflect:
                error_flag = True
        
        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
            
        os.remove(last_screenshot_file)

def save_checkpoint(*args, **kwargs):
    checkpoint_dir = kwargs.get('log_path', 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    iter = kwargs.get('iter', 1)
    kwargs.pop('iter', None)  # Remove 'iter' from kwargs to avoid saving it in the JSON
    
    # 创建状态字典，将位置参数和关键字参数分别保存
    state_data = {
        'instruction': args[0] if len(args) > 0 else None,
        'perception_infos': args[1] if len(args) > 1 else None,
        'width': args[2] if len(args) > 2 else None,
        'height': args[3] if len(args) > 3 else None,
        'keyboard': args[4] if len(args) > 4 else None,
        'summary_history': args[5] if len(args) > 5 else None,
        'action_history': args[6] if len(args) > 6 else None,
        'summary': args[7] if len(args) > 7 else None,
        'action': args[8] if len(args) > 8 else None,
        'add_info': args[9] if len(args) > 9 else None,
        'error_flag': args[10] if len(args) > 10 else None,
        'completed_requirements': args[11] if len(args) > 11 else None,
        'memory': args[12] if len(args) > 12 else None,
        'screenshot_file': args[13] if len(args) > 13 else None,
        'iter': iter,
        'timestamp': datetime.now().isoformat()
    }
    
    # 添加其他关键字参数
    state_data.update(kwargs)
    
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iter}.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, ensure_ascii=False, indent=4)
    
    print(f"Checkpoint saved at {checkpoint_file}")