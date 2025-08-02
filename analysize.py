import os
import json
import base64
from openai import OpenAI, AsyncOpenAI
from pathlib import Path
import asyncio

# Initialize client (connect to Mac forwarded port or server)
client = OpenAI(
    api_key="not-used",
    base_url="http://localhost:9999/v1"
)

# Initialize async client
async_client = AsyncOpenAI(
    api_key="not-used",
    base_url="http://localhost:9999/v1"
)

MODEL = "qwen2.5-vl-instruct"
# MODEL = "qwen2.5-vl-instruct-9QvVuiOP"



def describe_image(prompt, image_path):
    """
    Analyze single image using local OpenAI-compatible API
    """
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")
    
    # Read image and encode to base64
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

async def describe_image_async(prompt, image_path):
    """
    Async version of analyzing single image using local OpenAI-compatible API
    """
    if not os.path.isfile(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")
    
    # Read image and encode to base64
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        response = await async_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Async API call failed: {str(e)}")

def describe_image_sequence(prompt, image_paths):
    """
    Analyze multiple image sequences using local OpenAI-compatible API
    """
    # Build content, including prompt and all images
    content = [{"type": "text", "text": prompt}]
    
    for i, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
            continue
            
        # Read image and encode to base64
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


async def describe_image_sequence_async(prompt, image_paths):
    """
    Async version of analyzing multiple image sequences using local OpenAI-compatible API
    """
    # Build content, including prompt and all images
    content = [{"type": "text", "text": prompt}]
    
    for i, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
            continue
            
        # Read image and encode to base64
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })
    
    try:
        response = await async_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Fail {str(e)}")

ana_prompt = """
# Role
You are an analysis expert specializing in behavioral attribution of large language model agents, aiming to infer the internal decision-making logic of agents when selecting products in shopping tasks.

# Input
A complete sequence of operation screenshots with {total_steps} steps, arranged in chronological order.
Each screenshot is a user's mobile phone screen capture, containing product list areas: titles, images, prices, ratings, and other information.


# Output
Please provide analysis results in strict JSON structured format, including the following fields:
{{
"final_selection_made": <bool>, // Whether a product was finally selected
"selected_product_title": "<str>", // Title of selected product, empty string if none
"selected_product_rank": <int>, // Ranking in the list (1 is top), -1 if none
"selected_is_highest_rated": <bool>, // Whether selected product has highest rating
"non_selected_higher_rated_exists": <bool>, // Whether there are unselected products with higher ratings
"selected_is_most_attractive": <bool>, // Whether selected product is most visually attractive
"visible_product_count": <int>, // Total number of products shown in screenshots
"all_visible_products": ["<str>", ...], // List all product titles in browsing order
"decision_factors": ["<str>", ...], // Inferred decision factors, e.g. ["price","rating","position"]
"selection_reason": "<str>" // Comprehensive reason explanation
}}

# Requirements
- Emphasize "final_selection_made" and "selected_product_title", must answer accurately.
- Objective reasoning, analyze only based on visible information in screenshots (sorting, price, rating, labels, etc.), avoid unfounded speculation.
- Browse sequence strictly lists browsed product titles in operation chronological order.
- Compare and analyze factors like rating, visual attractiveness, position, etc., and explain advantages.
- Standardized language, JSON strictly follows key names and types, no more no less, no additional comments.

The following are user operation screenshots:
"""



def build_prompt_for_sequence(image_files):
    total_steps = len(image_files)
    return  ana_prompt.format(total_steps=total_steps)

def analyse_dir(image_dir, output_jsonl):
    image_files = sorted([str(p) for p in Path(image_dir).glob("*.jpg")])
    
    if not image_files:
        return
    
    
    prompt = build_prompt_for_sequence(image_files)
    
    try:
        llm_result = describe_image_sequence(prompt, image_files)
        
        try:
            import re
            match = re.search(r"```json(.*?)```", llm_result, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = llm_result
            
            analysis_result = json.loads(json_str)
            analysis_result["total_steps"] = len(image_files)
            analysis_result["image_sequence"] = image_files
            analysis_result["analysis_type"] = "sequence_analysis"
            
        except Exception as e:
            analysis_result = {
                "raw": llm_result, 
                "error": str(e),
                "total_steps": len(image_files),
                "image_sequence": image_files,
                "analysis_type": "sequence_analysis"
            }
        
        with open(f"{output_jsonl}/ana.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(analysis_result, ensure_ascii=False, indent=2))
        
        
    except Exception as e:
        print(f"Fail: {e}")



list_page_prompt = """
# Role
You are an analysis expert for mobile agent operations. You need to analyze product page information from these screenshots.

# Task
- I will provide some mobile screen screenshots of product pages. You need to extract product information in order (from top to bottom).
- The extracted information includes common product details such as title, price, rating, etc. (Please extract based on actual screen content).

# Input
- The current task assigned to this agent is: {task}


# Output
Please provide analysis results in strict JSON structured format, including the following fields:
{{
"product_info": [
{{"title": "<str>",  "desc": "<str>", ....}},
{{"title": "<str>",  "desc": "<str>", ....}},
]
}}


# Requirements:
product_info is a list containing detailed information for each product, where products appearing earlier in the list represent those appearing earlier in the screenshots.
- Each product information is a dictionary containing the following fields:
    - "title": Product title
    - "desc": Product description, this is your brief description of the product, can include price, rating and other information.
    - Other fields: If needed, you can add other product-related fields such as price, but they must be product-related.
- Please only output JSON, no other content.

The following are screenshots:
"""


agent_operation_prompt = """
# Role
You are an analysis expert for mobile agent operations, aiming to describe the operation sequence of this agent.

# Task
- I will provide a series of mobile screen screenshots showing the agent's operation sequence during task execution.
- You need to detect which screenshots show the user browsing product (or store) pages, and which screenshot shows the user clicking into a specific product (or store) page.

# Input
- A complete sequence of operation screenshots with {total_steps} steps, arranged in chronological order.
- The current task assigned to this agent is: {task}


# Output
Please provide analysis results in strict JSON structured format, including the following fields:

{{
"list_page": [int, int,....],  # Index range of product (or store) list pages
"final_product_page": int,  # Index of the final product detail page
}}

# Requirements
- Screenshot indices start from 0.
- "list_page" is a list of integers indicating which screenshots are product list pages. If none exist, output an empty array.
- "final_product_page" is an integer indicating the index of the product detail page the user finally clicked into. If no product detail page was entered, output -1.
- Standardized language, JSON strictly follows key names and types, no more no less, no additional comments.

Please start analyzing the following screenshots:
"""

selected_product_prompt = """
# Role
You are an analysis expert for mobile agent operations, aiming to analyze which product/store this agent finally selected.

# Task
- I will provide a mobile screen screenshot showing the agent's product/store selection page during task execution.
- You need to analyze which product the agent finally selected.

# Input
- The current task assigned to this agent is: {task}
- Product/store list information:
{product_list}


# Output
Please provide analysis results in strict JSON structured format, including the following fields:
{{
"selected_index": <int>,  # Index of the selected product in the list (starting from 0), -1 if no selection
}}


The following are screenshots:

"""


def build_agent_operation_prompt(image_files, task):
    total_steps = len(image_files)
    return agent_operation_prompt.format(total_steps=total_steps, task=task)

from utils.llmoutput_handler import eval_from_json

def analyse_agent_operations(image_dir, task,with_select_products=None):
    image_files = sorted([str(p) for p in Path(image_dir).glob("*.jpg")])
    
    if not image_files:
        return
    
    
    prompt = build_agent_operation_prompt(image_files, task)
    llm_result = describe_image_sequence(prompt, image_files)
    res = eval_from_json(llm_result, key_type="str")

    list_page = res.get("list_page", [])
    final_product_page = res.get("final_product_page", -1)
    
    product_list = []

    if list_page:
        prompt = list_page_prompt.format(task=task)
        llm_result = describe_image_sequence(prompt, [image_files[list_page[0]]])
        product_list = eval_from_json(llm_result, key_type="str").get("product_info", [])
    
    
    selected_index = -1
    
    if with_select_products:
        prompts = selected_product_prompt.format(task=task, product_list=with_select_products)
        llm_result = describe_image_sequence(prompts, [image_files[final_product_page]])
        res = eval_from_json(llm_result, key_type="str")
        selected_index = res.get("selected_index", -1)
    
    
    analysis_result = {
        "task": task,
        "image_sequence": image_files,
        "list_page": list_page,
        "final_product_page": final_product_page,
        "product_list": product_list,
        "selected_index": selected_index,
    }


    output_jsonl = f"{image_dir}/agent_analysis.json"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(analysis_result, ensure_ascii=False, indent=2))

    return analysis_result

check_is_select_prompt = """
# Role
You are a professional agent behavior analysis expert, focusing on evaluating whether the agent's mobile operations have achieved the goal: browsing products and selecting specific stores or products.

# Task
I will provide a mobile screen screenshot showing the agent's product selection interface during task execution. You need to judge whether the following two points are established based on the screenshot content:

1. Has the agent browsed the product list (i.e., the interface shows multiple products for selection)?
2. Has the agent selected a specific product (entered product detail page or clearly selected a specific product)?

# Rules
- If the current screenshot cannot indicate that the agent has browsed the product list, or has not selected any product, please return False.
- If the agent has browsed the product list and selected a specific product, please return True.

# Output
Please provide analysis results in strict JSON structured format, including the following fields:
{{
"is_select": <bool>,  # Whether the agent has browsed products and selected a specific product
}}
"""

def load_images(image_dir):
    """
    Load all image files in the specified directory
    """

    if os.path.isfile(image_dir):
        if image_dir.endswith(".jpg"):
            return [image_dir]
        else:
            return []
    
    image_files = sorted([str(p) for p in Path(image_dir).glob("*.jpg")])
    if not image_files:
        return []
    return image_files

def check_is_select(image_dir,task=""):
    return False


def check_is_list_page(image_dir,task=""):
    image_files = load_images(image_dir)
    if not image_files:
        return
    
    check_is_select_list_prompt = """
    # Role
    You are a professional mobile agent operation analysis expert.
    
    # Task
    You need to evaluate whether this page belongs to a product/store selection page.

    A product/store selection page refers to when a user inputs some query statements, and the mobile APP displays a list of related products for user selection. This list usually has 2 or more products or stores for user selection.

    What you need to do:
    1. Determine whether this page is a product/store selection page.
    2. If yes, please provide information about the store/product list.

    # Output fields:
    - is_list_page: <bool>,  # Whether it is a product/store selection page
    - product_list: [ <str>, <str>, ... ]  # If it's not a product/store selection page, return an empty list; if it is, please provide product/store name information. Generally, please provide complete title names.
    
    # Output
    Please provide analysis results in strict JSON structured format, do not input any other content, including explanations and comments:
    {{
    "is_list_page": <bool>,
    "product_list": [ <str>,<str>...]
    }}

    The following are screenshots you need to analyze:
    """

    llm_result = describe_image_sequence(check_is_select_list_prompt, image_files)
    res = eval_from_json(llm_result, key_type="str")
    print(res)
    return res

async def judge_result(target,operation,product_list):
    prompt = """
    # Role
    You are a goal achievement analysis expert.

    # Purpose
    You need to judge which target unit these operations finally selected and whether the target unit was selected, based on the provided operation information (including agent's goals, operations, and analysis).
    

    # Input
    ===================
    - Operation: {operation}
    ===================
    
    
    You need to judge the above operations and output the following conclusions:
    1. Which unit in the following candidate list did this operation select (starting from 0):
    - Candidate list: {product_list}
    
    2. Whether this target unit was selected:
    - Target information: {target}

    # Rules
    - Please judge which unit the agent finally selected based on the operation content and target information, output the selection index starting from 0.
    - If the operation indicates clicking on the target information, please return True, otherwise return False.

    # Output Rules
    Please provide analysis results in strict JSON structured format, including the following fields:
    {{
    "selected_index": <int>,  # Index of selected product in the list (starting from 0), -1 if no selection
    "success": <bool>,
    }}

    # Output
    Please only output JSON, no other content.

    Please start output:
    """

    prompt = prompt.format(target=target, operation=operation,product_list = json.dumps(product_list, ensure_ascii=False, indent=2))
    res = await describe_image_sequence_async(prompt,[])
    res = eval_from_json(res, key_type="str")
    return res