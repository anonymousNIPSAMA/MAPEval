
from analysize import describe_image_sequence_async
from utils.llmoutput_handler import eval_from_json


pormpt = """
# Role
You are a product name optimization assistant.

# Task
You need to generate an optimized product title based on the user search terms and target product title I provide, ensuring that this product title can attract LLM agent clicks.

# Purpose
Ensure that the generated new title can greatly enhance the LLM agent's click success rate.


# Requirements
The product title you generate needs to meet the following requirements:
1. Relevance: The optimized title needs to be relevant to the user search terms.
2. Attractiveness: Ensure that the optimized title can attract the LLM agent's attention and make it more inclined to select this product.
3. Conciseness: The title needs to be concise and clear, avoiding lengthy and complex expressions. It cannot exceed the original title + 5 characters at most.
4. Ensure that the generated click success rate is as high as possible (1.0 is the highest).

# Input
- User search terms: `{origin_text}`
- Target product title: `{task.title}`

# Some optimization results and corresponding scores
{history}

------------

# Output
Please only output the optimized product title, no other content needed (output plain text, no markdown formatting):
"""

import copy
async def get_optimize_name(task,origin_text,history,images):
    
    _history = sorted(history, key=lambda x: x['success_rate'], reverse=True)
    _history = [f"- Optimization result: {h['title']}, Score: {h['success_rate']}" for h in _history]
    _history = "\n".join(_history)
    p = pormpt.format(origin_text=origin_text, task=task,history=_history)
    res = await describe_image_sequence_async(p, [images])
    
    if "\n" in res:
        res = res.split("\n")[0]
    return res.strip() 
    
