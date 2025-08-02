
import re
import ast

from json_repair import json_repair, loads

def eval_from_json(response: str, key_type: str = "int"):
    if not response or not isinstance(response, str):
        return {}
        
    try:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            res = ast.literal_eval(json_content)
            if isinstance(res, dict) and key_type == "int":
                res = {int(k): v for k, v in res.items()}
            return res

        pattern = r"```(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            res = ast.literal_eval(json_content)
            if isinstance(res, dict) and key_type == "int":
                res = {int(k): v for k, v in res.items()}
            return res

        r = response.removeprefix("```json").removesuffix("```").strip()
        res = ast.literal_eval(r)
        if isinstance(res, dict) and key_type == "int":
            res = {int(k): v for k, v in res.items()}
        return res

    except Exception as e:
        try:
            repaired_json = json_repair.repair_json(response)
            if repaired_json:
                res = loads(repaired_json)
                if isinstance(res, dict) and key_type == "int":
                    res = {int(k): v for k, v in res.items()}
                return res
            else:
                print(f"json_repair failed to repair: {response[:200]}...")
        except Exception as e2:
            print(f"Error in json_repair: {e2} | response: {response[:200]}...")
        
        return {} if key_type else []
    



import json
def read_jsonl(file_path: str):
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if item:
                    data.append(item)
            except Exception as e:
                print(f"Error parsing line: {line.strip()} | Error: {e}")
    return data