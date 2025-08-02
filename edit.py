
from SelectionEval.attack_model.replace_text import replace_by_content
from benchmark.attack import AttackConfig, naive_attack,combined_attack
import json
from analysize import check_is_list_page

def apply_image_replace(image_path,app_type: str ,attack_setting:AttackConfig,save_path,target_text=None,
                        config = None
                        ):

    if not attack_setting:
        return  {
        "success": False,
        }

    if config is None:
        config_path = f"./config/{app_type}.json"
        with open(config_path, "r") as f:
            config = json.load(f)

    if not target_text:
        res = check_is_list_page(image_path)
        is_list_page = res.get("is_list_page", False)
        product_list = res.get("product_list", [])
        target_text = product_list[1] if (is_list_page and len(product_list) > 1) else None

    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if target_text:
        if not attack_setting.optimizer:
            new_text = target_text + attack_setting.suffix
        else:
            new_text = attack_setting.optimizer_text
        return replace_by_content(
                            image_path,
                            target_text= target_text,
                            new_text=new_text,
                            font_path=config.get("font_path", font_path),
                            font_size=config.get("font_size"),
                            line_height=config.get("line_height"),  
                            adjust_config = config.get("adjust_config", {}),
                            output_path = save_path,
                            text_color= config.get("color", "black"),  
                            background_color=config.get("background_color", "white"),  
                            upper_text=attack_setting.upper_text,  
                            upper_distance=config.get("upper_distance"), 
                            min_distance=config.get("min_distance", 10)  
                        )
