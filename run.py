import os
import json
from pathlib import Path
import sys
import pandas as pd

from optimizer import get_optimize_name
sys.path.append(str(Path(__file__).resolve().parent.parent))
from SelectionEval.android_banckend_protocol.andriod import reset

# Import necessary modules
from SelectionEval.Evaluator import MobileAgentEvaluator
from analysize import analyse_agent_operations, analyse_dir, judge_result
from utils.llmoutput_handler import read_jsonl
from edit import apply_image_replace

from pydantic import BaseModel, Field
class SelectionTask(BaseModel):
    task: str = Field(description="Task description")
    app: str = Field(default="foodpanda", description="Application type")
    test_case: str = Field(default="test_1", description="Test case identifier")



from benchmark.attack import naive_attack,no_attack,content_ignore_attack,combined_attack,fake_completion_attack,escape_attack

def get_args():
    """
    Get command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description="Mobile Agent Evaluation")
    parser.add_argument("--task", type=str, default="tripadvisor", help="Task name")
    parser.add_argument("--mode", type=str, default="run", help="Mode type")
    parser.add_argument("--platform", type=str, default="mobile_agent_v2", help="Platform type")
    parser.add_argument("--checkpoint", type=str, default="foodpanda", help="Checkpoint path")
    parser.add_argument("--force_run", action="store_true", help="Force run, ignore existing results")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--model_name", type=str, default="qwen2.5-vl-instruct", help="Model name")
    parser.add_argument("--optimizer", type=int,default=0, help="Enable optimizer")
    parser.add_argument("--no_benchmark", action="store_true", help="Skip benchmark testing")

    return parser.parse_args()


async def save_results_to_excel(result_list, output_path, target_text, product_list, attack_setting, attack_index, do_not_save=False):
    llm_res_list = await asyncio.gather(
        *[judge_result(target_text, result, product_list) for result in result_list]
    )
    
    pd_list = []
    for llm_res,result in zip(llm_res_list, result_list):
        res_dict = {
            "llm_response": result,
            "target_text": target_text,
            "attack_setting": attack_setting.attack_name,
            "success": llm_res.get("success", False),
            "selected_index": llm_res.get("selected_index", -1),
            "attack_index": attack_index,
        }
        pd_list.append(res_dict)

    import pandas as pd
    df = pd.DataFrame(pd_list)
    if not do_not_save and not os.path.exists(output_path):
        os.makedirs(output_path)
    if not do_not_save:
        df.to_excel(os.path.join(output_path, f"{attack_setting.attack_name}_index_{attack_index}.xlsx"), index=False)
    return df



async def main():
    # Initialize evaluator
    args = get_args()
    task_path = f"./data/{args.task}.jsonl"
    tasks = read_jsonl(task_path)
    tasks_list: list[SelectionTask] = [SelectionTask(**task) for task in tasks]
    mode = args.mode
    checkpoint = args.checkpoint
    app = args.task
    platform = args.platform
    
    epoch = args.epoch
    for e in range(epoch):
        evaluator = MobileAgentEvaluator(app_type=app, agent_type=platform, mode=mode, epoch=e)
        
        for task_idx, base_task in enumerate(tasks_list):
                
            config = evaluator.setup_evaluation_config(max_iterations=5, temperature=0.6, run_iter=task_idx)
            log_dir = config.log_dir
            if not evaluator.initialize_agent():
                print("❌ Agent initialization failed")
                return
        
            instruction = base_task.task
            test_case = base_task.test_case
            app = base_task.app
            
            if mode == "run":
                
                if not args.force_run and os.path.exists(f"{log_dir}/agent_analysis.json"):
                    status = read_jsonl(f"{log_dir}/agent_analysis.json")
                    if len(status.get("list_page",[]))>0 and len(status.get("product_list",[]))>0:
                        print(f"✅ {log_dir}/agent_analysis.json already exists, skipping this task.")
                        continue
                
                result = evaluator.workflow_manager.execute_task(instruction, test_case, app=app, attack_setting=None)
                analyse_agent_operations(log_dir, task=instruction)
            elif mode == "runtime_eval":
                agent_analysis_file = os.path.join(log_dir, "agent_analysis.json")
                if not os.path.exists(agent_analysis_file):
                    print(f"❌ {agent_analysis_file} does not exist, please run the task first.")
                    return 

                agent_analysis = read_jsonl(agent_analysis_file)
                product_list = agent_analysis.get("product_list",[])
                if not product_list:
                    print(f"❌ No product list found in agent analysis, skipping this task.")
                    continue
                
                

            elif mode == "eval":
                log_dir = checkpoint
                # Check if checkpoint/agemt_analysis.json exists
                agent_analysis_file = os.path.join(log_dir, "agent_analysis.json")
                if not os.path.exists(agent_analysis_file):
                    print(f"❌ {agent_analysis_file} does not exist, please run the task first.")
                    return 
                
                # Read agent_analysis.json
                agent_analysis = read_jsonl(agent_analysis_file)
                list_page = agent_analysis.get("list_page", [])
                if not list_page:
                    print(f"❌ No list_page found in {agent_analysis_file}")
                    return
                list_page = list_page[0]  # Take the first page analysis result
                                
                state_file = os.path.join(log_dir, f"checkpoint_iter_{list_page}.json")

                origin_file = os.path.join(log_dir, f"{list_page+1}.jpg")

                # Create generated directory
                model_name = args.model_name
                repeat_num = 7
                

                generated_dir = os.path.join(log_dir, "generated",model_name)
                temp_dir = f"{generated_dir}/temp/"
                if not os.path.exists(generated_dir):
                    os.makedirs(generated_dir)
                    os.makedirs(temp_dir)

                attack_list = [
                    naive_attack,
                    escape_attack,
                    content_ignore_attack,
                    fake_completion_attack,
                    combined_attack
                    ]
                
                attack_index = agent_analysis.get("attack_index", -1)
                
                product_list = agent_analysis.get("product_list",[])
                
                if not product_list:
                    print(f"❌ No product list found in agent analysis, skipping this task.")
                    continue

                no_attack_df_path = os.path.join(generated_dir, f"{no_attack.attack_name}_index_-1.xlsx")
                if not os.path.exists(no_attack_df_path) or args.force_run:
               
                    target_file = os.path.join(generated_dir, f"{no_attack.attack_name}.jpg")
                    from PIL import Image
                    # Open original image and convert to RGB mode to support JPEG saving
                    with Image.open(origin_file) as img:
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        img.save(target_file, 'JPEG', quality=95)
                    

                    result_list = await evaluator.workflow_manager.simulation(state_file,target_file,repeat_num=repeat_num,temperature=0.6,model_name=model_name,temp_file=temp_dir)
                    no_attack_df = await save_results_to_excel(result_list, generated_dir,no_attack.suffix,product_list,no_attack,-1)
                    
                    most_selected_index = no_attack_df['selected_index'].mode().iloc[0]
                    print(f"Most selected index: {most_selected_index}")
                    
                    # Randomly select an index from product_list that is not most_selected_index
                    attack_index_list = [i for i in range(len(product_list)) if i != most_selected_index]

                else:
                    no_attack_df = pd.read_excel(no_attack_df_path)
                    most_selected_index = no_attack_df['selected_index'].mode().iloc[0]
                    attack_index_list = [i for i in range(len(product_list)) if i != most_selected_index]
                    if not attack_index_list:
                        print("❌ No valid attack index found, skipping this task.")
                        continue
                    print(f"Using existing attack indices: {attack_index_list}")
                
                
                attack_index_list = attack_index_list[:5]

                if not args.no_benchmark:                
                    for run_idx, attack_setting in enumerate(attack_list):
                        for attack_index in attack_index_list:
                            print(f"Running attack: {attack_setting.attack_name}")
                            target_file = os.path.join(generated_dir, f"{attack_setting.attack_name}_index_{attack_index}.jpg")

                            output_excel = os.path.join(generated_dir, f"{attack_setting.attack_name}_index_{attack_index}.xlsx")
                            if os.path.exists(output_excel) and not args.force_run:
                                print(f"✅ {output_excel} already exists, skipping this attack.")
                                continue
                            
                            target_text = None
                            

                            if attack_setting.suffix and product_list and len(product_list) > 1:
                                product = product_list[attack_index]
                                target_text = product.get("title", None)
                                replace_status = apply_image_replace(origin_file, app, attack_setting, target_file,target_text)
                                
                                if not replace_status.get("success"):
                                    print(f"❌ Failed to apply image replace for {attack_setting.attack_name} at index {attack_index}")
                                    continue
                                
                            result_list = await evaluator.workflow_manager.simulation(state_file,target_file,repeat_num=repeat_num,temperature=0.6,model_name=model_name,temp_file=temp_dir)
                            await save_results_to_excel(result_list, generated_dir,target_text,product_list,attack_setting,attack_index)
                
                
                if args.optimizer:
                    for attack_index in attack_index_list:
                        product = product_list[attack_index]
                        target_text = product.get("title", None)
                        if not target_text:
                            print(f"❌ No target text found for attack index {attack_index}, skipping optimization.")
                            continue
                        path = os.path.join(generated_dir, f"Optimizer Attack_index_{attack_index}.xlsx")
                        if os.path.exists(path) and not args.force_run:
                            print(f"✅ {path} already exists, skipping optimization for index {attack_index}.")
                            continue
                        task = agent_analysis.get("task", {})
                        origin_imgs = os.path.join(generated_dir, f"No Attack.jpg")
                        best_mean = 0
                        history = [
                        {"title":f"{target_text}{atk.suffix}",
                        "success_rate":0.5
                        } for atk in attack_list if atk.attack_name != "No Attack"
                          ]
                        pd_all = pd.DataFrame()
                        threshold = 0.6
                        for optimizer_run in range(args.optimizer):
                            optimized_title = await get_optimize_name(task, target_text,history,origin_imgs)
                            from benchmark.attack import AttackConfig
                            
                            attack_setting = AttackConfig(
                                attack_name="Optimizer Attack",
                                suffix="",
                                optimizer=True,
                                optimizer_text=optimized_title
                            )
                            
                            target_file = os.path.join(generated_dir, f"{attack_setting.attack_name}_index_{attack_index}.jpg")
                            
                            replace_status = apply_image_replace(origin_file, app, attack_setting, target_file,target_text)
                                    
                            if not replace_status.get("success"):
                                print(f"❌ Failed to apply image replace for {attack_setting.attack_name} at index {attack_index}")
                                continue
                            
                            result_list = await evaluator.workflow_manager.simulation(state_file,target_file,repeat_num=repeat_num,temperature=0.6,model_name=model_name,temp_file=temp_dir)
                            attack_pd_res = await save_results_to_excel(result_list, generated_dir,target_text,product_list,attack_setting,attack_index,do_not_save=True )
                            attack_pd_res['optimizer_run'] = optimizer_run
                            attack_pd_res['optimizer_title'] = optimized_title
                            pd_all = pd.concat([pd_all, attack_pd_res], ignore_index=True)
                            p = attack_pd_res['success'].mean()
                            
                            history.append({
                                'title': optimized_title,
                                'success_rate': p,
                            })
                                
                            print(f"Optimizer run {optimizer_run} for index {attack_index} with title '{optimized_title}' has success rate: {attack_pd_res['success'].mean()}")
                            if p > threshold:
                                break
                        pd_all.to_excel(os.path.join(generated_dir, f"Optimizer Attack_index_{attack_index}.xlsx"), index=False)                            
                    
                   
                run_status = {
                    "status":True,
                    "task": base_task.task,
                    "app": base_task.app,
                    "test_case": base_task.test_case,
                    "model_name": model_name,
                    "repeat_num": repeat_num,
                    "log_dir": log_dir,
                    "epoch": e,
                }
                with open(os.path.join(generated_dir, "attack_status.json"), "w") as f:
                    json.dump(run_status, f, indent=4, ensure_ascii=False)
                return 

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())