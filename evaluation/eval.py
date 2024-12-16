from datasets import load_from_disk
from inference.inference_utils import chameleon_generate, load_chameleon
from evaluation.eval_utils import add_results_to_json, log_samples, gpt_score, calculate_accuracy_and_stderr, chameleon_prompt_processor, sft_prompt_processor
import argparse
from tqdm import tqdm
import os



# def run_evals(tasks):

    
def run_vqa_eval(task_name, task_type, model, prompt_processor, save_dir, save_name, eval_data_dir):
    dataset = load_from_disk("/localhome/data/datasets/medmax_eval_data/vqa_eval_data")
    dataset = dataset.filter(lambda example: example['task_name'] == task_name and example['question_type'] == task_type)
    
    scores = []
    logs = []
    
    print(f"Running evaluation for {task_type} task: {task_name}...")
    
    for example in tqdm(dataset):
        print(example)
        question = example['question']
        answer = example['answer']
        image_path = os.path.join(eval_data_dir, example['image_path'])
        
        content, modality = prompt_processor(question, image_path, task_type) # SFT tokens inserted manually
        
        if task_type == "closed":
            generated_text = chameleon_generate(model, 
                                                content=content, 
                                                modality=modality, 
                                                task="text-gen", 
                                                sft=False, 
                                                max_gen_len=1, 
                                                save_dir=f"{save_dir}/inference")[0]
            generated_text = generated_text.lower()
            is_correct = (answer == 'yes' and 'yes' in generated_text and 'no' not in generated_text) or \
                (answer == 'no' and 'no' in generated_text and 'yes' not in generated_text)
            
        elif task_type == "open":
            generated_text = chameleon_generate(model, 
                                                content=content, 
                                                modality=modality, 
                                                task="text-gen", 
                                                sft=False, 
                                                max_gen_len=10, 
                                                save_dir=f"{save_dir}/inference")[0]
            generated_text = generated_text.lower()
            
            gpt_response = gpt_score(question, answer, generated_text)
            
            if 'Correctness: 1' in gpt_response:
                is_correct = 1
            elif 'Correctness: 0' in gpt_response:
                is_correct = 0
            else:
                continue
            
            
        elif task_type == "mcq":
            generated_text = chameleon_generate(model, 
                                                content=content, 
                                                modality=modality, 
                                                task="text-gen", 
                                                sft=False, 
                                                max_gen_len=1, 
                                                save_dir=f"{save_dir}/inference")[0]
            generated_text = generated_text.upper().strip()
            is_correct = answer == generated_text
        else:
            raise ValueError("Invalid task type")
        
        
        scores.append(is_correct)
        logs.append({"question": question, "image_path": image_path, "answer": answer, "generated_text": generated_text, "is_correct": is_correct})
    
    
    task_id = f"{task_name}_{task_type}"
    file_path = f"{save_dir}/logs/{save_name}.json"
    log_samples(file_path, task_id, logs)
    
    
    file_path = f"{save_dir}/results/{save_name}.json"
    
    accuracy, std_err = calculate_accuracy_and_stderr(scores)
    metrics = {f"{task_id}" : {"accuracy": accuracy, "std_err": std_err}}
    add_results_to_json(file_path, metrics)
        
        
def run_all_vqa_evals(model, prompt_processor, save_dir, save_name, eval_data_dir):

    VQA_TASKS = [("vqa_rad", "closed"), ("slake", "closed"), ("pathvqa", "closed"), ("quilt_vqa", "closed"),
                 ("vqa_rad", "open"), ("slake", "open"), ("pathvqa", "open"), ("quilt_vqa", "open"),
                 ("pmc_vqa", "mcq"), ("omnimed_vqa", "mcq"), ("path_mmu", "mcq"), ("probmed", "closed")]
    for task_name, task_type in VQA_TASKS:
        run_vqa_eval(task_name, task_type, model, prompt_processor, save_dir, save_name, eval_data_dir)

    
def main(args):
    model = load_chameleon(args.ckpt)
    
    # run_all_vqa_evals(model, chameleon_prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    
    run_all_vqa_evals(model, sft_prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default='/localhome/data/ckpts/Anole-7b-v0.1', type=str, help="Path to checkpoint")
    parser.add_argument("--save_dir", default="evaluation/outputs", type=str, help="The directory to save model outputs, eval logging, and results.")
    parser.add_argument("--eval_data_dir", default="/localhome/data/datasets/medmax_eval_data", type=str, help="The directory to save the logs.")
    parser.add_argument("--save_name", default=None, type=str, help="The name of the saved file.")
    
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)


    
    
    
    