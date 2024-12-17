from inference.inference_utils import load_chameleon
from evaluation.eval_utils import chameleon_prompt_processor, sft_prompt_processor, default_prompt_processor, set_seeds
import argparse
from evaluation.vqa import run_vqa_evals
from evaluation.mm_out import run_mm_out_eval
from evaluation.visual_chat import run_visual_chat_eval
from evaluation.image_text import run_image_captioning_eval, run_image_generation_eval


def main(args):
    model = load_chameleon(args.ckpt)
    
    if args.prompt_processor == "sft":
        prompt_processor = sft_prompt_processor
    elif args.prompt_processor == "chameleon":
        prompt_processor = chameleon_prompt_processor
    else:
        prompt_processor = default_prompt_processor

    set_seeds(42)
    run_vqa_evals(model, prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    
    set_seeds(42)
    run_mm_out_eval(model, prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    
    set_seeds(42)
    run_visual_chat_eval(model, prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    
    set_seeds(42)
    run_image_captioning_eval(model, prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    
    set_seeds(42)
    run_image_generation_eval(model, prompt_processor, args.save_dir, args.save_name, args.eval_data_dir)
    

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default='/localhome/data/ckpts/Anole-7b-v0.1', type=str, help="Path to checkpoint")
    parser.add_argument("--save_dir", default="evaluation/outputs", type=str, help="The directory to save model outputs, eval logging, and results.")
    parser.add_argument("--save_name", default=None, type=str, help="The name of the saved file.")
    parser.add_argument("--eval_data_dir", default="/localhome/data/datasets/medmax_eval_data", type=str, help="The directory to save the logs.")
    parser.add_argument("--prompt_processor", default="default", type=str, help="How to prompt the model for the evaluation")
    
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)


    
    
    
    