import torch
import argparse
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from enum import Enum
from open_clip import create_model_from_pretrained, get_tokenizer
from evaluation.eval_utils import add_results_to_json, log_samples, calculate_accuracy_and_stderr, get_biomed_clip_model, SimilarityType
from inference.inference_utils import chameleon_generate, load_chameleon
import os
from datasets import load_from_disk



# def main(args):
def run_image_captioning_eval(model, prompt_processor, save_dir, save_name, eval_data_dir):

    similarity_calculator = get_biomed_clip_model()
    
    datanames = ["pmc_oa", "quilt", "report"]
    
    results_dict = {}
    
    for dataname in datanames:

        captioning_similarity_scores = []
        
        text_samples = []
        
        data = load_from_disk("/localhome/data/datasets/medmax_eval_data/eval_data")
        data = data.filter(lambda example: example['task_name'] == "image_captioning" and example['question_type'] == dataname)
        
        for instance in tqdm(data):
            
            image_path = os.path.join(eval_data_dir, instance["image_path"])
            
            # caption = instance["prompt"]
            
            content, modality = prompt_processor(None, image_path, "image_captioning")
            
                # text_outputs = anole_generate(model, content=image_path, modality="image", task="text-gen", sft=args.sft, max_gen_len=60)[0]
            text_outputs = chameleon_generate(model, content=content, modality=modality, task="text-gen", sft=False, max_gen_len=4096)[0]
            
            captioning_score = similarity_calculator.calculate_similarity(
            image_path, text_outputs, SimilarityType.IMAGE_CAPTION
            )
            
            captioning_similarity_scores.append(captioning_score)
            text_samples.append({"image": image_path, "caption": text_outputs, "score": captioning_score})
    
        avg_captioning_similarity, captioning_stderr = calculate_accuracy_and_stderr(captioning_similarity_scores)

        
        results_dict[dataname] = {"captioning_similarity": avg_captioning_similarity,
                                    "captioning_stderr": captioning_stderr}
    
    
    log_samples(f"{save_dir}/logs/{save_name}", "image_captioning", text_samples)
    
    results_dict = {"image_captioning": results_dict}
    save_path = f"{save_dir}/results/{save_name}.json"
    add_results_to_json(save_path, results_dict)


def run_image_generation_eval(model, prompt_processor, save_dir, save_name, eval_data_dir):
    
    similarity_calculator = get_biomed_clip_model()
    # model = load_anole(args.ckpt)
    
    
    datanames = ["pmc_oa", "quilt", "report"]
    results_dict = {}
    for dataname in datanames:

        data = load_from_disk("/localhome/data/datasets/medmax_eval_data/eval_data")
        data = data.filter(lambda example: example['task_name'] == "image_generation" and example['question_type'] == dataname)
        
        generation_similarity_scores = []
        image_samples = []
            
        for instance in tqdm(data):
            caption = instance["prompt"]
            
            content, modality = prompt_processor(caption, None, "image_generation")
            image_outputs = chameleon_generate(model, content=content, modality=modality, task="image-gen", sft=False, max_gen_len=60, save_dir=f"{save_dir}/inference")[0] 
        
            generation_score = similarity_calculator.calculate_similarity(
                image_outputs, caption, SimilarityType.IMAGE_CAPTION
            )
            
            generation_similarity_scores.append(generation_score)
            image_samples.append({"caption": caption, "image": image_outputs, "score": generation_score})
            
        avg_generation_similarity, generation_stderr = calculate_accuracy_and_stderr(generation_similarity_scores)
        
        results_dict[dataname] = {"generation_similarity": avg_generation_similarity,
                                    "generation_stderr": generation_stderr}
        
    
    log_samples(f"{save_dir}/logs/{save_name}", "image_generation", image_samples)
    
    results_dict = {"image_generation": results_dict}
    save_path = f"{save_dir}/results/{save_name}.json"
    add_results_to_json(save_path, results_dict)

# def parse_arguments() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Generate interleaved image-text content based on text instructions.")
#     parser.add_argument("--ckpt", default='/localhome/data/ckpts/Anole-7b-v0.1', type=str, help="path to anole checkpoint.")
#     parser.add_argument("--save_dir", default="eval/outputs", type=str, help="The directory to save the generated images.")
#     parser.add_argument("--save_name", default=None, type=str, help="The name of the saved file.")
#     parser.add_argument("--sft", default=False, type=bool, help="Use sft")
#     parser.add_argument("--temp", default=1.0, type=float, help="Text decoding temp")
#     args: argparse.Namespace = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args: argparse.Namespace = parse_arguments()
#     main(args)


