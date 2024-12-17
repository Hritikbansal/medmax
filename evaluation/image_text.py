from tqdm import tqdm
from datasets import load_dataset
from evaluation.eval_utils import add_results_to_json, log_samples, calculate_accuracy_and_stderr, get_biomed_clip_model, SimilarityType
from inference.inference_utils import chameleon_generate
import os



def run_image_captioning_eval(model, prompt_processor, save_dir, save_name, eval_data_dir):

    similarity_calculator = get_biomed_clip_model()
    
    datanames = ["pmc_oa", "quilt", "report"]
    
    results_dict = {}
    
    for dataname in datanames:

        captioning_similarity_scores = []
        
        text_samples = []
        
        data = load_dataset("mint-medmax/medmax_eval_data")
        data = data.filter(lambda example: example['task_name'] == "image_captioning" and example['question_type'] == dataname)
        
        for instance in tqdm(data):
            
            image_path = os.path.join(eval_data_dir, instance["image_path"])
            
            content, modality = prompt_processor(None, image_path, "image_captioning")
            
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
    
    datanames = ["pmc_oa", "quilt", "report"]
    results_dict = {}
    for dataname in datanames:

        data = load_dataset("mint-medmax/medmax_eval_data")
        data = data.filter(lambda example: example['task_name'] == "image_generation" and example['question_type'] == dataname).select(range(100)) # 100 examples is sufficient to compute a score
        
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

