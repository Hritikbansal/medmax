
import json
import os 
import numpy as np
import openai
from enum import Enum
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import random
import transformers
from evaluation.const import OAI_KEY


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    transformers.set_seed(seed)



def add_results_to_json(file_path, metrics):
    try:
        with open(file_path, 'r') as f: 
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    for key in metrics:
        data[key] = metrics[key]

    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    with open(file_path, 'w+') as f:
        json.dump(data, f, indent=4)
        
def log_samples(file_path, task_id, samples):
    try:
        with open(file_path, 'r') as f: 
            data = json.load(f)
    except FileNotFoundError:
        data = {}
        
    data[task_id] = samples
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    with open(file_path, 'w+') as f:
        json.dump(data, f, indent=4)
        
        
def calculate_accuracy_and_stderr(scores):
    scores = np.array(scores)  # Convert to NumPy array if necessary
    accuracy = np.mean(scores)
    standard_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return accuracy, standard_error



def compare_messages_gen(question, true_answer, generated_answer):
    messages = []
    prompt = f"""
        Given a question about an medical image, there is a correct answer to the question and an answer to be determined. If the answer to be determined matches the correct answer or is a good enough answer to the question, output 1; otherwise output 0. Evaluate the answer to be determined (1 or 0).

        Question:
        - question about the medical image: {question}\n

        Answers:
        - correct answer(ground truth): {true_answer}\n
            answer to be determined: {generated_answer}\n

        Task:\n
        - Given a question about an medical image, there is a correct answer to the question and an answer to be determined. If the answer to be determined matches the correct answer or is a good enough answer to the question, output 1; otherwise output 0. Evaluate the answer to be determined (1 or 0).

        Output Format:
        Correctness: your answer\n
        """

    messages.append({"role": "user", "content": prompt})
    return messages

class GPT:
  prompt_percent = 0.8

  openai_cxn_dict = {
    'default': {
        'api_key': OAI_KEY,
    },
  }

  deployment_max_length_dict = {
    'gpt-4': 8192,
    'gpt-4-0314': 8192,
    'gpt-4-32k': 32768,
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16385,
    'gpt-4o-mini': 16384,
  }

  def __init__(self, model_id):
    self.temperature = 0.0
    self.top_k = 1
    self.openai_api = 'default'
    self.model_id = model_id
    self.max_length = self.deployment_max_length_dict[model_id]
    self.client = openai.OpenAI(api_key=self.openai_cxn_dict[self.openai_api]['api_key'])

  # @backoff.on_exception(backoff.expo, openai.RateLimitError)
  def make_api_call_to_gpt(self, messages):
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=messages,
    )
    return response.choices[0].message.content

  def infer(self, messages):
    result = self.make_api_call_to_gpt(messages)
    return result

def gpt_score(question, true_answer, generated_answer):
    model_inst = GPT("gpt-4o-mini")
    input_msg = compare_messages_gen(question, true_answer, generated_answer)
    response = model_inst.infer(input_msg)
    return response

class SimilarityType(Enum):
    IMAGE_CAPTION = "image_caption"
    CAPTION_CAPTION = "caption_caption"
    IMAGE_IMAGE = "image_image"


class CLIPSimilarity:
    def __init__(self, model_name, context_length=256, device=None):
        self.model_name = model_name
        self.context_length = context_length
        self.device = (
            device if device else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters"
        )

    def _process_image(self, image_path):
        """Process a single image and return its features"""
        with Image.open(image_path) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            image = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features / image_features.norm(dim=-1, keepdim=True)

    def _process_text(self, text):
        """Process a single text and return its features"""
        texts = self.tokenizer([text], context_length=self.context_length).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            return text_features / text_features.norm(dim=-1, keepdim=True)

    def calculate_similarity(self, source1, source2, similarity_type: SimilarityType):
        # cosine similarity, higher score means more similar
        if similarity_type == SimilarityType.IMAGE_CAPTION:
            img_features = self._process_image(source1)
            txt_features = self._process_text(source2)
            if img_features is None or txt_features is None:
                return None
            similarity = (img_features @ txt_features.t()).item()

        elif similarity_type == SimilarityType.CAPTION_CAPTION:
            txt1_features = self._process_text(source1)
            txt2_features = self._process_text(source2)
            if txt1_features is None or txt2_features is None:
                return None
            similarity = (txt1_features @ txt2_features.t()).item()

        elif similarity_type == SimilarityType.IMAGE_IMAGE:
            img1_features = self._process_image(source1)
            img2_features = self._process_image(source2)
            if img1_features is None or img2_features is None:
                return None
            similarity = (img1_features @ img2_features.t()).item()

        return similarity

def get_biomed_clip_model():
    return CLIPSimilarity(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")


def chameleon_prompt_processor(question, image_path, task_type):
    
    if task_type == "visual_chat":
        return False # Flag to turn off sft mode
    
    if task_type == "mm_out":
        
        question += " Answer the question with one picture."
        content = [question]
        modality = ["text"]
        return content, modality
    
    if task_type == "image_captioning":
        content = [image_path, "Please describe this picture.",]
        modality = ["image", "text"]
        return content, modality
    
    if task_type == "image_generation":
        content = [question]
        modality = ["text"]
        return content, modality
    
    if not question.endswith('\n'):
        question += '\n'
        
    question = f"Question: {question}Answer:"
        
    if task_type == "closed":
        question = f"""Answer the question based on this image and respond 'yes' or 'no'.\n{question}"""
    
    elif task_type == "open":
        question = f"""Answer the question based on this image.\n{question}"""
        
    elif task_type == "mcq":
        question = f"""Answer the question based on this image and respond 'A', 'B', 'C', or 'D'.\n{question}"""
    
    
    content = [image_path, question]
    modality = ["image", "text"]
    
    return content, modality

def sft_prompt_processor(question, image_path, task_type):
    
    if task_type == "visual_chat":
        return True # Flag to turn on sft mode
    
    if task_type == "mm_out" or task_type == "image_generation":
        
        content = [question, "<END-OF-TURN>"]
        modality = ["text", "sentinel"]
        return content, modality
    
    if task_type == "image_captioning":
        content = [image_path, "Please describe this picture.", "<END-OF-TURN>"]
        modality = ["image", "text", "sentinel"]
        return content, modality
    
    content = [image_path, question, "<END-OF-TURN>"]
    modality = ["image", "text", "sentinel"]
    return content, modality
    

def default_prompt_processor(question, image_path, task_type):
    content = [image_path, question]
    modality = ["image", "text"]
    return content, modality