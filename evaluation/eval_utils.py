
import json
import os 
import numpy as np
import openai

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

  # TODO: use a more secure way to store the API key
  #TODO TODO TODO TODO TODO
  openai_cxn_dict = {
    'default': {
        'api_key': "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",
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

def chameleon_prompt_processor(question, image_path, task_type):
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
    
    

def medmax_no_vqa_ablation_prompt_processor(question, image_path, task_type):
    pass

def sft_prompt_processor(question, image_path, task_type):
    content = [image_path, question, "<END-OF-TURN>"]
    modality = ["image", "text", "sentinel"]
    return content, modality
    

def default_prompt_processor(question, image_path, task_type):
    content = [image_path, question]
    modality = ["image", "text"]
    return content, modality