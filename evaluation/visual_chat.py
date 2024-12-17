import os
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pandas as pd
from evaluation.eval_utils import add_results_to_json
from inference.inference_utils import chameleon_generate
import json
import shortuuid
import math
import tiktoken
import openai
from datasets import load_dataset
import ast
from evaluation.const import OAI_KEY


INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
  Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
  Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
ROLE = 'Assistant'


def load_file_jsonl(path):
   with open(path) as f:
        return [json.loads(row) for row in f]

def get_avg(x):
  return sum([float(y) for y in x])/len(x)

def get_domain(x):
  for domain in ['chest_xray', 'mri', 'histology', 'gross', 'ct_scan']:
    in_domain = x['domain'][domain]
    if in_domain:
      return domain

def chunk(lst, n):
  for i in range(0, len(lst), n):
    if i+(1.5*n)<len(lst):
      end = i + n
    else:
      end = len(lst)
    yield lst[i:end]
    if end==len(lst):
      return

def conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2):
  return (f'[Context]\n'
          f'Figure Caption:\n{fig_label}: {fig_caption}\n\n'
          f'Figure Context:\n\t- {fig_context}\n\n'
          f'[Question]\n{question}\n\n'
          f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
          f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
          f'[System]\n{INSTRUCT_PROMPT}\n\n')

def compare_messages_gen(fig_label, fig_caption, fig_context, question, ans1, ans2):
  messages = [
  {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
  ]
  messages.append({"role": "user", "content": conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2)})
  return messages

class GPT:
  prompt_percent = 0.8

  # TODO: use a more secure way to store the API key
  openai_cxn_dict = {
    'default': {
        'api_key': OAI_KEY
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
    self.encoding = tiktoken.encoding_for_model("-".join(model_id.split("-", 2)[:2]).replace('5', '.5'))
    self.openai_api = 'default'
    self.model_id = model_id
    self.max_length = self.deployment_max_length_dict[model_id]
    self.client = openai.OpenAI(api_key=self.openai_cxn_dict[self.openai_api]['api_key'])

  def gen_messages(self, fixed_instruction, few_shot_examples, input, input_header, output_header):
    messages = [
      {
          "role": "system",
          "content": fixed_instruction,
      },
    ]
    for example in few_shot_examples:
      messages.extend([
          {
            "role": "user",
            "content": input_header+'\n'+example['user']+'\n\n'+output_header,
          },
          {
            "role": "assistant",
            "content": example['assistant'],
          },
      ])
    messages.extend([
        {
          "role": "user",
          "content": input_header+'\n'+input+'\n\n'+output_header,
        },
    ])
    return messages

  # @backoff.on_exception(backoff.expo, openai.RateLimitError)
  def make_api_call_to_gpt(self, messages):
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=messages,
    )
    return response.choices[0].message.content

  def infer(self, messages_list):
    results = []
    for messages in messages_list:
      result = self.make_api_call_to_gpt(messages)
      results.append(result)
    return results

  def split_input(self, fixed_instruction, few_shot_examples, splittable_input, input_header, output_header):
    # Tokenize fixed_prompt
    fixed_token_ids = self.encoding.encode(fixed_instruction+' '.join([x['user']+' '+x['assistant'] for x in few_shot_examples]))
    # Calculate remaining token length
    remaining_token_len = math.ceil((self.prompt_percent*self.max_length)-len(fixed_token_ids))
    
    # Tokenize splittable_input
    split_token_ids = self.encoding.encode(splittable_input)

    # Split tokenized split_prompt into list of individual inputs strings
    split_token_ids_list = [split_token_ids[i:i+remaining_token_len+10] for i in range(0, len(split_token_ids), remaining_token_len)] 
    split_input_list = [self.encoding.decode(split_token_ids) for split_token_ids in split_token_ids_list]

    # Generate list of prompt strings
    return [self.gen_messages(fixed_instruction, few_shot_examples, split_input, input_header, output_header) for split_input in split_input_list]

def infer(samples):
    model_inst = GPT("gpt-4o-mini")

    BATCH_SIZE = 1
    batch_samples = []
    results = []
    batch = []
    
    print('Starting Multimodal Chat GPT Scoring Eval')

    for sample in tqdm(samples):
        sample_copy = deepcopy(sample)
        input_msg = compare_messages_gen(sample_copy['fig_label'], sample_copy['fig_caption'], sample_copy['in_text_mention'], sample_copy['question'], sample_copy['ans1'], sample_copy['ans2'])
        batch.append(input_msg)
        batch_samples.append(sample_copy)
        if len(batch)>=BATCH_SIZE:
            inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
            for item, inference_result in zip(batch_samples, inference_results):
                item['gpt_eval'] = inference_result
            results.extend(batch_samples)
            batch = []
            batch_samples = []
    inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
    for item, inference_result in zip(batch_samples, inference_results):
        item['gpt_eval'] = inference_result
    results.extend(batch_samples)
    print(f"Result Size: {len(results)}")
    return results

def run_visual_chat_eval(model, prompt_processor, save_dir, save_name, eval_data_dir):
    
    DEFAULT_IMAGE_TOKEN = "<image>"

    answers_file = f"{save_dir}/logs/visual_chat_answers_model_{save_name}.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    dataset = load_dataset("mint-medmax/medmax_eval_data")
    
    dataset = dataset.filter(lambda example: example['task_name'] == "visual_chat")
    for idx, example in tqdm(enumerate(dataset)):

        image_path = os.path.join(eval_data_dir, example['image_path'])
        
        qs = example['prompt'].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        content = []
        modality = []

        if example['question_type'] == 'detailed_description':
          content.append(image_path)
          modality.append("image")   
          content.append("Analyze the image in a comprehensive and detailed manner.")
          modality.append("text")

        else:
          if example['prompt'].endswith(DEFAULT_IMAGE_TOKEN):
            # put image at the end
            content.append(qs)
            modality.append("text")
            content.append(image_path)
            modality.append("image")        
          else:
            # put image at the start
            content.append(image_path)
            modality.append("image")
            content.append(qs)
            modality.append("text")
        
        sft = prompt_processor(None, None, "visual_chat")
        
        generated_text = chameleon_generate(model, content=content, modality=modality, task="text-gen", sft=sft, max_gen_len=4096)[0] 
        generated_text = generated_text.strip()
    
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": generated_text,
                                   "answer_id": ans_id,
                                   "model_id": save_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    
    ### score the answers
    gen_answer_data = load_file_jsonl(answers_file)

    question_data = dataset['prompt']
    answer_data = dataset['answer']
    type_data = dataset['question_type']
    

    id = 0
    samples = []
    for question, gen_answer, answer, question_type in zip(question_data, gen_answer_data, answer_data, type_data):
        question_copy = deepcopy(question)

        answer_dict = ast.literal_eval(answer)
        
        answer_dict['question'] = question_copy
        answer_dict['ans1'] = answer_dict['gpt4_answer']
        answer_dict['ans2'] = gen_answer
        answer_dict['question_id'] = id
        answer_dict['type'] = question_type
        
        id += 1
        
        samples.append(answer_dict)

    results = infer(samples)

    # Create parent directory of output score files if it doesn't exist
    scores_file = f"{save_dir}/logs/{save_name}_visual_chat_gpt4_scores_model.jsonl"
    os.makedirs(Path(scores_file).parent, exist_ok=True)

    with open(scores_file, 'w') as f:
        for row in results:
            f.write(json.dumps(row)+'\n')

    # summarize the results
    scores_data = load_file_jsonl(scores_file)
    predictions = [(x['question_id'], x['type'], get_domain(x), x['gpt_eval'].split('\n')[0].strip().split(' ')) for x in scores_data]
    
    score_type_dict = defaultdict(lambda: defaultdict(list))
    for q_id, q_type, domain, (a1_score, a2_score) in predictions:
        score_type_dict[q_type][1].append(a1_score)
        score_type_dict[q_type][2].append(a2_score)
        score_type_dict['overall'][1].append(a1_score)
        score_type_dict['overall'][2].append(a2_score)
        score_type_dict[domain][1].append(a1_score)
        score_type_dict[domain][2].append(a2_score)

    result = defaultdict(dict)

    for q_type, score_dict in score_type_dict.items():
        result[q_type]['gpt4_score'] = get_avg(score_dict[1])
        result[q_type]['pred_score'] = get_avg(score_dict[2])
        result[q_type]['pred_relative_score'] = get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])])*100
        result[q_type]['data_size'] = len(score_dict[1])

    df = pd.DataFrame.from_dict(result).filter(['conversation', 'detailed_description', 'chest_xray', 'mri', 'histology', 'gross', 'ct_scan', 'overall'])
    
    metrics = {"visual_chat": df.to_dict()}
    file_path = f"{save_dir}/results/{save_name}.json"
    add_results_to_json(file_path, metrics)

