import os
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from evaluation.eval_utils import add_results_to_json, get_biomed_clip_model, SimilarityType
from inference.inference_utils import chameleon_generate
import json
import math
import tiktoken
import openai
from datasets import load_dataset
from evaluation.const import OAI_KEY


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

INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
  Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
  Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
ROLE = 'Assistant'

def conv_to_str(question, ans1, ans2):
  return (f'[Context]\n'
          f'[Question]\n{question}\n\n'
          f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
          f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
          f'[System]\n{INSTRUCT_PROMPT}\n\n')


def compare_messages_gen(question, ans1, ans2):
  messages = [
  {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
  ]
  messages.append({"role": "user", "content": conv_to_str(question, ans1, ans2)})
  return messages

class GPT:
  prompt_percent = 0.8

  openai_cxn_dict = {
    'default': {
        'api_key': OAI_KEY
        }
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
        input_msg = compare_messages_gen(sample_copy['question'], sample_copy['gt_answer'], sample_copy['pred_answer'])
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


def run_mm_out_eval(model, prompt_processor, save_dir, save_name, eval_data_dir):
    
    dataset = load_dataset("mint-medmax/medmax_eval_data")
    dataset = dataset.filter(lambda example: example['task_name'] == "mm_out")
    
    answers_file = f"{save_dir}/logs/{save_name}.jsonl"

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for example in tqdm(dataset):
        question = example['prompt']
        text_answer = example['answer']
        image_answer = os.path.join(eval_data_dir, example['image_path'])
        content, modality = prompt_processor(question, None, "mm_out")
        
        generated_outputs = chameleon_generate(model, content=content, modality=modality, task="any", sft=False, max_gen_len=1024, save_dir=f"{save_dir}/inference")
        text_output = []
        image_output = []
        for out in generated_outputs:
            if out.endswith(".png"):
               image_output.append(out)
            else:
                text_output.append(out)
        text_output = " ".join(text_output)
        image_output = "" if len(image_output) == 0 else image_output[0]
        if save_dir not in image_answer:
            image_answer = os.path.join(f"{save_dir}/inference", image_answer)
    
        ans_file.write(json.dumps({
                                   "question": question,
                                   "text": text_output,
                                   "image": image_output,
                                   "text_answer": text_answer,
                                   "image_answer": image_answer,
                                }) + "\n")
        ans_file.flush()
    ans_file.close()
    
    ### score the answers
    answer_data = load_file_jsonl(answers_file)

    samples = []
    for instance in answer_data:
        samples.append({'question': instance['question'], 'gt_answer': instance['text_answer'], 'pred_answer': instance['text']})
    
    results = infer(samples)
    
    # Create parent directory of output score files if it doesn't exist
    scores_file = f"{save_dir}/logs/{save_name}_mm_out_gpt4_text_scores.jsonl"
    os.makedirs(Path(scores_file).parent, exist_ok=True)

    with open(scores_file, 'w') as f:
        for row in results:
            f.write(json.dumps(row)+'\n')

    # summarize the results
    scores_data = load_file_jsonl(scores_file)
    predictions = [x['gpt_eval'].split('\n')[0].strip().split(' ') for x in scores_data]
    score_type_dict = defaultdict(lambda: defaultdict(list))
    for (a1_score, a2_score) in predictions:
        score_type_dict['overall'][1].append(a1_score)
        score_type_dict['overall'][2].append(a2_score)

    result = defaultdict(dict)

    for q_type, score_dict in score_type_dict.items():
        result[q_type]['gpt4_score'] = get_avg(score_dict[1])
        result[q_type]['pred_score'] = get_avg(score_dict[2])
        result[q_type]['pred_relative_score'] = get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])])*100
        result[q_type]['data_size'] = len(score_dict[1])

    text_score = result['overall']['pred_relative_score']
    num_text = result['overall']['data_size']

    similarity_calculator = get_biomed_clip_model()

    scores_file = f"{save_dir}/logs/{save_name}_clip_image_scores.jsonl"
    os.makedirs(Path(scores_file).parent, exist_ok=True)

    generation_similarity_scores = []
    for instance in answer_data:
      if instance['image'] != "":
        score = similarity_calculator.calculate_similarity(instance['image'], instance['image_answer'], SimilarityType.IMAGE_IMAGE)
        instance['clipscore'] = score
        generation_similarity_scores.append(score)
        with open(scores_file, 'a') as f:
           f.write(json.dumps(instance) + "\n")
     
    image_score = 0 if len(generation_similarity_scores)==0 else sum(generation_similarity_scores) / len(generation_similarity_scores)
    results_dict = {'text_score': text_score, 'num_text': num_text, 'image_score': image_score, 'num_images': len(generation_similarity_scores)}
    
    metrics = {"mm_out": results_dict}
    save_path = f"{save_dir}/results/{save_name}.json"
    add_results_to_json(save_path, metrics)