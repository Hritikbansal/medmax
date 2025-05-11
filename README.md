# MedMax
MedMax: Mixed-Modal Instruction Tuning for Training Biomedical Assistants

[[Webpage](https://mint-medmax.github.io/)] [[Paper](https://arxiv.org/abs/2412.12661)] [[Train Dataset 🤗](https://huggingface.co/datasets/mint-medmax/medmax_data)] [[Eval Dataset 🤗](https://huggingface.co/datasets/mint-medmax/medmax_eval_data)] [[Model 🤗](https://huggingface.co/mint-medmax/medmax_7b)] [[Demo 🤗]](https://huggingface.co/spaces/mint-medmax/medmax-demo-v1.0)

<p align="center">
    <img src="static/logo.png" width="30%"> <br>.
</p>


## Abstract

Recent advancements in mixed-modal generative models have unlocked seamless multimodal information integration, opening transformative possibilities for biomedical AI in image analysis, diagnosis, and dataset creation. However, existing resources are limited data availability, narrow domain coverage, and restricted origins (e.g., medical papers). To address these gaps, we present MedMax, the first large-scale multimodal biomedical instruction-tuning dataset for mixed-modal foundation models. With 1.47 million instances, MedMax encompasses a diverse range of tasks, including multimodal content generation (interleaved image-text data), biomedical image captioning and generation, visual chatting, and report understanding. These tasks span diverse medical domains such as radiology and histopathology. Subsequently, we fine-tune a mixed-modal foundation model on the MedMax dataset, achieving significant performance improvements: a 26% gain over the base Chameleon model and an 18.3% improvement over GPT-4o across 12 downstream biomedical visual question-answering tasks. Additionally, we introduce a unified evaluation suite for biomedical tasks, providing a robust framework to guide the development of next-generation mixed-modal biomedical AI assistants.

<p align="center">
    <img src="static/main_result.png" width="80%"> <br>Main results.
</p>

## Installation

```
conda create -n medmax python=3.10
conda activate medmax
pip install -r requirements.txt
```

- The above command will install transformers too. However, we will use our custom transformers present in this repo. Hence, `pip uninstall transformers`.

## Custom Inference

First, install gradio through `pip install gradio==5.6.0`

Then, Use the following command to launch gradio demo

```
mkdir -p .gradio
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_TEMP_DIR=.gradio python demo.py -c <your checkpoint folder>
```


## Evaluation

#### Setup
Request access to the MedMax Evaluation Data at https://huggingface.co/datasets/mint-medmax/medmax_eval_data.

Once granted access, install the data to your desired directory `eval_data_dir`. Unzip the images folder in the cloned data repository.
```
cd <eval_data_dir>
git lfs install
git clone https://huggingface.co/datasets/mint-medmax/medmax_eval_data
cd medmax_eval_data
tar -xzvf images.tar.gz -C .
```

Open ended evaluations require an OpenAI API access key which should be entered in `evaluations/const.py` e.g.
```
OAI_KEY = <your key here>
```

Install the MedMax model checkpoint to a predetermined directory `ckpt_dir`
```
git lfs install
git clone https://huggingface.co/mint-medmax/medmax_7b
```
#### Evaluation

To run the evaluation suite for MedMax 7B run
```
CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval --ckpt <ckpt_dir>  --prompt_processor sft --eval_data_dir <eval_data_dir> --save_dir <output_location> --save_name <save_file_name>
```

To run the evaluation suite for Chameleon 7B run
```
CUDA_VISIBLE_DEVICES=0 python -m evaluation.eval --ckpt <ckpt_dir>  --prompt_processor chameleon --eval_data_dir <eval_data_dir> --save_dir <output_location> --save_name <save_file_name>
```


## Data setup

1. We provide the data on the huggingface datasets at [https://huggingface.co/datasets/mint-medmax/medmax_data](https://huggingface.co/datasets/mint-medmax/medmax_data).
2. The dataset is divided into two parts: (a) credential = 'YES' and (b) credential = 'NO'. In particular, we do not provide the `image_path` and `tokens` for `credential=YES` split.
3. If you want to train a model on the `credential=NO` split, you can directly skip to the finetuning section.
4. Use our instructions on the HF datasets' README to get the remaining image data in the `credential=YES` split. 

Specifically, we provide the instructions to get the `tokens` column once you access to the `image_path` for the remaining datasets.

1. Create a csv file with the absolute image paths in the `img_path` column.
2. Download the MedMax [checkpoint](https://huggingface.co/mint-medmax/medmax_7b) that comes with the VQGAN checkpoint. 
3. Run the following command:
```python
    CUDA_VISIBLE_DEVICES=0,1 torchrun -m --nproc-per-node=2 src.image_tokenization --input <path to a csv file> --output <path to save folder> --ckpt <path to medmax folder>
```
4. The code will generate `parquet` files with an additional column of `img_tokens`. 
5. Specifically, the elements of the `img_tokens` should range from `0-8191` (8192 tokens). 


## Tokenizing Multimodal Sequence

1. Create a jsonl file with the following elements:
```python
    text: the multimodal text with <image> placeholder (this is present in our original dataset)
    image_path: path to the image
    image_tokens: image tokens from the VQGAN tokenizer (as described in the previous section)
    source: (this is present in our original dataset)
    task: (this is present in our original dataset)
```
2. Run the following tokenization code:
```python
    python src/tokenization.py --input_file <input jsonl file> --tokenizer_file <tokenizer> --output_file <output jsonl filename>
```
3. You should use the `text_tokenized_modified.json` on huggingface - [https://huggingface.co/mint-medmax/medmax_7b/blob/main/tokenizer/text_tokenizer_modified.json](https://huggingface.co/mint-medmax/medmax_7b/blob/main/tokenizer/text_tokenizer_modified.json).
4. In our code, we add an offset of 4 to each element of the image tokens to map them to the corresponding vocab ID in the MedMax tokenizer. For example, a VQGAN token ID of 5678 corresponds to a MedMax token ID of 5682. Consequently, the updated tokens will reflect this mapping.

## Finetuning

1. Here, the users can finetune the [base model](https://huggingface.co/mint-medmax/anole_7b_hf) on the Medmax training data. Specifically, we will finetune the Anole model using HF codebase and then convert it to the Chameleon supported format to allow evaluations. 
2. Please download the base model in your local machine and also download the medmax training data. Remove the instances where `credential=yes` or get the tokens for the same using the instructions above. 
3. Eventually, we will just use the `tokens` element in each row of the jsonl data.
4. Run the following command for launching multi-gpu finetuning with Low-rank adaptation:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m training.train --train_data <train.jsonl> --val_data <val.jsonl> --ckpt <path to Anole-7b-v0.1-hf> --ds training/ds_config.json --output_dir <path to output dir> --epoch 3 --bs 1 --save_strategy steps --warmup_ratio 0.1 --name <exp name> --lr 1e-4 --bf16 --wandb --wandb_entity <name of wandb entity> 
```
5. At the end of the finetuning, you will have the trained LoRA adapters. Now, we will want to merge them to the original model to get the merged model. Run the following code:
```python
    CUDA_VISIBLE_DEVICES=1 python -m src.load_lora_and_merge --ckpt_path <path to output dir/checkpoint-number> --output_dir <path to the output dir for the merged model> --base_path <path to the Anole-7b-1.0-hf model>
```
6. Now, we have to convert the merged model to the chameleon's format so that we can use the original chameleon's [code](https://github.com/facebookresearch/chameleon) for inference. Run the following command:
```
    CUDA_VISIBLE_DEVICES=1 python -m src.bin_to_pth --trained_ckpt <path to the merged checkpoint> --original_ckpt <original Anole model in the chameleon's format> --new_ckpt <save folder with the chameleon's format>
```
7. Original Anole model in the chameleon's format is present in [https://huggingface.co/GAIR/Anole-7b-v0.1/tree/main](https://huggingface.co/GAIR/Anole-7b-v0.1/tree/main).

**Note:** During finetuning, there might be an error due to versioning of the transformers and deepspeed. To fix this, we comment this line that was throwing the error. 
```python
        # if compare_versions("transformers", "<", "4.33"):
            # from transformers.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config
        # else:
        from transformers.integrations import HfDeepSpeedConfig, unset_hf_deepspeed_config
```
in `"/opt/conda/envs/medmax/lib/python3.10/site-packages/accelerate/utils/dataclasses.py", line 1295`.


## Converting from Chameleon Format to Huggingface Format

1. We provide the Huggingface checkpoint of MedMax-7B at [https://huggingface.co/mint-medmax/medmax_7b_hf](https://huggingface.co/mint-medmax/medmax_7b_hf).
2. This checkpoint was obtained after running the following command on the MedMax checkpoint:
```python
    CUDA_VISIBLE_DEVICES=0 python -m transformers.models.chameleon.convert_chameleon_weights_to_hf --input_dir medmax_7b --model_size 7B --output_dir medmax_7b_hf
```
3. You can use the above command to convert your own finetuned checkpoints to the HF checkpoint.
4. The HF checkpoint can used with the HF's Chameleon inference code: [https://huggingface.co/docs/transformers/main/en/model_doc/chameleon](https://huggingface.co/docs/transformers/main/en/model_doc/chameleon).

## VQGAN Training

See [Instructions](vqgan/readme.MD)

## Acknowledgements

1. Anole: https://github.com/GAIR-NLP/anole (Model and Finetuning support)
2. Chameleon: https://github.com/facebookresearch/chameleon (Inference support)
3. HF Transformers: https://github.com/huggingface/transformers (Framework support)
4. VQGAN: https://github.com/CompVis/taming-transformers (VQGAN training support)
