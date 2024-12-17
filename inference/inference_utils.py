
import os
import torch
import uuid
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
import json
import numpy as np
import openai

def load_chameleon(ckpt_path):

    ckpt_path = Path(ckpt_path)

    print(ckpt_path)

    MODEL_7B_PATH = ckpt_path / "models" / "7b"
    TOKENIZER_TEXT_PATH = ckpt_path / "tokenizer" / "text_tokenizer.json"
    TOKENIZER_IMAGE_PATH = ckpt_path / "tokenizer" / "vqgan.ckpt"
    TOKENIZER_IMAGE_CFG_PATH = ckpt_path / "tokenizer" / "vqgan.yaml"

    # Load Chameleon model
    model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )
    
    return model

# https://github.com/GAIR-NLP/anole/blob/main/inference.py
def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments


# https://github.com/GAIR-NLP/anole/blob/main/inference.py
def chameleon_generate(model, content, modality="text", task="text-gen", sft=True, max_gen_len=None, temp=None, greedy=True, save_dir="evaluation/outputs/inference"): 
    
    
    # Generate options
    options = Options()
    if max_gen_len is not None:
        options.max_gen_len = max_gen_len
    if temp is not None:
        options.txt.temp = temp
        
    options.txt.greedy = greedy
        
    if type(modality) == list:
        input_segs = [{"type": m, "content": c} for m, c in zip(modality, content)]
    else:
        input_segs = [{"type": modality, "content": content}]

    batch_prompt_ui = [[]]
    for input_seg in input_segs:
        if input_seg["type"] == "text":
            batch_prompt_ui[0] += [{"type": "text", "value": input_seg["content"]}]
        elif input_seg["type"] == "image":
            abs_path: Path = os.path.abspath(input_seg["content"])
            batch_prompt_ui[0] += [{"type": "image", "value": f"file:{abs_path}"}]
        else:
            assert input_seg["type"] == "sentinel"
            batch_prompt_ui[0] += [{"type": "sentinel", "value": input_seg["content"]}]
            
    
    if sft:
        batch_prompt_ui[0] += [{"type": "sentinel", "value": "<END-OF-TURN>"}]
    
    if task == 'text-gen':
        options.img = False
    
    if task == 'image-gen':
        options.txt = False
        batch_prompt_ui[0] += [{"type": "sentinel", "value": "<START-OF-IMAGE>"}]

    # generate
    tokens: torch.LongTensor = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )
    if task == 'image-gen':
        segments = [('image_seg', tokens.reshape(1, -1))]
    else:
        # split
        boi, eoi = model.vocab.begin_image, model.vocab.end_image   # 8197(boi), 8196(eoi)
        segments = split_token_sequence(tokens, boi, eoi)


    outputs = []
    # decode
    os.makedirs(save_dir, exist_ok=True)
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == 1024
            img: Image = model.decode_image(seg_tokens)[0]
            image_path = os.path.join(save_dir, f"{uuid.uuid4()}_{seg_id}.png")
            img.save(image_path)
            outputs.append(image_path)
        else:
            assert seg_type == "text_seg"
            decoded_text = model.decode_text(seg_tokens)[0]
            outputs.append(decoded_text)
            
    return outputs