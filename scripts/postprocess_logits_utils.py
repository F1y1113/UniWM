import torch
import torch.nn as nn

from typing import Optional, Literal, List, Tuple

def split_token_sequence(
    tokens: torch.LongTensor,
    image_seq_length: int, 
    boi: int,
    eoi: int,
    max_length: int,
    pad_token_id: int
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

    generated_imgs = []
    generated_texts = []
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == image_seq_length
            generated_imgs.append(seg_tokens)
        else:
            assert seg_type == "text_seg"
            generated_texts.append(seg_tokens.view(-1))

    text_tokens = torch.cat(generated_texts)
    if max_length > text_tokens.shape[-1]:
        text_tokens = torch.cat((text_tokens, torch.full((max_length-text_tokens.shape[-1],), fill_value=pad_token_id, device=text_tokens.device))).unsqueeze(0)
    elif max_length < text_tokens.shape[-1]:
        text_tokens = text_tokens.unsqueeze(0)[:, :max_length]
    else:
        text_tokens = text_tokens.unsqueeze(0)
    return {
        "texts": text_tokens,
        "images": generated_imgs if len(generated_imgs) != 0 else None
    }