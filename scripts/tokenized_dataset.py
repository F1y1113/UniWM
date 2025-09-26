import os
import json
import torch
import copy

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AnoleTokenizedDataset(Dataset):
    def __init__(self,
                 dataset,
                 model,
                 processor,
                 split,
                 input_max_length,
                 label_max_length,
                 input_format,
                 **kwargs):
        self.model = model
        self.processor = processor
        self.split = split
        self.dataset = dataset

        self.input_max_length = input_max_length
        self.label_max_length = label_max_length

        format_json = os.path.join('prompt', input_format + '.json')
        with open(format_json) as f:
            template = json.load(f)

        self.input_template = template['input_prompt']
        self.output_template = template['output_prompt']
        
        self.label_processor = copy.deepcopy(self.processor)
        self.label_processor.tokenizer.padding_side = "right"

        self.processor.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_text = item['text']
        input_img = item['input_img'].convert("RGB")
        label_text = item['labels']
        label_img = item['label_img'].convert("RGB")

        instruction = item['instruction']
    
        input_str = (self.input_template.replace('<INPUT>', input_text.strip())).replace("<INSTRUCTION>", instruction)

        label_str = self.output_template.replace("<LABEL>", label_text.strip()).replace("<sketch>", "<image>")

        if self.split in ['train']:
            tokenized_input = self.processor(
                [input_str],
                images=[input_img],
                padding="max_length",
                return_tensors="pt",
                max_length=self.input_max_length
            )
            
            tokenized_label = self.label_processor(
                [label_str],          # for padding
                images=[label_img],
                padding="max_length",
                return_tensors="pt",
                max_length=self.label_max_length,
            )
            tokenized_label = {k: v[:, 1:] if k in ['input_ids', 'attention_mask'] else v for k, v in tokenized_label.items()}     # omit <s> starting token
            tokenized_label['input_ids'][tokenized_label['input_ids'] == self.model.config.image_token_id] = self.model.model.model.get_image_tokens(tokenized_label['pixel_values'].to(self.model.device).to(torch.bfloat16)).to(torch.int64).to(tokenized_label['input_ids'].device)

            label_ids = torch.cat((torch.full(tokenized_input['input_ids'].shape, -100), tokenized_label['input_ids']), 1)
            label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100
            
            tokenized_input['input_ids'] = torch.cat((tokenized_input['input_ids'], tokenized_label['input_ids']), 1)
            tokenized_input['attention_mask'] = torch.cat([tokenized_input.pop('attention_mask'), tokenized_label["attention_mask"]], 1)
            if 'image_embeds_position_mask' in tokenized_input:
                tokenized_input['image_embeds_position_mask'] = torch.cat((tokenized_input['image_embeds_position_mask'], torch.zeros(tokenized_label['input_ids'].shape)), 1)

            tokenized_input['pixel_values'] = tokenized_input['pixel_values'].to(torch.bfloat16)

            return {
                **tokenized_input,
                "labels": label_ids,
                "img_label": label_img
            }
        
        else:
            tokenized_input = self.processor(
                text=input_str,
                images=input_img,
                padding="max_length",
                return_tensors="pt",
                max_length=self.input_max_length
            )
            tokenized_input['pixel_values'] = tokenized_input['pixel_values'].to(torch.bfloat16)
            return {
                **tokenized_input
            }