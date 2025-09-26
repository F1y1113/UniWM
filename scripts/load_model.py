import requests
import torch
import math

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

def load_model(args, training_cfg):
    print("image_seq_length received:", args.image_seq_length)
    model_name = args.model

    model_ckpt_path = args.model_ckpt

    if model_name in ['anole']:
        image_token_num = args.image_seq_length

        # Use MemoryBankAnoleForConditionalGeneration for prediction tasks
        if args.use_memory_bank_inference and args.do_task_level_eval and not args.do_train:
            from model_utils.memory_bank_visualizer import MemoryBankAnoleForConditionalGeneration
            print("Loading MemoryBankAnoleForConditionalGeneration for prediction task")
            model = MemoryBankAnoleForConditionalGeneration.from_pretrained(
                "leloy/Anole-7b-v0.1-hf",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                codebook_sim="mse"
            )
        else:
            from model_utils.wrapped_visualizer import AnoleforConditionalGeneration
            model = AnoleforConditionalGeneration.from_pretrained(
                "leloy/Anole-7b-v0.1-hf",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                codebook_sim="mse"
            )

        # NEW: Conditionally load processor from ckpt in inference to match extended vocab size
        is_inference_only = (args.do_single_step_eval or args.do_task_level_eval or args.do_rollout_eval) and not args.do_train and model_ckpt_path
        if is_inference_only:
            print("Loading processor from checkpoint for inference (to match vocab size).")
            processor = AutoProcessor.from_pretrained(model_ckpt_path, image_seq_length=image_token_num)
        else:
            print("Loading processor from base for training.")
            processor = AutoProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf", image_seq_length=image_token_num)

        # NEW: Always resize base model if tokenizer size doesn't match (handles inference case)
        tokenizer_size = len(processor.tokenizer)
        model_embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f"Tokenizer size: {tokenizer_size}. Base model embedding size: {model_embedding_size}")
        if model_embedding_size != tokenizer_size:
            print("Resizing model embeddings to match tokenizer size.")
            model.resize_token_embeddings(tokenizer_size)

        # NEW: Set padding_side attribute (required for correct generation)
        processor.tokenizer.padding_side = "left"

        # NEW: Monkey patch to ignore 'padding_side' kwarg in tokenizer (workaround for library bug)
        def patched_batch_encode_plus(self, *args, **kwargs):
            kwargs.pop('padding_side', None)  # Remove if library passes it
            return self.__original_batch_encode_plus(*args, **kwargs)

        if not hasattr(processor.tokenizer, '__original_batch_encode_plus'):
            processor.tokenizer.__original_batch_encode_plus = processor.tokenizer._batch_encode_plus
            processor.tokenizer._batch_encode_plus = patched_batch_encode_plus.__get__(processor.tokenizer)

        # Rest of your original code unchanged
        processor.image_processor.size = {"shortest_edge": 448}
        processor.image_processor.crop_size = {
            "height": 448,
            "width": 448
        }

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
        model.model.vqmodel.quantize.quant_state_dims = [
            model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
        ] * 2

        args.sketch_resolution = model.model.vqmodel.config.resolution
        model.sketch_resolution = (args.sketch_resolution, args.sketch_resolution)
        model.image_token_num = image_token_num

        model.get_vis_codebook_sim()

        from peft import LoraConfig, get_peft_model
        from peft.peft_model import PeftModel

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],
        )
        lora_model = get_peft_model(model, config)

        if is_inference_only:
            lora_model = PeftModel.from_pretrained(model, model_ckpt_path, is_trainable=False)

        return {
            'processor': processor,
            'model': lora_model
        }
    else:
        raise ValueError("Unsupported model type. ")