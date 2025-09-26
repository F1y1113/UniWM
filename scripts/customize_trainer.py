import os
import json
import pickle
import torch
import importlib.metadata
import re

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from transformers import Trainer, Seq2SeqTrainer

from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

import time
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from packaging import version
from torch import nn

from PIL import Image

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.file_utils import is_datasets_available

from transformers.trainer_utils import EvalPrediction

from transformers.trainer_utils import PredictionOutput, speed_metrics

from scripts.training_arguments import WrappedSeq2SeqTrainingArguments
from scripts.postprocess_logits_utils import split_token_sequence
from scripts.action_utils import generate_bin_tokens, extract_bin_values, DATASET_RANGES, DEFAULT_RANGES, action_to_text
from model_utils.memory_bank_visualizer import MemoryBankAnoleForConditionalGeneration
from scripts.prompt_builder import build_action_prompt, build_joint_prompt, build_viz_prompt
from scripts.metrics import coords_to_evo_traj, eval_ate_rpe, ImageMetricsCalculator


_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


from torch.utils.data import DataLoader, Dataset
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    EvalLoopOutput,
    PredictionOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size
)
from transformers.utils import (
    is_peft_available,
    logging,
)


logger = logging.get_logger(__name__)


class MetricEvalPrediction(NamedTuple):
    predictions: List[dict]
    items: List[dict]
    sketches: Union[List[str], np.ndarray]     # the file path for the sketch


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    sketches: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


if is_peft_available():
    from peft import PeftModel


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

def get_concat_h(im1, im2):
    # resize the image first wo the same height
    height = im1.height
    ratio = im1.height / im2.height
    im2_width = int(im2.width * ratio)
    im2 = im2.resize((im2_width, height))

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class MemoryBankManager:
    def __init__(self, model: nn.Module, trainer, use_memory_bank_inference: bool):
        self.model = model
        self.trainer = trainer # To access is_world_process_zero
        self.is_enabled = use_memory_bank_inference
        self.current_step = 0
    
    def is_world_process_zero(self):
        """Helper method to access the trainer's is_world_process_zero."""
        return self.trainer.is_world_process_zero()

    def setup_for_trajectory(self, traj_dirname: str):
        """Prepares the memory bank for a new trajectory."""
        if not self.is_enabled:
            return

        # Reset memory bank for each trajectory to ensure independence
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.is_world_process_zero():
                print(f"  Intra-step memory bank reset for trajectory {traj_dirname}")
        elif hasattr(self.model, 'memory_bank_initialized'):
            # Fallback: manually reset memory bank state
            self.model.memory_bank_initialized = False
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                for layer in self.model.model.model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'reset_memory_bank'):
                        layer.self_attn.reset_memory_bank()
            if self.is_world_process_zero():
                print(f"  Fallback intra-step memory bank reset for trajectory {traj_dirname}")
        
        # Reset global cross-step memory bank for each trajectory
        if hasattr(self.model, 'reset_global_memory_bank'):
            self.model.reset_global_memory_bank()
            if self.is_world_process_zero():
                print(f"  Cross-step memory bank reset for trajectory {traj_dirname}")
        
        # Enable global memory bank functionality
        if hasattr(self.model, 'enable_global_memory_bank'):
            self.model.enable_global_memory_bank()
            if self.is_world_process_zero():
                print(f"  Global memory bank enabled for trajectory {traj_dirname}")
        
        # Enable memory bank functionality if available
        if hasattr(self.model, 'enable_memory_bank'):
            self.model.enable_memory_bank()
            if self.is_world_process_zero():
                print(f"  Memory bank functionality enabled for trajectory {traj_dirname}")
        
        self.current_step = 0

    def start_new_step(self):
        """Prepares for a new step within a trajectory."""
        if not self.is_enabled:
            return
        
        self.current_step += 1
        if self.is_world_process_zero():
            print(f"\n=== Step {self.current_step} Action Prediction Substep ===")
        
        # Reset intra memory bank for action prediction substep (but keep cross memory bank)
        if hasattr(self.model, 'reset_memory_bank'):
            self.model.reset_memory_bank()
            if self.is_world_process_zero():
                print(f"  Step {self.current_step}: intra memory bank reset for action prediction")
    
    def get_action_kwargs(self, action_inputs, action_gen_kwargs, step):
        if not self.is_enabled:
            return action_gen_kwargs
        
        # Check for memory bank initialization (third pair of 8197 and 8196 tokens)
        input_ids_list = action_inputs['input_ids'][0].tolist()
        # Always try to initialize memory bank for each step (intra-step memory bank)
        if hasattr(self.model, 'initialize_memory_bank') and not getattr(self.model, 'memory_bank_initialized', False):
            
            # Count pairs of 8197 and 8196 tokens
            pairs_count = 0
            i = 0
            while i < len(input_ids_list) - 1:
                if input_ids_list[i] == 8197:
                    for j in range(i + 1, len(input_ids_list)):
                        if input_ids_list[j] == 8196:
                            pairs_count += 1
                            i = j
                            break
                    else:
                        break
                i += 1
              
            # Initialize memory bank if we have at least 3 pairs, or fallback to any image tokens
            should_initialize = False
            init_method = ""
            
            if pairs_count >= 3:
                should_initialize = True
                init_method = f"special token pairs (found {pairs_count})"
            
            if should_initialize:
                if self.is_world_process_zero():
                    print(f"  Step {step+1}: Initializing memory bank using {init_method}")

                # try:
                self.model.initialize_memory_bank(
                    input_ids=action_inputs['input_ids'],
                    pixel_values=action_inputs['pixel_values'],
                    attention_mask=action_inputs['attention_mask']
                )
                if self.is_world_process_zero():
                    print(f"  Step {step+1}: Memory bank initialization completed successfully")
                    # Print memory bank storage details if available
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                        for layer_idx, layer in enumerate(self.model.model.model.layers):
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                                if layer.self_attn.stored_keys is not None:
                                    print(f"    - Layer {layer_idx}: Stored keys shape: {layer.self_attn.stored_keys.shape}")
                                    print(f"    - Layer {layer_idx}: Stored values shape: {layer.self_attn.stored_values.shape}")
                                    print(f"    - Layer {layer_idx}: Memory bank storage size: {layer.self_attn.stored_keys.numel() + layer.self_attn.stored_values.numel()} elements")
                # except Exception as e:
                #     if self.is_world_process_zero():
                #         print(f"  Warning: Memory bank initialization failed: {e}")
            else:
                if self.is_world_process_zero():
                    print(f"  Step {step+1}: Warning - No suitable tokens found for memory bank initialization")
        
        # Print memory bank usage details before generation
        if hasattr(self.model, 'memory_bank_initialized') and self.model.memory_bank_initialized:
            if self.is_world_process_zero():
                print(f"  Step {step+1}: Using memory bank for action generation")

                # Print stored K,V sizes if available
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                            if layer.self_attn.stored_keys is not None:
                                print(f"    - Layer {layer_idx}: Using stored K shape: {layer.self_attn.stored_keys.shape}")
                                print(f"    - Layer {layer_idx}: Using stored V shape: {layer.self_attn.stored_values.shape}")
        
        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            # Two-phase action generation with memory bank
            action_gen_kwargs_with_memory = action_gen_kwargs.copy()
            # Remove any existing memory bank parameters to avoid conflicts
            action_gen_kwargs_with_memory.pop('use_memory_bank', None)
            action_gen_kwargs_with_memory.pop('is_memory_bank_init', None)
            action_gen_kwargs_with_memory.pop('current_step', None)
            action_gen_kwargs_with_memory.pop('current_substep', None)
            action_gen_kwargs_with_memory.pop('use_global_memory_bank', None)
            
            # Use global memory bank if we have previous steps (current_step > 1)
            use_global_mb = self.current_step > 1
            
            # Phase 1: Initialize memory bank (dummy generation to extract K,V)
            init_kwargs = action_gen_kwargs_with_memory.copy()
            init_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': True,  # Initialize memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': False,  # Don't use global during init
                'max_new_tokens': 1  # Minimal generation for initialization
            })
            
            if self.is_world_process_zero():
                print(f"  Step {step+1}: Memory bank initialization phase")
            _ = self.model.generate(**action_inputs, **init_kwargs)
            
            # Phase 2: Actual action generation using initialized memory bank
            gen_kwargs = action_gen_kwargs_with_memory.copy()
            gen_kwargs.update({
                'use_memory_bank': True,
                'is_memory_bank_init': False,  # Use existing memory bank
                'current_step': self.current_step,
                'current_substep': 'action',
                'use_global_memory_bank': use_global_mb
            })
            
            if self.is_world_process_zero():
                print(f"  Step {step+1}: Action generation using memory bank (global: {use_global_mb})")
            return gen_kwargs
            
    def get_viz_kwargs(self, viz_inputs, viz_gen_kwargs, step):
        if not self.is_enabled:
            return viz_gen_kwargs
        if self.is_world_process_zero():
            print(f"\n=== Step {self.current_step} Visualization Substep ===")
        
        
        with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
            # Enable memory bank for visualization generation
            viz_gen_kwargs_with_memory = viz_gen_kwargs.copy()
            # Use global memory bank for visualization (always available since we're in step >= 1)
            use_global_mb_viz = self.current_step >= 1
            viz_gen_kwargs_with_memory.update({
                'use_memory_bank': True,
                'is_memory_bank_init': False,  # Use existing memory bank for visualization
                'current_step': self.current_step,
                'current_substep': 'visualization',
                'use_global_memory_bank': use_global_mb_viz
            })
            return viz_gen_kwargs_with_memory

    def store_step_memory(self):
        """Stores the current step's K,V pairs into the global memory bank."""
        if not self.is_enabled:
            return
            
        # Store current step's intra-step K,V to global cross-step memory bank
        # This happens after both action prediction and visualization substeps are completed
        if hasattr(self.model, 'store_to_global_memory_bank'):
            self.model.store_to_global_memory_bank(self.current_step)
            if self.is_world_process_zero():
                print(f"  Step {self.current_step}: Stored intra-step K,V to global memory bank")
    
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(self.model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'global_stored_keys'):
                            if len(layer.self_attn.global_stored_keys) > 0:
                                print(f"    - Layer {layer_idx}: Global memory bank now has {len(layer.self_attn.global_stored_keys)} steps")
                                print(f"    - Layer {layer_idx}: Latest stored K shape: {layer.self_attn.global_stored_keys[-1].shape}")
                                print(f"    - Layer {layer_idx}: Latest stored V shape: {layer.self_attn.global_stored_values[-1].shape}")
                                break  # Only print for first layer to avoid spam



class CustomizeSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self, 
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            image_loss_func: Optional[torch.nn.Module],
            action_cfg: Optional[Dict] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir

        self.image_loss_func = image_loss_func

        if action_cfg:
            self.action_cfg = action_cfg
        else:
            print("[WARNING] action_cfg not provided to Trainer. Using default values for loss calculation.")
            self.action_cfg = {
                'min_dxy': -2.46, 'max_dxy': 2.46,
                'min_dyaw': -2.82, 'max_dyaw': 2.82,
                'bin_step': 0.01
            }
    
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs,
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        gen_kwargs = gen_kwargs.copy()

        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens
        
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        
        if hasattr(self.args, 'customize_gen_stopping_criteria'):
            if self.args.customize_gen_stopping_criteria:
                gen_kwargs['stopping_criteria'] = self.args.customize_gen_stopping_criteria
        
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        
        if self.compute_metrics:
            print("self.compute_metrics exists, type:", type(self.compute_metrics))


        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch)
            )
            summary = self.compute_metrics(eval_preds, section="dev", finish=True)
            # output.metrics.update(summary)
            if summary is not None:
                output.metrics.update(summary)
            else:
                print("[Warning] compute_metrics returned None — skipping metric update.")

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            predict_task: Optional[str] = None,
            **gen_kwargs
            # max_length: Optional[int] = None,
            # max_time: Optional[int] = None,
            # num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        gen_kwargs = gen_kwargs.copy()

        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_new_tokens is not None
        ):
            gen_kwargs["max_new_tokens"] = self.args.generation_max_new_tokens

        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        
        if test_examples and test_examples[0].get('train_task') == 'task_level_evaluation':
            if predict_task == "task_level_evaluation":
                metrics = self.task_level_evaluation_loop(test_dataset, test_examples, metric_key_prefix)
            if predict_task == "rollout_evaluation":
                metrics = self.run_gt_action_rollout(test_dataset, test_examples, metric_key_prefix)
            return PredictionOutput(predictions=None, sketches=None, label_ids=None, metrics=metrics)

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics


        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples,
                output.predictions,
                metric_key_prefix
            )
            output.metrics.update(self.compute_metrics(eval_preds, section="test", finish=True))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def _post_process_function(
            self, 
            examples: Dataset, 
            predictions: np.ndarray, 
            # sketches: np.ndarray, 
            stage: str
    ) -> MetricEvalPrediction:
        # assert isinstance(examples, Dataset)
        if self.args.local_rank <= 0:
            print("*"*20)
            print(len(predictions))
            print(len(examples))
            print("*"*20)
        
        tokens = []
        sketches = []
        for r_ids in predictions:
            generated_results = split_token_sequence(
                tokens=torch.tensor(r_ids).unsqueeze(0).to(self.model.device), 
                image_seq_length=self.model.image_token_num,
                boi=self.model.config.boi_token_id, 
                eoi=self.model.config.eoi_token_id,
                max_length=predictions.shape[-1],
                pad_token_id=self.model.config.pad_token_id
            )
            tokens.append(generated_results['texts'])
            if generated_results["images"]:
                generated_imgs = torch.cat(generated_results["images"], dim=0).to(self.model.device)
                generated_imgs = self.model.decode_image_tokens(generated_imgs)
                generated_imgs = self.tokenizer.postprocess_pixel_values(generated_imgs)
            else:
                generated_imgs = None
            sketches.append(generated_imgs)
        
        predictions = torch.cat(tokens, dim=0)

        predictions[predictions == -100] = self.tokenizer.tokenizer.pad_token_id

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)

        predictions = [i.split("<reserved08706>")[0] for i in predictions]

        sketch_dir = f"{self.args.output_dir}/sketch_{stage}"

        # Save locally.
        if self.is_world_process_zero():
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_paths": examples[idx]['label_img_paths'],
                            "input_img_paths": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        
            if stage.startswith("eval"):
                
                if not os.path.exists(sketch_dir):
                    os.mkdir(sketch_dir)
                
                sketch_files = []
                for idx in range(len(examples)):
                    sketches_per_item = sketches[idx]
                    sketch_files_per_item = []
                    if sketches_per_item is not None:
                        for i in range(len(sketches_per_item)):
                            file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                            tensor_img = sketches_per_item[i, :, :, :]
                            print(f"[DEBUG] tensor_img.shape: {tensor_img.shape}")  # (C, H, W)

                            np_img = tensor_img.cpu().detach().to(torch.uint8).numpy()
                            np_img = np.transpose(np_img, (1, 2, 0))  # (H, W, C)
                            print(f"[DEBUG] np_img.shape: {np_img.shape}")

                            img = Image.fromarray(np_img.astype(np.uint8))
                            print(f"[DEBUG] PIL image size: {img.size}")  # (W, H)

                            if len(examples[idx]["label_imgs"]) != 0:
                                concat_img = get_concat_h(im1=examples[idx]["label_imgs"][-1], im2=img)
                                concat_img = get_concat_h(im1=examples[idx]['input_imgs'][-1], im2=concat_img)
                            else:
                                concat_img = img
                            concat_img.save(file_path)

                            sketch_files_per_item.append(file_path)

                    sketch_files.append(sketch_files_per_item)

        # Save to wandb.
        if self.wandb_run_dir and self.is_world_process_zero():
            with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(
                        **{
                            "idx": examples[idx]['idx'],
                            "prediction": predictions[idx],
                            "text": examples[idx]["input_text"],
                            "labels": examples[idx]['label_text'],
                            "predicted_sketch_paths": [os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png") for i in range(len(sketches[idx]))] if sketches[idx] is not None else None, 
                            "label_img_path": examples[idx]['label_img_paths'],
                            "input_img_path": examples[idx]['input_img_paths']
                        }
                        ) for idx in range(len(examples))],
                    f,
                    indent=4,
                )
        if not self.is_world_process_zero():
            sketch_files = []
            for idx in range(len(examples)):
                sketches_per_item = sketches[idx]
                sketch_files_per_item = []
                if sketches_per_item is not None:
                    for i in range(len(sketches_per_item)):
                        file_path = os.path.join(sketch_dir, rf"{str(examples[idx]['idx'])}_{i}_{stage}.png")
                        sketch_files_per_item.append(file_path)
                sketch_files.append(sketch_files_per_item)

        return MetricEvalPrediction(predictions=predictions, sketches=sketches, items=[examples[idx] for idx in range(len(examples))])

    def _compute_metrics(self, eval_prediction: MetricEvalPrediction, section, finish=False) -> dict:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, eval_prediction.sketches, section, finish=finish)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        if "labels" in inputs:
            labels = inputs.pop("labels")
            if "img_label" in inputs:
                img_label = inputs.pop("img_label")
            else:
                img_label = None
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # text-wise loss
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or model_name.endswith('ConditionalGeneration'):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

            min_dxy = self.action_cfg['min_dxy']
            max_dxy = self.action_cfg['max_dxy']
            min_dyaw = self.action_cfg['min_dyaw']
            max_dyaw = self.action_cfg['max_dyaw']
            bin_step = self.action_cfg['bin_step']

            dx_tokens = generate_bin_tokens("dx", min_dxy, max_dxy, bin_step)
            dy_tokens = generate_bin_tokens("dy", min_dxy, max_dxy, bin_step)
            dyaw_tokens = generate_bin_tokens("dyaw", min_dyaw, max_dyaw, bin_step)

            # 2. Convert to token ids
            dx_ids = set(self.tokenizer.tokenizer.convert_tokens_to_ids(dx_tokens))
            dy_ids = set(self.tokenizer.tokenizer.convert_tokens_to_ids(dy_tokens))
            dyaw_ids = set(self.tokenizer.tokenizer.convert_tokens_to_ids(dyaw_tokens))

            # 3. Prepare shifted logits and labels
            logits = outputs.logits[:, :-1, :].contiguous().view(-1, outputs.logits.shape[-1])
            labels_shifted = labels[:, 1:].contiguous().view(-1)

            stop_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids("stop")
            is_stop = labels_shifted == stop_token_id
            stop_loss = (
                nn.CrossEntropyLoss()(logits[is_stop], labels_shifted[is_stop])
                if is_stop.any()
                else torch.tensor(0.0, device=logits.device)
            )

            # 4. Helper to compute loss
            def compute_bin_ce(bin_ids):
                mask = torch.isin(labels_shifted, torch.tensor(list(bin_ids), device=labels.device))
                if mask.any():
                    return nn.CrossEntropyLoss()(logits[mask], labels_shifted[mask])
                else:
                    return None

            dx_loss = compute_bin_ce(dx_ids)
            dy_loss = compute_bin_ce(dy_ids)
            dyaw_loss = compute_bin_ce(dyaw_ids)

            # 5. Combine with equal weights or customize
            loss_components = [l for l in [dx_loss, dy_loss, dyaw_loss] if l is not None]
            if loss_components:
                bc_loss = sum(loss_components) / len(loss_components)
  
                loss += bc_loss
                if self.state.global_step == self._globalstep_last_logged and self.state.global_step != 0:
                    self.log({"bc_loss": float(bc_loss)})

        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
 
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
 
        image_mask = torch.isin(labels, torch.tensor(unwrapped_model.model.bpe_indices).to(labels.device))

        if torch.any(image_mask) and self.image_loss_func:
            # use image_mask to retrieve image tokens from labels and logits distribution from output.logits as well
            image_labels = labels[image_mask]
            image_logits = outputs.logits[:, :-1, :][image_mask[:, 1:], :]

            # for image tokens in the labels, we use model.model.model.convert_bpe2img_tokens to convert it back to visual token indices
            vis_img_tokens = unwrapped_model.model.model.convert_bpe2img_tokens(image_labels)
            # for logits distributions from outputs.logits, we retrieve the corresponding indices from 60k dimensions 1) using torch matmul or 2) just retrieve
            image_probs = torch.nn.functional.softmax(image_logits[:, unwrapped_model.model.bpe_indices], dim=-1)

            label_one_hot = torch.nn.functional.one_hot(vis_img_tokens.reshape(-1).to(torch.int64), num_classes=unwrapped_model.model.model.vqmodel.quantize.embedding.weight.shape[0]).to(torch.bfloat16)
            label_sim_matrix = torch.matmul(label_one_hot.to(unwrapped_model.device), unwrapped_model.model.codebook_sim_matrix)
            discrepancy_loss = torch.mean(torch.sum(label_sim_matrix * image_probs.to(torch.bfloat16), -1))
            # print(f"[DEBUG] discrepancy_loss: {discrepancy_loss}")

            loss += discrepancy_loss
            
            # log the image loss every logging step in addition to total loss
            if self.state.global_step == self._globalstep_last_logged and self.state.global_step != 0:
                self.log({"discrepancy_loss": float(discrepancy_loss)})


        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        if "pixel_values" in inputs:
            model_dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

            if inputs["pixel_values"].ndim == 5:
                # [1, 3, 3, 448, 448] -> [1, 3, 448, 448]
                inputs["pixel_values"] = inputs["pixel_values"][:, 0, :, :, :]

        if hasattr(self.model, "image_decoder"):
            generated_tokens, generated_sketch = self.model.generate(**inputs, **gen_kwargs)
        else:
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)
            generated_sketch = None
        
        print(f"  Generated tokens: {generated_tokens}")

        if hasattr(self.model, "image_postprocess"):
            if self.model.image_postprocess and generated_sketch is not None:
                generated_sketch["sketch"] = self.tokenizer.postprocess_pixel_values(generated_sketch["sketch"])
        
        if hasattr(generated_tokens, "sequences"):
            generated_tokens = generated_tokens.sequences

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(
                    model=self.model,
                    inputs=inputs,
                    return_outputs=False
                )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        if generated_sketch is not None:
            return loss, (generated_tokens, generated_sketch), labels
        else:
            return loss, generated_tokens, labels
    

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # comment it for now
            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)
        
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # calculate the metric in a post-hoc way
            if self.args.batch_eval_metrics:

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs

                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        if metrics is None:
            metrics = {}

        # # Compute metrics if a compute_metrics function is provided
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     metrics = self.compute_metrics(
        #         EvalPrediction(predictions=all_preds, label_ids=all_labels)
        #     )

        return EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=num_samples
            )

    def _sanitize_generated_image(self, pil_img: Image) -> Image:
        if pil_img is None:
            return None
        inputs = self.tokenizer(text="<image>", images=pil_img, return_tensors="pt").to(self.args.device)
        pixel_values = inputs['pixel_values'].to(self.model.dtype)
        
        with torch.no_grad():
            img_tokens = self.model.model.model.get_image_tokens(pixel_values)
            reconstructed_pixels = self.model.model.decode_image_tokens(img_tokens)
        
        processed_pixels = self.tokenizer.postprocess_pixel_values(reconstructed_pixels)[0]
        np_img = processed_pixels.cpu().numpy().transpose(1, 2, 0)
        sanitized_img = Image.fromarray(np_img.astype(np.uint8))
        
        return sanitized_img

    def _parse_and_save_image(self, generated_tokens, model, tokenizer, save_path):
        if not isinstance(generated_tokens, torch.Tensor):
            raise TypeError(f"generated_tokens must be a torch.Tensor, but got {type(generated_tokens)}")

        if generated_tokens.dim() == 1:
            tokens_for_split = generated_tokens.unsqueeze(0)
        elif generated_tokens.dim() == 2:
            if generated_tokens.shape[0] != 1:
                tokens_for_split = generated_tokens[0].unsqueeze(0)
            else:
                tokens_for_split = generated_tokens
        else:
            raise ValueError(f"Unsupported shape for generated_tokens: {generated_tokens.shape}. Expected 1D or 2D tensor.")
        
        generated_output = split_token_sequence(
            tokens=tokens_for_split,
            image_seq_length=model.image_token_num,
            boi=model.config.boi_token_id,
            eoi=model.config.eoi_token_id,
            max_length=tokens_for_split.shape[-1],
            pad_token_id=tokenizer.tokenizer.pad_token_id,
        )
        predicted_image_tokens = generated_output['images']

        if predicted_image_tokens and len(predicted_image_tokens) > 0:
            with torch.no_grad():
                decoded_img = model.decode_image_tokens(predicted_image_tokens[0])
            processed_img = tokenizer.postprocess_pixel_values(decoded_img)[0]
            np_img = processed_img.cpu().numpy().transpose(1, 2, 0)
            new_obs_img = Image.fromarray(np_img.astype(np.uint8))
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                new_obs_img.save(save_path)
            return new_obs_img
        print(f"  Generated failed visualization tokens: {generated_tokens}")
        return None

    
    def _process_single_trajectory(self, raw_item: Dict, output_dir: str, model: Any,
                                   action_gen_kwargs: Dict, viz_gen_kwargs: Dict,
                                   manager: MemoryBankManager) -> Dict[str, Any]:
        # --- 1. Initialization ---
        traj_dirname = os.path.basename(os.path.dirname(raw_item['input_img_paths'][0]))
        if self.is_world_process_zero():
            print(f"\n--- Processing Trajectory {traj_dirname} ---")
        
        manager.setup_for_trajectory(traj_dirname)

        print(raw_item['train_task'])
        traj_output_dir = os.path.join(output_dir, traj_dirname)
        os.makedirs(traj_output_dir, exist_ok=True)

        start_pose = raw_item['coords'][0]
        current_pose = list(start_pose)  # [x, y, yaw]
        predicted_coords = [list(start_pose)]

        gt_coords_arr = np.array(raw_item['coords'])
        gt_goal_pos = gt_coords_arr[-1, :2] if len(gt_coords_arr) > 1 else None

        start_img = raw_item['input_imgs'][0].convert("RGB").resize((256, 256))
        goal_img = raw_item['input_imgs'][1].convert("RGB").resize((256, 256))
        current_observation = self._sanitize_generated_image(start_img)
        
        start_pose_str = f"Starting Point Coordinate: x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, yaw={start_pose[2]:.3f}\n"

        # --- 2. Step-by-step Loop ---
        current_trajectory_steps = []
        max_steps = 10

        all_actions = []
        all_decoded = []

        def _get_dataset_ranges(image_path):
            """ Determine the dataset ranges based on the image path."""
            for name, ranges in DATASET_RANGES.items():
                if f"/{name}/" in image_path.replace("\\", "/"):
                    return ranges
            return DEFAULT_RANGES

        ranges = _get_dataset_ranges(raw_item['input_img_paths'][0])
        print(f"  Using ranges: {ranges} for trajectory {raw_item['input_img_paths'][0]}")
        dxy_range, dyaw_range = ranges['dxy'], ranges['dyaw']
        

        for step in range(max_steps):
            step_log = {"step": step + 1}

            manager.start_new_step()
            
            # --- Action Prediction ---
            action_prompt = build_action_prompt(
                start_pose_str=start_pose_str,
                dxy_range=dxy_range,
                dyaw_range=dyaw_range
            )
            action_inputs = self.tokenizer(text=[action_prompt], images=[start_img, goal_img, current_observation], return_tensors="pt").to(self.args.device)
            action_gen_kwargs = manager.get_action_kwargs(action_inputs, action_gen_kwargs, step)

            with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                action_outputs = model.generate(**action_inputs, **action_gen_kwargs)

            decoded = self.tokenizer.batch_decode(action_outputs[0], skip_special_tokens=False)[0].strip()
            is_stop = decoded.lower() == "stop"
            pattern = r'(<d[^>]+>)+(<d[^>]+>)'
            decoded = re.sub(pattern, r'\2', decoded)
            all_decoded.append(decoded)
            
            dx, dy, dyaw = 0.0, 0.0, 0.0
            if not is_stop:
                bin_step = self.action_cfg['bin_step']
                dx = extract_bin_values(decoded, "dx", bin_step)
                dy = extract_bin_values(decoded, "dy", bin_step)
                dyaw = extract_bin_values(decoded, "dyaw", bin_step)
                action_string = f"dx: {dx}, dy: {dy}, dyaw: {dyaw}"
            else:
                action_string = "Stop"

            step_log["predicted_action"] = action_string
            if self.is_world_process_zero():
                print(f"  Step {step+1}: Predicted Action: '{action_string}'")

            all_actions.append(action_string)

            current_pose[0] += dx
            current_pose[1] += dy
            current_pose[2] += dyaw
            predicted_coords.append(list(current_pose))

            # Termination condition
            should_stop = is_stop or (dx == 0.0 and dy == 0.0 and dyaw == 0.0)
            if should_stop or step == max_steps - 1:
                current_trajectory_steps.append(step_log)
                if self.is_world_process_zero():
                    print("  Stop action predicted or max steps reached. Ending trajectory.")
                break

            # --- Visualization Generation ---
            
            viz_prompt = build_viz_prompt(
                decoded_action=decoded,
                start_pose_str=start_pose_str,
            )
            viz_inputs = self.tokenizer(text=[viz_prompt], images=[start_img, goal_img, current_observation], return_tensors="pt").to(self.args.device)
                      
            viz_gen_kwargs = manager.get_viz_kwargs(viz_inputs, viz_gen_kwargs, step)
                
            with torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                viz_outputs = model.generate(**viz_inputs, **viz_gen_kwargs)
                
            save_path = os.path.join(traj_output_dir, f"step_{step+1}_observation.png")
            new_obs_img = self._parse_and_save_image(viz_outputs[0], model, self.tokenizer, save_path)
            
            step_log["predicted_observation_path"] = save_path if new_obs_img else "GENERATION_FAILED"
            current_trajectory_steps.append(step_log)

            if not new_obs_img:
                if self.is_world_process_zero():
                    logger.warning("  Visualization failed to generate an image. Ending trajectory.")
                break
            
            current_observation = self._sanitize_generated_image(new_obs_img)

            manager.store_step_memory()
            
        # --- 3. Trajectory Evaluation ---
        final_position = np.array(current_pose[:2])
        distance_to_goal = np.linalg.norm(final_position - gt_goal_pos) if gt_goal_pos is not None else None

        pred_coords_arr = np.array(predicted_coords)
        ate, rpe_trans, rpe_rot = float('nan'), float('nan'), float('nan')
        if len(pred_coords_arr) >= 2:
            try:
                pred_traj_evo = coords_to_evo_traj(pred_coords_arr)
                gt_traj_evo = coords_to_evo_traj(gt_coords_arr)
                ate, rpe_trans, rpe_rot = eval_ate_rpe(gt_traj_evo, pred_traj_evo)
            except Exception as e:
                print(f"Warning: ATE/RPE calculation failed for {traj_dirname} with error: {e}")
        
        if self.is_world_process_zero():
            print(f"--- Trajectory {traj_dirname} finished. ATE: {ate:.3f}, RPE_trans: {rpe_trans:.3f}, RPE_rot: {rpe_rot:.3f} ---")

        # --- 4. Save and Return ---
        final_eval_record = {
            "final_position": final_position.tolist(),
            "goal_position": gt_goal_pos.tolist() if gt_goal_pos is not None else None,
            "distance_to_goal": distance_to_goal,
            "ate": ate,
            "rpe_trans": rpe_trans,
            "rpe_rot": rpe_rot,
            "trajectory_actions": all_actions,
            "trajectory_decoded": all_decoded,
        }
        
        eval_file_path = os.path.join(traj_output_dir, "eval_result.json")
        with open(eval_file_path, "w") as f_eval:
            json.dump(final_eval_record, f_eval, indent=4)

        summary_log = {
            "trajectory_id": traj_dirname,
            "input_start_img_path": raw_item['input_img_paths'][0],
            "input_goal_img_path": raw_item['input_img_paths'][1],
            "ground_truth_goal_path": raw_item['label_img_paths'][0],
            "steps": current_trajectory_steps
        }
        
        return {**final_eval_record, "summary_log": summary_log}


    def task_level_evaluation_loop(self, test_dataset: Dataset, test_examples: Dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
        logger.info("\n***** Running Task-Level Trajectory Generation *****")

        # --- 1. Setup ---
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        manager = MemoryBankManager(
            model=model, 
            trainer=self, 
            use_memory_bank_inference=self.args.use_memory_bank_inference
        )
        # Enable memory bank if the model supports it
        if manager.is_enabled and hasattr(model, 'enable_memory_bank'):
            try:
                model.enable_memory_bank()
                if self.is_world_process_zero():
                    print("  Memory bank functionality enabled for task-level evaluation")
                    print("  Memory bank configuration:")
                    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                        memory_bank_layers = 0
                        for layer_idx, layer in enumerate(model.model.model.layers):
                            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'use_memory_bank'):
                                if layer.self_attn.use_memory_bank:
                                    memory_bank_layers += 1
                                    print(f"    - Layer {layer_idx}: Memory bank enabled")
                        print(f"    - Total layers with memory bank: {memory_bank_layers}")
            except Exception as e:
                if self.is_world_process_zero():
                    print(f"  Warning: Failed to enable memory bank: {e}")
        
        output_dir = os.path.join(self.args.output_dir, f"{metric_key_prefix}_task_level_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving task-level results to: {output_dir}")
            
        dataloader = self.get_eval_dataloader(test_dataset)

        viz_generation_kwargs = {
            "num_beams": 1,
            "max_new_tokens": self.model.image_token_num + 20, 
            "temperature": 0.7, 
            "top_p": 0.9,      
        }
        action_generation_kwargs = self._gen_kwargs if hasattr(self, '_gen_kwargs') else {}


        # --- 2. Main Loop ---
        all_trajectories_log = []
        all_trajectory_distances = []
        all_trajectory_ate = []
        all_trajectory_rpe = []

        for item_idx, inputs in enumerate(dataloader):
            raw_item = test_examples[item_idx]
            
            result = self._process_single_trajectory(
                raw_item, 
                output_dir,
                model,
                action_generation_kwargs,
                viz_generation_kwargs,
                manager
            )
            
            # Aggregate results
            if result.get("distance_to_goal") is not None:
                all_trajectory_distances.append(result["distance_to_goal"])
            all_trajectory_ate.append(result.get("ate", float('nan')))
            all_trajectory_rpe.append(result.get("rpe_trans", float('nan')))
            all_trajectories_log.append(result["summary_log"])

            # Print running averages
            if self.is_world_process_zero() and all_trajectory_distances:
                current_avg_distance = np.mean(all_trajectory_distances)
                current_avg_ate = np.nanmean(all_trajectory_ate)
                current_avg_rpe = np.nanmean(all_trajectory_rpe)
                print(f"--- Running Avg Distance: {current_avg_distance:.3f}, Avg ATE: {current_avg_ate:.3f}, Avg RPE: {current_avg_rpe:.3f} ---")

        # --- 3. Final Logging and Return ---
        log_file_path = os.path.join(output_dir, "predictions_log_by_name.json")
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(all_trajectories_log, f, indent=4, ensure_ascii=False)
        print(f"\nFull trajectory log saved to: {log_file_path}")
        
        # Print final memory bank statistics
        if manager.is_enabled and hasattr(model, 'memory_bank_initialized') and model.memory_bank_initialized:
            if self.is_world_process_zero():
                print("\n=== Memory Bank Final Statistics ===")
                total_stored_elements = 0
                if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                    for layer_idx, layer in enumerate(model.model.model.layers):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'stored_keys'):
                            if layer.self_attn.stored_keys is not None:
                                layer_elements = layer.self_attn.stored_keys.numel() + layer.self_attn.stored_values.numel()
                                total_stored_elements += layer_elements
                                print(f"  Layer {layer_idx}: {layer_elements} elements stored")
                print(f"  Total memory bank storage: {total_stored_elements} elements")
                print(f"  Memory bank was used across {len(all_trajectories_log)} trajectories")
                print("=" * 40)

        metrics = {
            f"{metric_key_prefix}_mean_distance": np.nanmean(all_trajectory_distances),
            f"{metric_key_prefix}_mean_ate": np.nanmean(all_trajectory_ate),
            f"{metric_key_prefix}_mean_rpe": np.nanmean(all_trajectory_rpe),
            f"{metric_key_prefix}_samples_processed": len(test_dataset)
        }
        self.log(metrics)
        return metrics

    def _stitch_and_save_comparison_image(self, gt_img: Image.Image, pred_img: Image.Image, save_path: str):
        gt_img = gt_img.resize((256, 256))
        pred_img = pred_img.resize((256, 256))
        
        comparison_img = Image.new('RGB', (256 * 2, 256))
        comparison_img.paste(gt_img, (0, 0))
        comparison_img.paste(pred_img, (256, 0))
        comparison_img.save(save_path)

    def _process_gt_action_rollout_trajectory(self, raw_item: Dict, output_dir: str, model: Any, viz_gen_kwargs: Dict, image_metrics_calculator) -> Dict[str, Any]:
        start_path = raw_item['input_img_paths'][0]
        traj_dir = os.path.dirname(start_path)
        traj_dirname = os.path.basename(traj_dir)
        traj_output_dir = os.path.join(output_dir, traj_dirname)
        os.makedirs(traj_output_dir, exist_ok=True)

        if self.is_world_process_zero():
            print(f"--- Processing Trajectory {traj_dirname} ---")

        # --- 1. Load all required data from files ---
        try:
            # Load GT actions from pickle file
            with open(os.path.join(traj_dir, 'traj_data.pkl'), 'rb') as f:
                gt_actions = pickle.load(f)['delta']
            if hasattr(gt_actions, 'tolist'):
                gt_actions = gt_actions.tolist()

            # Reconstruct GT image paths
            img_files = sorted(
                [f for f in os.listdir(traj_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda f: int(os.path.splitext(f)[0])
            )
            gt_img_paths = [os.path.join(traj_dir, f) for f in img_files]
        except Exception as e:
            logger.error(f"  Failed to load data for {traj_dirname}. Error: {e}")
            return {} # Return empty dict on failure

        # --- 2. Initialize rollout state ---
        start_img = self._sanitize_generated_image(raw_item['input_imgs'][0])
        goal_img = self._sanitize_generated_image(raw_item['input_imgs'][1])
        start_pose_str = f"Start Pose: x={raw_item['coords'][0][0]:.3f}, y={raw_item['coords'][0][1]:.3f}, yaw={raw_item['coords'][0][2]:.3f}\n"
        
        current_observation = start_img
        num_steps = 5

        # --- 3. The 5-Step Rollout Loop ---
        for step in range(num_steps):
            if not (step < len(gt_actions) and (step + 1) < len(gt_img_paths)):
                logger.warning(f"  Trajectory {traj_dirname} too short. Stopping.")
                return {} # Skip this trajectory

            gt_action_vec = gt_actions[step]
            action_text = action_to_text(gt_action_vec)
            
            viz_prompt = build_viz_prompt(action_text, start_pose_str)
            viz_inputs = self.tokenizer(
                text=[viz_prompt], images=[start_img, goal_img, current_observation], 
                return_tensors="pt"
            ).to(self.args.device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=self.model.dtype):
                viz_outputs = model.generate(**viz_inputs, **viz_gen_kwargs)

            save_path = os.path.join(traj_output_dir, f"step_{step+1}_observation.png")
            new_obs_img = self._parse_and_save_image(viz_outputs[0], self.model, self.tokenizer, save_path)

            if not new_obs_img:
                logger.warning(f"  Visualization failed at step {step + 1}. Stopping.")
                return {} # Skip this trajectory
            
            # Stitch GT and Pred images and save
            gt_current_img = Image.open(gt_img_paths[step + 1])
            self._stitch_and_save_comparison_image(gt_current_img, new_obs_img, save_path)

            current_observation = self._sanitize_generated_image(new_obs_img)

        # --- 4. Final Evaluation and Saving ---
        final_predicted_img = current_observation
        gt_final_img = Image.open(gt_img_paths[num_steps]).convert("RGB")
        
        final_viz_metrics = image_metrics_calculator.calculate(final_predicted_img, gt_final_img)
        
        eval_file_path = os.path.join(traj_output_dir, "eval_result.json")
        with open(eval_file_path, "w") as f_eval:
            json.dump(final_viz_metrics, f_eval, indent=4)
        
        # Return only the metrics for aggregation
        return final_viz_metrics


    def run_gt_action_rollout(self, test_dataset: Dataset, test_examples: Dataset, metric_key_prefix: str = "eval") -> Dict[str, float]:
        logger.info("\n***** Running GT Action Viz Rollout *****")

        # --- 1. Setup ---
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        output_dir = os.path.join(self.args.output_dir, f"{metric_key_prefix}_gt_action_rollout_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving task-level results to: {output_dir}")
            
        dataloader = self.get_eval_dataloader(test_dataset)

        viz_generation_kwargs = {
            "num_beams": 1,
            "max_new_tokens": self.model.image_token_num + 20,
            "temperature": 0.7, 
            "top_p": 0.9,
        }
        action_generation_kwargs = self._gen_kwargs if hasattr(self, '_gen_kwargs') else {}
        
        image_metrics_calculator = ImageMetricsCalculator(device=self.args.device)

        # --- 2. Main Loop ---
        all_viz_metrics = defaultdict(list)

        for item_idx, inputs in enumerate(dataloader):
            raw_item = test_examples[item_idx]
            
            metrics = self._process_gt_action_rollout_trajectory(
                raw_item, 
                output_dir,
                model,
                viz_generation_kwargs,
                image_metrics_calculator
            )
            
            if metrics:
                for key, value in metrics.items():
                    all_viz_metrics[key].append(value)
        
        # --- 3. Final Logging and Return ---
        final_summary = {}
        print("\n--- Final Aggregated Metrics ---")
        for key, values in all_viz_metrics.items():
            if values:
                mean_value = np.mean(values)
                final_summary[f"{metric_key_prefix}_{key}_mean"] = mean_value
                print(f"  Average {key}: {mean_value:.4f}")

        summary_path = os.path.join(output_dir, "final_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=4)
        print(f"Final summary saved to {summary_path}")

        self.log(final_summary)
        return final_summary