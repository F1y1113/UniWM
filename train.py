import os
import wandb
import torch
import logging
import argparse
import yaml
import copy
import math
import configparser
import transformers
import torch.distributed as dist

from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, StopStringCriteria, set_seed, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from transformers.generation import StoppingCriteriaList

from scripts.run_config import create_run_name
from scripts.training_arguments import WrappedSeq2SeqTrainingArguments
from scripts.load_data import load_data, tokenize_dataset
from scripts.load_model import load_model
from scripts.evaluator import VisualizationEvaluator
from scripts.action_utils import generate_bin_tokens

from transformers.utils import logging as hf_logging 

hf_logging.set_verbosity_error()  

logger = logging.getLogger(__name__)

WANDB_API_KEY = "<YOUR_WANDB_KEY_API>"
WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"
PROJECT_NAME = "<YOUR_PROJECT_NAME>"

import os
import re

def find_latest_valid_checkpoint(ckpt_dir):
    """Find the latest valid checkpoint (contains trainer_state.json) in the directory."""
    if not os.path.exists(ckpt_dir):
        return None

    checkpoint_dirs = [
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and re.match(r"checkpoint-\d+", d)
    ]

    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]), reverse=True)

    for ckpt in checkpoint_dirs:
        trainer_state_path = os.path.join(ckpt_dir, ckpt, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            print(f"[INFO] Found valid checkpoint: {ckpt}")
            return os.path.join(ckpt_dir, ckpt)

    print("[INFO] No valid checkpoint found.")
    return None

def init(args):
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    # Read in the training arguments
    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, setting_type + '.yaml')) as f:
        file = f.read()
        training_cfg = yaml.safe_load(file)

    if args.train_bz:
        training_cfg['hyper']['train_batch_size'] = args.train_bz
    if args.val_bz:
        training_cfg['hyper']['val_batch_size'] = args.val_bz
    if args.grad_acc:
        training_cfg['hyper']['grad_accumulation'] = args.grad_acc

    sup_hyper = training_cfg["hyper"]

    # Construct the run_name of the task
    args.run_name = create_run_name(args, training_cfg)

    args.run_name = args.note + args.run_name

    training_args = WrappedSeq2SeqTrainingArguments(
        output_dir=os.path.join(args.output, args.run_name),
        remove_unused_columns=False,
        evaluation_strategy=training_cfg['eval']['eval_strategy'],
        eval_steps=training_cfg['eval']['eval_steps'] if training_cfg['eval']['eval_strategy'] == "steps" else None,
        save_strategy=training_cfg['save']['save_strategy'],
        save_steps=training_cfg['save']['save_steps'] if training_cfg['save']['save_strategy'] == "steps" else 2000,
        save_total_limit=40,
        seed=args.seed,
        # note: for supervised tuning
        #############################
        learning_rate=sup_hyper['lr'] if sup_hyper else 0,
        per_device_train_batch_size=sup_hyper['train_batch_size'] if sup_hyper else 0,
        gradient_accumulation_steps=sup_hyper['grad_accumulation'] if sup_hyper else 0,
        per_device_eval_batch_size=sup_hyper['val_batch_size'] if sup_hyper else training_cfg['hyper']['val_batch_size'],
        num_train_epochs=sup_hyper['epochs'] if sup_hyper else 0,
        #############################
        # warmup_ratio=0.1,
        logging_steps=training_cfg['logging']['logging_step'],
        logging_strategy="steps",
        disable_tqdm=False,
        push_to_hub=False,
        # customize
        predict_with_generate=training_cfg['model']['predict_with_generate'],
        generation_max_new_tokens=training_cfg['model']['generation_max_new_tokens'],
        generation_num_beams=training_cfg['model']['generation_num_beams'],
        use_memory_bank_inference=args.use_memory_bank_inference,
    )

    # Initialize the wandb logger if specified
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    args.local_rank = rank

    if args.report_to == "wandb" and rank == 0:
        import wandb
        init_args = {}

        # note: my new wandb api key
        wandb.login(key=WANDB_API_KEY)

        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        
        if args.local_rank == 0 or args.local_rank is None:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", PROJECT_NAME),
                name=args.run_name,
                entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                **init_args,
            )
            wandb.config.update(training_args, allow_val_change=True)
    else:
        training_args.report_to = []

    # if os.path.exists(training_args.output_dir):
    #     args.model_ckpt = training_args.output_dir

    # # Detect the checkpoint
    # if args.model_ckpt is not None:
    #     training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    # else:
    #     training_args.load_weights_from = None

    # Detect the checkpoint

    # Detect the checkpoint
    if args.model_ckpt is not None:
        trainer_state_file = os.path.join(args.model_ckpt, "trainer_state.json")
        if os.path.isfile(trainer_state_file):
            training_args.load_weights_from = args.model_ckpt
        else:
            latest_ckpt = find_latest_valid_checkpoint(args.model_ckpt)
            if latest_ckpt:
                print(f"[INFO] Found latest valid checkpoint: {latest_ckpt}")
                training_args.load_weights_from = latest_ckpt
            else:
                print(f"[INFO] No valid checkpoint found in directory: {args.model_ckpt}")
                training_args.load_weights_from = None
    else:
        training_args.load_weights_from = None


    return training_cfg, training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anole")
    parser.add_argument("--data", type=str, nargs="+")
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default='anole')
    parser.add_argument('--note', type=str, default="debug")
    parser.add_argument('--image_seq_length', type=int, default=784)
    parser.add_argument('--no_perceptual_loss', action="store_true")

    # model argument
    parser.add_argument('--model_ckpt', type=str, default=None, help='path of the checkpoint')
    parser.add_argument('--load_last_checkpoint', action='store_true')

    # training arguments
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_single_step_eval', action='store_true')
    parser.add_argument('--do_task_level_eval', action='store_true')
    parser.add_argument('--do_rollout_eval', action='store_true')
    parser.add_argument('--cfg_path', type=str, default='cfg')
    parser.add_argument('--patience', type=int, default=5)

    # input format argument
    parser.add_argument('--input_format', type=str, default="anole")

    # output configuration
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--report_to', type=str, default="wandb")
    parser.add_argument('--cache_dir', type=str, default=None)

    # randomness
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int)

    # debug
    parser.add_argument('--toy', action='store_true')

    # shortcut customization
    parser.add_argument('--train_bz', type=int, default=None)
    parser.add_argument('--val_bz', type=int, default=None)
    parser.add_argument('--grad_acc', type=int, default=None)
    
    # Precision argument: only support bfloat16, remove fp16
    parser.add_argument('--bfloat16', action='store_true', help='Whether to use bfloat16 precision')

    parser.add_argument('--use_memory_bank_inference', action='store_true')
    args = parser.parse_args()

    if args.model in ['anole']:
        args.decoder_type = args.model
        assert args.input_format == "anole"

    if args.decoder_type in ['anole']:
        args.note = args.note + f"image_seq_len-{str(args.image_seq_length)}-"

    # Pass bfloat16 argument to WrappedSeq2SeqTrainingArguments via args
    def _add_bfloat16_to_args(args, training_args):
        if hasattr(args, "bfloat16") and args.bfloat16:
            # Set the attribute on training_args if supported
            setattr(training_args, "bfloat16", True)
        return training_args

    training_cfg, training_args = init(args)
    training_args = _add_bfloat16_to_args(args, training_args)

    print(f'Preparing the {args.data} dataset... ')
    data = load_data(dataset=args.data, data_dir=args.data_dir)

    if len(data) == 2:
        train_split, eval_split, test_split = data['train'], None, data['test']
    else:
        try:
            train_split, eval_split, test_split = data['train'], data['dev'], data['test']
        except:
            train_split, eval_split, test_split = data['train'], data['validation'], data['test']

    if args.toy:
        print('Only using toy examples for debugging...')
        train_split = train_split.select(list(range(min(100, len(train_split)))))
        if eval_split:
            eval_split = eval_split.select(list(range(min(10, len(eval_split)))))
        test_split = test_split.select(list(range(min(10, len(test_split)))))

    model_processor = load_model(args, training_cfg)
    model, processor = model_processor['model'], model_processor["processor"]

    # def generate_bin_tokens(prefix, vmin, vmax, step):
    #     nbins = int((vmax - vmin) / step) + 1
    #     return [f"<{prefix}_bin_{i:02d}>" for i in range(nbins)]
    
    bin_tokens = []
    if args.model_ckpt is None:
        # --- Read action token settings from the YAML config ---
        print("Reading action token generation settings from config...")
        action_cfg = training_cfg['action_token_generation']
        MIN_DXY = action_cfg['min_dxy']
        MAX_DXY = action_cfg['max_dxy']
        MIN_DYAW = action_cfg['min_dyaw']
        MAX_DYAW = action_cfg['max_dyaw']
        BIN_STEP = action_cfg['bin_step']

        print(f"DX/DY Range for Vocabulary: [{MIN_DXY}, {MAX_DXY}]")
        print(f"DYAW Range for Vocabulary: [{MIN_DYAW}, {MAX_DYAW}]")
        
        # --- Generate vocabulary of action tokens ---
        print("Generating vocabulary of action tokens...")
        
        bin_tokens += generate_bin_tokens("dx", MIN_DXY, MAX_DXY, BIN_STEP)
        bin_tokens += generate_bin_tokens("dy", MIN_DXY, MAX_DXY, BIN_STEP)
        bin_tokens += generate_bin_tokens("dyaw", MIN_DYAW, MAX_DYAW, BIN_STEP)
        
        print(f"Generated a total of {len(bin_tokens)} action tokens.")


    existing_vocab = set(processor.tokenizer.get_vocab().keys())
    new_tokens = [t for t in bin_tokens if t not in existing_vocab]

    # new_tokens.append("<IMS>")
    # new_tokens.append("<IME>")

    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)

        # Handle PEFT model token embedding resizing
        # Check for both possible PEFT wrapper types
        if (hasattr(model.model.lm_head, 'base_layer') or 
            hasattr(model.model.lm_head, '__class__') and 
            'ModulesToSaveWrapper' in str(model.model.lm_head.__class__)):
            from peft.utils.other import ModulesToSaveWrapper
            # Unwrap the lm_head
            original_lm_head = model.model.lm_head
            if hasattr(model.model.lm_head, 'base_layer') or 'ModulesToSaveWrapper' in str(type(model.model.lm_head)):
                # PEFT model - unwrap, resize, and re-wrap
                if hasattr(model.model.lm_head, 'base_layer'):
                    model.model.lm_head = model.model.lm_head.base_layer
                elif hasattr(model.model.lm_head, 'original_module'):
                    model.model.lm_head = model.model.lm_head.original_module
                
                # Resize embeddings
                model.model.resize_token_embeddings(len(processor.tokenizer))
                
                # Re-wrap the lm_head with the correct adapter_name
                model.model.lm_head = ModulesToSaveWrapper(model.model.lm_head, "default")
            else:
                # Standard model without PEFT
                model.model.resize_token_embeddings(len(processor.tokenizer))

    # Remove this line - it's causing the error
    # model.model.resize_token_embeddings(len(processor.tokenizer))
    eval_data_num = (len(eval_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    eval_split = eval_split.select(list(range(eval_data_num)))
    # eval_split = eval_split.filter(lambda ex: len(ex['label_imgs']) > 0 and ex['train_task'] == 'single_step_visualization')
    # eval_split = eval_split.filter(lambda ex: len(ex['label_imgs']) > 0)
    eval_split = eval_split.filter(
    lambda ex: ex['train_task'] == 'action_reasoning' or len(ex['label_imgs']) > 0
    )
    test_data_num = (len(test_split) // (training_args.per_device_eval_batch_size * torch.cuda.device_count())) * (training_args.per_device_eval_batch_size * torch.cuda.device_count())
    test_split = test_split.select(list(range(test_data_num)))

    print(f"Eval Num: {eval_data_num}")

    tokenized_data, max_source_length, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        model=model,
        processor=processor,
        input_format=args.input_format,
        interleave=True,
        data_name = "-".join(args.data),
    )

    training_args.generation_max_new_tokens = max_target_length + 100
    print(f"generation_max_new_tokens: {training_args.generation_max_new_tokens}")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience)
    label_pad_token_id = -100
    
    # Data collator: 
    from scripts.data_collator import customize_data_collator
    data_collator = customize_data_collator
    
    from scripts.customize_trainer import CustomizeSeq2SeqTrainer
    trainer_type = CustomizeSeq2SeqTrainer

    # fixme:
    training_args.label_smoothing_factor = 0.1

    if args.model in ['anole']:
        # used in evaluation when do_eval
        kwargs = dict()
        kwargs['multimodal_generation_mode'] = "interleaved-text-image"     # see L217 in wrapped_visualizer.py
        kwargs['stopping_criteria'] = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])
        # used in evaluation during training
        training_args.customize_gen_stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])

    trainer = trainer_type(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args),
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval'] if 'eval' in tokenized_data.keys() else tokenized_data['test'],
        eval_examples=eval_split if 'eval' in tokenized_data.keys() else test_split,
        wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
        # callbacks=[early_stopping_callback],  # currently disabled early stopping for now
        image_loss_func=not args.no_perceptual_loss, 
        action_cfg=training_cfg.get('action_token_generation'),
    )

    # if not dist.is_initialized() or dist.get_rank() == 0:
    #     print("DEBUG SAMPLE:", eval_dataset[0])

    print('Trainer build successfully.')

    # for anole, there would be different inference mode. We use kwargs to pass these settings into the inference process.
    checkpoint = None
    if training_args.load_weights_from is not None:
        checkpoint = training_args.load_weights_from

    # NOTE: train the model with supervision
    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(tokenized_data['train'])
        metrics["train_samples"] = min(max_train_samples, len(tokenized_data['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_single_step_eval:
        logger.info("*** Single Step Eval ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            **kwargs
        )
        max_eval_samples = len(tokenized_data['eval'])
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_data['eval']))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_task_level_eval:
        logger.info("*** Task Level Eval ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict",
            predict_task = "task_level_evaluation",
            **kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    if args.do_rollout_eval:
        logger.info("*** Rollout ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict",
            predict_task = "rollout_evaluation",
            **kwargs
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)