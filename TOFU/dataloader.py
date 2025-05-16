import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
import datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals, get_kl_divergence, get_masked_kl_divergence
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss, get_single_token_loss
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
import pickle
import math

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')

        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')

        # the coefficient of each part in the loss function. This is used in ablation study.
        self.npo_coeff=kwargs.pop('npo_coeff')
        self.grad_diff_coeff=kwargs.pop('grad_diff_coeff')
        self.KL_coeff=kwargs.pop('KL_coeff')

        self.ref_policy = kwargs.pop('ref_policy')

        self.beta = kwargs.pop('beta')
        self.gamma = kwargs.pop('gamma')
        self.model_family = kwargs.pop('model_family')

        self.loss_log = []

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        # Here, we always need the oracle model to compute the KL distance in the evaluation time.
        self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def get_train_dataloader(self):
        """
        Override the original get_train_dataloader function simply for debugging.
        This is identical to the get_train_dataloader function in transformer.Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def adjust_logtis_batch(self, output_logits_batch, labels_start, labels_end, n):
        for i in range(output_logits_batch.size(0)):  # Number of samples is the first dimension (8)
            start = labels_start[i]  # Starting index for the window for sample i
            end = labels_end[i] 
            window_logits = output_logits_batch[i, start:end, :]

            for pos in range(window_logits.size(0)): 
                position_logits = window_logits[pos, :]  # Logits for a specific position (shape: [vocab_size])
                    
                top_n_values, top_n_indices = torch.topk(position_logits, k=n)
                
                # Decrease the top n tokens' logits
                output_logits_batch[i, start + pos, top_n_indices] = float(-1e4)
        return output_logits_batch

    def save_loss_log_csv(self, output_dir):
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the output CSV file path
        csv_file_path = os.path.join(output_dir, "loss_log.csv")

        # Save the loss log to the CSV file
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["step", "loss", "loss_untarget"])
            writer.writeheader()  # Write the header
            writer.writerows(self.loss_log)  # Write the rows

        print(f"Loss log saved to: {csv_file_path}")
    
    def find_sublist_position(self, tensor_list, sublist):
        for i in range(len(tensor_list) - len(sublist) + 1):
            if torch.equal(tensor_list[i:i+len(sublist)], sublist):
                return i
        return -1  # Return -1 if not found
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)         ##attention_mask is used to indicate which tokens to attend to ()
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
            print(f"********************** forget_loss: {'%.4f'%(forget_loss.item())} **********************")

        elif self.loss_type == 'tpo':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            reversed_labels = []

            for ids, label in zip(input_ids, labels):
                # Create a new labels tensor by reversing the mask
                reversed_labels = torch.where(label == -100, ids, torch.tensor(-100, device=labels.device))
                
                if self.model_family == 'llama2-7b':
                    token_pad_indices = (reversed_labels == 2).nonzero(as_tuple=True)[0]
                    answer_tag_ids = torch.tensor([29914, 25580, 29962]).to(reversed_labels.device)

                elif self.model_family == 'llama3-3b':
                    token_pad_indices = (reversed_labels == 128001).nonzero(as_tuple=True)[0]
                    answer_tag_ids = torch.tensor([16533, 25]).to(reversed_labels.device)
                else:
                    token_pad_indices = (reversed_labels == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                
                answer_tag_position = self.find_sublist_position(reversed_labels, answer_tag_ids)

                if len(token_pad_indices) > 1:
                    reversed_labels[token_pad_indices[1:]] = -100  # Mask all occurrences except the first
                reversed_labels[:answer_tag_position+len(answer_tag_ids)] = -100


                reversed_labels.append(reversed_labels)
            reversed_labels = torch.stack(reversed_labels)
  

            outputs2 = model(input_ids,labels=reversed_labels, attention_mask=attention_mask)         ##attention_mask is used to indicate which tokens to attend to ()
            pl_loss = get_batch_loss(outputs2.logits, reversed_labels).mean()


            batch_size, seq_len = input_ids[:,1:].size()
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)  # (batch_size, seq_len)
            seq_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)    # (batch_size, seq_len)

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits.clone()
                    forget_logits_oracle[batch_idx, seq_idx, input_ids[:,1:]] = float(-1e4)
            else:
                raise NotImplementedError

            input_ids_expanded = input_ids[:,1:].unsqueeze(-1)
            logits = outputs.logits[:,:-1,:]
            logits_oracle  = forget_outputs_oracle.logits[:,:-1,:]
            loss_indexes = (labels[:,1:]!=-100).float()

            
            lpl = ((torch.gather(logits_oracle, dim=-1, index=input_ids_expanded).squeeze(-1) * loss_indexes).sum(-1) / loss_indexes.sum(-1)).mean() - ((torch.gather(logits, dim=-1, index=input_ids_expanded).squeeze(-1) * loss_indexes).sum(-1) / loss_indexes.sum(-1)).mean()

            loss =  -F.logsigmoid(self.beta *lpl)* 2/self.beta  + pl_loss
            
            print(f"********************** lpl: {'%.4f'%(lpl.item())} -- pl_loss: {'%.4f'%(pl_loss.item())}  -- Final_loss: {'%.4f'%(loss.item())} **********************")
            
            self.loss_log.append({
                "step": len(self.loss_log) + 1,  # Track the step number
                "loss": loss.item(),
                "loss_untarget": pl_loss.item()
            })

        elif self.loss_type == 'tpo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            new_labels = []

            for ids, label in zip(input_ids, labels):
                # Create a new labels tensor by reversing the mask
                new_label = torch.where(label == -100, ids, torch.tensor(-100, device=labels.device))
                
                if self.model_family == 'llama2-7b':
                    token_pad_indices = (new_label == 2).nonzero(as_tuple=True)[0]
                    answer_tag_ids = torch.tensor([29914, 25580, 29962]).to(new_label.device)

                elif self.model_family == 'llama3-3b':
                    token_pad_indices = (new_label == 128001).nonzero(as_tuple=True)[0]
                    answer_tag_ids = torch.tensor([16533, 25]).to(new_label.device)
                else:
                    token_pad_indices = (new_label == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                
                answer_tag_position = self.find_sublist_position(new_label, answer_tag_ids)

                if len(token_pad_indices) > 1:
                    new_label[token_pad_indices[1:]] = -100  # Mask all occurrences except the first
                new_label[:answer_tag_position+len(answer_tag_ids)] = -100

                new_labels.append(new_label)
            new_labels = torch.stack(new_labels)

            outputs2 = model(input_ids,labels=new_labels, attention_mask=attention_mask)         ##attention_mask is used to indicate which tokens to attend to ()
            loss2 = get_batch_loss(outputs2.logits, new_labels).mean()


            batch_size, seq_len = input_ids[:,1:].size()
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)  # (batch_size, seq_len)
            seq_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)    # (batch_size, seq_len)

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits.clone()
                    forget_logits_oracle[batch_idx, seq_idx, input_ids[:,1:]] = float(-1e4)
            else:
                raise NotImplementedError

            input_ids_expanded = input_ids[:,1:].unsqueeze(-1)
            logits = outputs2.logits[:,:-1,:]
            logits_oracle  = forget_outputs_oracle.logits[:,:-1,:]
            loss_indexes = (labels[:,1:]!=-100).float()

            
            fgt_loss = ((torch.gather(logits_oracle, dim=-1, index=input_ids_expanded).squeeze(-1) * loss_indexes).sum(-1) / loss_indexes.sum(-1)).mean() - ((torch.gather(logits, dim=-1, index=input_ids_expanded).squeeze(-1) * loss_indexes).sum(-1) / loss_indexes.sum(-1)).mean()

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss =  -F.logsigmoid(self.beta *fgt_loss)* 2/self.beta  + loss2 + retain_loss
         
            print(f"********************** ldl: {'%.4f'%(fgt_loss.item())} -- loss2: {'%.4f'%(loss2.item())}  -- re_loss: {'%.4f'%(retain_loss.item())}  -- Final_loss: {'%.4f'%(loss.item())} **********************")
            

        elif self.loss_type in ["dpo","dpo_grad_diff","dpo_gpt"]:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

                idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
                forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()*2/self.beta

            if self.loss_type == 'dpo_grad_diff':
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                loss = loss + retain_loss

        elif self.loss_type == 'kto_logsigmoid':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()

                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta

        elif self.loss_type == 'kto_logsigmoid_grad_diff':
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            
            with torch.no_grad():
                idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                idk_loss_log = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
                idk_loss_log_oracle = -1 * get_batch_loss(idk_outputs_oracle.logits, idk_labels)
                
                KL_term = (idk_loss_log - idk_loss_log_oracle).mean()

                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_loss_oracle = -1 * get_batch_loss(forget_outputs_oracle.logits, forget_labels)

            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            log_ratios = forget_loss - forget_loss_oracle
            forget_loss = 1.0 - F.logsigmoid(KL_term - self.beta * log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss

            loss = forget_loss + retain_loss

        ### Implement the NPO
        elif self.loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle 
            else:
                raise NotImplementedError

            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

        elif self.loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
            

        elif self.loss_type == "simnpo":
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - self.gamma

            loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        elif self.loss_type == 'simnpo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - self.gamma
            forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss

        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        '''
        RZ: Call this function in Trainer.train() when evakluating the performace of each checkpoint.
        '''

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

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
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split_list[-1].split('_')[0]
        
        # masked_kl_divergence_log = 0

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                # print(eval_cfg)

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                # if int(os.environ.get('RANK', '0')) == 0:
                #    import pdb; pdb.set_trace()
                # print(eval_cfg)

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                
                kl_divergence_log = get_kl_divergence(model, self.oracle_model, eval_dataloader)
                eval_logs['kl_divergence'] = kl_divergence_log

                # masked_kl_divergence_log = get_masked_kl_divergence(model, self.oracle_model, eval_dataloader)
                # eval_logs['masked_kl_divergence'] = masked_kl_divergence_log

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:
                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)
                
                model_utility = get_model_utility(aggregated_eval_logs)
                retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                forget_quality, trust_ratio = get_forget_quality(aggregated_eval_logs, retain_result)
                aaggregate_stat = {**model_utility, **forget_quality}

                aaggregate_stat['curr_step'] = curr_step
                aaggregate_stat['seed'] = self.seed
                aaggregate_stat['loss_type'] = self.loss_type

                # aaggregate_stat['masked_kl_divergence_log'] = masked_kl_divergence_log

                with open(os.path.join(curr_save_dir, "aggregate_stat.txt"), 'w') as txtfile:
                    for key, value in aaggregate_stat.items():
                        txtfile.write(f"{key}: {value}\n")

                with open(os.path.join(curr_save_dir, "truth_ratio.pkl"), 'wb') as picklefile:
                    pickle.dump(trust_ratio, picklefile)

class CustomTrainerRetraining(Trainer):
    def __init__(self, *args, **kwargs):
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')
        super(CustomTrainerRetraining, self).__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

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
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split.split('_')[0]

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):

                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:

                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

class PCL_Trainer(Trainer):
    """
    This class is inherited from transformers.Trainer as a custom Trainer for PCL
    Methods: 
        compute_jsd: compute jsd of two distributions
        get_jsd_loss; compute jsd loss given clean and perturbed logits
    """
    def __init__(self, model, tokenizer, args, train_dataset, data_collator, ref_model, forget_loss):
        super().__init__(model = model, tokenizer=tokenizer, args=args, train_dataset=train_dataset,  data_collator=data_collator)
  
        # self.lambda1 = arguments.lambda1
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.forget_loss = forget_loss
        # self.params_map = self.get_params_map()
        # self.param_partition = self.create_param_partition(self.params_map, dim_to_agg=-1)
        # self.perturbation = arguments.perturbation
        # if self.perturbation == 'paraphrases':
        #     self.jsd_type = 'avg'
        # else:
        #     self.jsd_type = arguments.jsd_type



     

def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
         
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


