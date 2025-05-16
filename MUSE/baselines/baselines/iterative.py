from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM

from torch import nn


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    beta: float = 0.1,
    coeff: float = 1.0,
    npo_coeff: float = 1.0,
    pl_coeff: float = 1.0,
    gamma: float = 0.0,
    flag: bool = True,
    common_words_file: str = ''
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type or 'tpo' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        flag,
        common_words_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_strategy='epoch',  # Save every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none'        # Disable wandb
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        beta=beta,
        coeff=coeff,
        npo_coeff=npo_coeff,
        pl_coeff=pl_coeff,
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 coeff: float = 1.0,
                 npo_coeff: float = 1.0,
                 pl_coeff:float = 1.0,
                 gamma: float = 0.0,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when `'po' in self.loss_type`
        self.coeff = coeff
        self.npo_coeff = npo_coeff
        self.pl_coeff = pl_coeff
        self.gamma = gamma

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        self.grads = []

        super().__init__(*args, **kwargs)


    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
        ### 1. Run model ###
        x_f, x_r = x
        # print(x_f['input_ids'].shape)
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type or 'tpo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        ### 2. Compute Loss ###
        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type and 'simnpo' not in self.loss_type:
            # print(outputs_f_ref.logits.mean())
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif 'tpo' in self.loss_type:
            reversed_labels = []
            for ids, label in zip(x_f['input_ids'], x_f['labels']):
                # Create a new labels tensor by reversing the mask
                reversed_labels = torch.where(label == -100, ids, torch.tensor(-100, device= x_f['labels'].device))
                
                token_2_indices = (reversed_labels == 2).nonzero(as_tuple=True)[0]
                if len(token_2_indices) > 1:
                    reversed_labels[token_2_indices[1:]] = -100  # Mask all occurrences except the first
             
                reversed_labels.append(reversed_labels)
            reversed_labels = torch.stack(reversed_labels)
            outputs = model(x_f['input_ids'],labels=reversed_labels, attention_mask=x_f['attention_mask'])         ##attention_mask is used to indicate which tokens to attend to ()
            pl_loss = get_batch_loss(outputs.logits, reversed_labels).mean()
     

            logits_f = outputs_f.logits[:,:-1,:]
            logits_oracle_f = outputs_f_ref.logits[:,:-1,:]
            input_ids_expanded_f = x_f['input_ids'][:,1:].unsqueeze(-1)
            loss_indexes_f = (x_f['labels'][:,1:]!=-100).float()
            lpl_loss = ((torch.gather(logits_oracle_f, dim=-1, index=input_ids_expanded_f).squeeze(-1) * loss_indexes_f).sum(-1) / loss_indexes_f.sum(-1)).mean() - ((torch.gather(logits_f, dim=-1, index=input_ids_expanded_f).squeeze(-1) * loss_indexes_f).sum(-1) / loss_indexes_f.sum(-1)).mean()
            
            loss +=  -F.logsigmoid(self.beta *lpl_loss)* 2/self.beta + self.pl_coeff*pl_loss


        elif 'simnpo' in self.loss_type:
            neg_log_ratio = - outputs_f.logits - self.gamma
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            if 'tpo' in self.loss_type:
                print(f"loss_lpl: {lpl_loss}, loss_pl: {pl_loss}, loss_r: {loss_r}")
            else:
                print(f"loss_f: {loss_f}, loss_r: {loss_r}")

            loss = self.npo_coeff * loss + self.coeff * loss_r

        elif 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        elif 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            print(f"loss_f: {loss_f}, loss_r: {kl_r}")

            loss += self.coeff * kl_r
        elif 'tpo' in self.loss_type:
            print(f"loss_ldl: {lpl_loss}, loss_pl: {pl_loss}")
        else:
            print(f"loss_f: {loss_f}")

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
