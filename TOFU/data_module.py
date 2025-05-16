import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
from utils import get_model_identifiers_from_yaml
import json

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs, common_words=None):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    if common_words != None:
        new_question = question_start_token + question + question_end_token + answer_token
        new_answer =  answer
    else:
        new_question = question_start_token + question + question_end_token
        new_answer =  answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
        return_offsets_mapping=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    offset_mapping = encoded['offset_mapping']
    if 'Llama-3' in model_configs['hf_key']:
        for i in range(len(offset_mapping) - 1):
            offset_mapping[i] = (offset_mapping[i][0], offset_mapping[i + 1][0])
        offset_mapping[-1] = (offset_mapping[-1][0], offset_mapping[-1][0]) 

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        # if 'Phi' in model_configs['hf_key']:
        # label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
        # else:
        label = encoded['input_ids']  + [-100] * (pad_length)

    # print(encoded.input_ids)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100
    # print(label)


    if common_words != None:
        # common_words =  json.loads(common_words)

        len_prefix = len(new_question) 
        for word in common_words:
            word = json.loads(word)
            start, end = len_prefix+word['start'], len_prefix+word['end']
            # word_json = json.loads(word)
            # start, end = len_prefix+word_json['start'], len_prefix+word_json['end']
            for i, (encoded_start, encoded_end) in enumerate(offset_mapping):
                if encoded_start < end and encoded_end > start:
                    label[i] =-100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk", fill_mask=False, classifier=None):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fill_mask = fill_mask
        self.classifier = classifier
 
        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            if self.fill_mask:
                if self.classifier == 'gpt':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_gpt")

                elif self.classifier == 'bert':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_bert")
            else:
                self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else: # load dataset from local files.
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            
            torch.manual_seed(idx)
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']
            common_words = None

            if self.fill_mask and data_type != 'retain' and data_type != "idk":
                common_words = data[idx]['common_words']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
            
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, common_words)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", fill_mask=False, classifier=None):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fill_mask = fill_mask
        self.classifier = classifier
        
        if './TOFU_data' not in data_path:
            if self.fill_mask:
                print("||||||||||||||||||||||||||")
                print("Applied token identification process")
                print("||||||||||||||||||||||||||")
                if self.classifier == 'gpt':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_gpt")

                elif self.classifier == 'bert':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_bert")
            else:
                self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']

            common_words = None

            if data_type != "idk":
                if self.fill_mask and data_type != 'retain':
                    common_words = data[idx]['common_words']
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, common_words)
            rets.append(converted_data)
        return rets


class TextForgetDatasetKTOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", fill_mask=False, classifier=None):
        super(TextForgetDatasetKTOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fill_mask = fill_mask
        self.classifier = classifier
        
        if './TOFU_data' not in data_path:
            if self.fill_mask:
                print("||||||||||||||||||||||||||")
                print("Applied token identification process")
                print("||||||||||||||||||||||||||")
                if self.classifier == 'gpt':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_gpt")

                elif self.classifier == 'bert':
                    self.forget_data = datasets.load_from_disk(f"../data/{split}_with_common_words_bert")
            else:
                self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            common_words = None


            if data_type != "idk":
                answer = data[idx]['answer']
                if self.fill_mask and data_type != 'retain':
                    common_words = data[idx]['common_words']
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, common_words)
                rets.append(converted_data)
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()

                answer = self.idk[rand_pos].strip()
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        pad_length_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def get_single_token_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    print(output.shape)

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    # loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    loss = loss_function(output.transpose(-1,-2), shifted_labels)
    print(loss.shape)

    return loss

def get_single_token_loss(output, labels, input_ids, beta):
    shifted_labels = labels[..., 1:].contiguous()
    input_ids_shifted = input_ids[..., 1:].contiguous().unsqueeze(-1)
    output = output[..., :-1, :].contiguous()
    # print(output.shape)
    # print(shifted_labels.shape)
    # labels_expanded = shifted_labels.unsqueeze(-1)  # Shape: [8, 255, 1]

    # Gather the logits corresponding to the labels
    label_logits = torch.gather(output, dim=2, index=input_ids_shifted)  # Shape: [8, 255, 1]

    loss_indexes = (shifted_labels !=-100) 
    # If you want to squeeze out the last dimension to get a tensor of shape [8, 255]:
    label_logits = label_logits.squeeze(-1)*loss_indexes  # Shape: [8, 255]

    with torch.no_grad():
        softmax_output = torch.log_softmax(output, dim=-1)
        labels_probabilities = torch.gather(softmax_output, dim=2, index=input_ids_shifted)  # Shape: [8, 255, 1]
        labels_probabilities = (beta*(labels_probabilities.squeeze(-1)*loss_indexes).sum(-1,keepdim=True)/loss_indexes.sum(-1,keepdim=True)).exp()   # Shape: [8, 255]
        labels_probabilities.detach()

    loss = ((label_logits * labels_probabilities).sum(-1)/loss_indexes.sum(-1)).mean()
    
    return loss