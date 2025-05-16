from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple
from pathlib import Path
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        flag: bool,
        words_file,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            self.labels = []
            for index, s in enumerate(self.strings):
                offset_mapping = []
                encoded: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_offsets_mapping=True,
                    return_tensors='pt'
                )
                encoding = encoded.input_ids[0]

                offset_mapping = encoded['offset_mapping']
                labels = encoding.clone()
                if flag:
                    for word in self.common_words[index]:
                        word = json.loads(word)
                        start, end = word['start'], word['end']

                        # Extract the relevant offset_mapping for the batch
                        offsets = offset_mapping[0]  # Shape: (275819, 2)
                        
                        # Extract start and end positions from offsets
                        encoded_start = offsets[:, 0]  # All start positions
                        encoded_end = offsets[:, 1]    # All end positions

                        # Create a mask for tokens that overlap with the word range
                        overlap_mask = (encoded_start < end) & (encoded_end > start)

                        # Apply the mask to labels
                        labels[overlap_mask] = -100

                print(encoding.shape)
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )

                encoding_labels = pad_or_trim_tensor(
                    labels,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )

                self.input_ids.append(encoding)
                print(len(self.input_ids))
                self.labels.append(encoding_labels)

            return; # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        data = read_text(file_path)
        encoded = tokenizer(data, add_special_tokens=False, return_offsets_mapping=True, return_tensors='pt')
        tokens = encoded.input_ids[0]
        offset_mapping = encoded['offset_mapping']
        labels = tokens.clone()

        if flag:
            if 'bert' in words_file:
                with open(words_file, 'r') as f:
                    common_words = json.load(f)
                for word in tqdm(common_words, desc="Processing common words", ncols=100):
                    word = json.loads(word)
                    start, end = word['start'], word['end']

                    # Extract the relevant offset_mapping for the batch
                    offsets = offset_mapping[0]  # Shape: (275819, 2)
                    
                    # Extract start and end positions from offsets
                    encoded_start = offsets[:, 0]  # All start positions
                    encoded_end = offsets[:, 1]    # All end positions

                    # Create a mask for tokens that overlap with the word range
                    overlap_mask = (encoded_start < end) & (encoded_end > start)

                    # Apply the mask to labels
                    labels[overlap_mask] = -100

        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]

            self.labels = [
                F.pad(
                    labels[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(labels), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]
            self.labels = [
                labels[i : i + max_len]
                for i in range(0, len(labels), max_len)
            ]


        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]
            self.labels[-1] = torch.concat(
                [self.labels[-1], self.labels[0]], dim=-1
            )[:max_len]

        # self.labels = self.input_ids.copy()
        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass    # def __init__()

    # def get_common_words(self):
    #     return self.common_words

    def __getitem__(self, index):
        # return self.input_ids[index]
        return self.input_ids[index], self.labels[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch_ids = torch.stack([pair[0] for pair in batch])
            batch_labels = torch.stack([pair[1] for pair in batch])
            return {
                "input_ids": batch_ids,
                "labels": batch_labels
            }

        return collate_fn



class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        flag: bool,
        common_words_file,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True
    ):
        self.flag = flag

        self.forget_dataset = DefaultDataset(
            forget_file_path, flag, common_words_file, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, False, common_words_file, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        # a =self.forget_dataset[index]
        # print(a[0].shape)
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            # batch_forget = torch.stack([pair[0] for pair in batch])

            batch_forget = torch.stack([pair[0][0] for pair in batch])
            batch_forget_labels = torch.stack([pair[0][1] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget_labels,
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                # batch_retain = torch.stack([pair[1] for pair in batch])
                batch_retain = torch.stack([pair[1][0] for pair in batch])
                batch_retain_labels = torch.stack([pair[1][1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain_labels,
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn
