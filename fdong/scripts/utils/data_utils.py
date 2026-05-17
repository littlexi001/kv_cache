import os
import random
import torch
import json

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map


class TokenizedJSONLData(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding=True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        # 递归收集所有子文件夹中的jsonl文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    self.files.append(os.path.join(root, filename))
        self.files = sorted(self.files)
        # self.files = self.files[0*8800:0*8800+352]+self.files[1*8800:1*8800+210]+self.files[2*8800:2*8800+210]+self.files[3*8800:3*8800+210]
        
        self._load_file(0)  # 加载第一个文件
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = 1 # int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[self.cur_file_idx], 'r', encoding='utf-8') as f:
            # 直接读取每行作为文本，去除首尾空白字符
            self.file_texts = [line.strip() for line in f if line.strip()]
            # 过滤掉空行

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        
        # 计算行索引并确保不越界
        line_idx = index % self.lines_per_file
        max_line_idx = len(self.file_texts) // self.combine_sequence - 1
        line_idx = min(line_idx, max_line_idx)

        # 组合多个文本片段
        # start_idx = line_idx * self.combine_sequence
        # end_idx = start_idx + self.combine_sequence
        text = json.loads(self.file_texts[line_idx]) # ''.join(self.file_texts[start_idx:end_idx])

        # 处理tokenization
        if self.padding:
            token_ids = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        
        return token_ids[0, :-1], token_ids[0, 1:], real_len


class HierarchicalPatternData(Dataset):
    """
    Synthetic next-token dataset generated from repeated hierarchical token patterns.

    The dataset builds a small grammar:
      layer 0 unit: block_size raw token ids
      layer k unit: block_size units from layer k - 1

    A sequence is produced by sampling top-level units and flattening them back to
    token ids. This creates repeated compositional structure while keeping the
    output compatible with the existing causal LM training loop.
    """

    def __init__(
        self,
        max_seq_len,
        num_samples=100000,
        block_size=4,
        num_hierarchy_layers=2,
        content_token_count=2048,
        num_units_per_layer=256,
        seed=0,
        pad_token_id=0,
        min_token_id=1,
        sampling_distribution="uniform",
        zipf_alpha=1.0,
        zipf_shuffle_ranks=True,
        padding=False,
        return_metadata=False,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if num_hierarchy_layers < 1:
            raise ValueError("num_hierarchy_layers must be at least 1")
        if content_token_count < block_size:
            raise ValueError("content_token_count must be >= block_size")
        if num_units_per_layer < block_size:
            raise ValueError("num_units_per_layer must be >= block_size")
        if sampling_distribution not in ("uniform", "zipf"):
            raise ValueError("sampling_distribution must be 'uniform' or 'zipf'")
        if zipf_alpha <= 0:
            raise ValueError("zipf_alpha must be positive")

        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.block_size = block_size
        self.num_hierarchy_layers = num_hierarchy_layers
        self.content_token_count = content_token_count
        self.num_units_per_layer = num_units_per_layer
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.min_token_id = min_token_id
        self.sampling_distribution = sampling_distribution
        self.zipf_alpha = zipf_alpha
        self.zipf_shuffle_ranks = zipf_shuffle_ranks
        self.padding = padding
        self.return_metadata = return_metadata

        self.units_by_layer = self._build_units()
        self.top_layer = num_hierarchy_layers - 1
        self.top_units = self.units_by_layer[self.top_layer]
        self.top_unit_sample_weights = self._build_top_unit_sample_weights()

    def __len__(self):
        return self.num_samples

    def _rng(self, *items):
        seed = self.seed
        for item in items:
            seed = (seed * 1000003 + int(item)) % (2**32)
        return random.Random(seed)

    def _build_units(self):
        rng = self._rng(17)
        token_ids = list(range(self.min_token_id, self.min_token_id + self.content_token_count))

        units_by_layer = []
        base_units = []
        for _ in range(self.num_units_per_layer):
            unit = tuple(rng.choice(token_ids) for _ in range(self.block_size))
            base_units.append(unit)
        units_by_layer.append(base_units)

        for layer_idx in range(1, self.num_hierarchy_layers):
            previous_units = units_by_layer[layer_idx - 1]
            current_units = []
            for _ in range(self.num_units_per_layer):
                child_indices = [
                    rng.randrange(len(previous_units))
                    for _ in range(self.block_size)
                ]
                current_units.append(tuple(child_indices))
            units_by_layer.append(current_units)

        return units_by_layer

    def _build_top_unit_sample_weights(self):
        if self.sampling_distribution == "uniform":
            return None

        num_top_units = len(self.top_units)
        ranks = list(range(1, num_top_units + 1))
        weights = [1.0 / (rank ** self.zipf_alpha) for rank in ranks]
        if self.zipf_shuffle_ranks:
            rng = self._rng(7919)
            rng.shuffle(weights)
        return weights

    def _flatten_unit(self, layer_idx, unit_idx, output, metadata=None, ancestor_unit_ids=None):
        if ancestor_unit_ids is None:
            ancestor_unit_ids = [-1] * self.num_hierarchy_layers
        ancestor_unit_ids = list(ancestor_unit_ids)
        ancestor_unit_ids[layer_idx] = unit_idx

        unit = self.units_by_layer[layer_idx][unit_idx]
        if layer_idx == 0:
            for token_id in unit:
                output.append(token_id)
                if metadata is not None:
                    metadata.append(list(ancestor_unit_ids))
            return
        for child_idx in unit:
            self._flatten_unit(layer_idx - 1, child_idx, output, metadata, ancestor_unit_ids)

    def _generate_tokens(self, index, with_metadata=False):
        rng = self._rng(1009, index)
        required_len = self.max_seq_len + 1
        tokens = []
        metadata = [] if with_metadata else None

        while len(tokens) < required_len:
            if self.top_unit_sample_weights is None:
                unit_idx = rng.randrange(len(self.top_units))
            else:
                unit_idx = rng.choices(
                    range(len(self.top_units)),
                    weights=self.top_unit_sample_weights,
                    k=1,
                )[0]
            self._flatten_unit(self.top_layer, unit_idx, tokens, metadata)

        tokens = tokens[:required_len]
        if metadata is not None:
            metadata = metadata[:required_len]

        if self.padding and len(tokens) < required_len:
            tokens.extend([self.pad_token_id] * (required_len - len(tokens)))
            if metadata is not None:
                metadata.extend([[-1] * self.num_hierarchy_layers for _ in range(required_len - len(metadata))])

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        if metadata is None:
            return token_tensor
        metadata_tensor = torch.tensor(metadata, dtype=torch.long)
        return token_tensor, metadata_tensor

    def get_metadata(self, index):
        token_ids, metadata = self._generate_tokens(index, with_metadata=True)
        return {
            "token_ids": token_ids,
            "unit_ids_by_layer": metadata,
        }

    def __getitem__(self, index):
        if self.return_metadata:
            token_ids, metadata = self._generate_tokens(index, with_metadata=True)
            return token_ids[:-1], token_ids[1:], len(token_ids), metadata[:-1]
        token_ids = self._generate_tokens(index)
        return token_ids[:-1], token_ids[1:], len(token_ids)
