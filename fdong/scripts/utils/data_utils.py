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


class FixedUnitPatternData(Dataset):
    """
    Synthetic next-token dataset generated by sampling fixed token units.

    Metadata has two layers:
      layer 0: raw token id
      layer 1: sampled unit id
    """

    def __init__(
        self,
        max_seq_len,
        num_samples=100000,
        units=((1, 2, 3), (1, 2, 4)),
        probabilities=(0.7, 0.3),
        seed=0,
        padding=False,
        pad_token_id=0,
        return_metadata=False,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if not units:
            raise ValueError("units must not be empty")
        if len(units) != len(probabilities):
            raise ValueError("units and probabilities must have the same length")
        normalized_units = [tuple(int(token) for token in unit) for unit in units]
        if any(len(unit) == 0 for unit in normalized_units):
            raise ValueError("unit entries must not be empty")
        unit_lens = {len(unit) for unit in normalized_units}
        if len(unit_lens) != 1:
            raise ValueError("all units must have the same length")
        if any(float(prob) < 0 for prob in probabilities):
            raise ValueError("probabilities must be non-negative")
        prob_sum = sum(float(prob) for prob in probabilities)
        if prob_sum <= 0:
            raise ValueError("at least one probability must be positive")

        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.units = normalized_units
        self.probabilities = [float(prob) / prob_sum for prob in probabilities]
        self.seed = seed
        self.padding = padding
        self.pad_token_id = pad_token_id
        self.return_metadata = return_metadata
        self.unit_size = next(iter(unit_lens))

    def __len__(self):
        return self.num_samples

    def _rng(self, *items):
        seed = self.seed
        for item in items:
            seed = (seed * 1000003 + int(item)) % (2**32)
        return random.Random(seed)

    def _generate_tokens(self, index, with_metadata=False):
        rng = self._rng(1009, index)
        required_len = self.max_seq_len + 1
        tokens = []
        metadata = [] if with_metadata else None

        while len(tokens) < required_len:
            unit_idx = rng.choices(
                range(len(self.units)),
                weights=self.probabilities,
                k=1,
            )[0]
            for token_id in self.units[unit_idx]:
                tokens.append(token_id)
                if metadata is not None:
                    metadata.append([token_id, unit_idx])

        tokens = tokens[:required_len]
        if metadata is not None:
            metadata = metadata[:required_len]

        if self.padding and len(tokens) < required_len:
            tokens.extend([self.pad_token_id] * (required_len - len(tokens)))
            if metadata is not None:
                metadata.extend([[-1, -1] for _ in range(required_len - len(metadata))])

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


class StructuredLanguageData(Dataset):
    """
    Synthetic next-token dataset with topic spans, ambiguous entities, shared
    function tokens, long-ish dependencies, and filler noise.

    Metadata columns:
      0: syntax role id
      1: topic id
      2: entity id
      3: span id
      4: relation/template id
    """

    ROLE_NOISE = 0
    ROLE_TOPIC = 1
    ROLE_ENTITY = 2
    ROLE_VERB = 3
    ROLE_OBJECT = 4
    ROLE_FUNCTION = 5
    ROLE_COPY = 6

    TEMPLATE_STATEMENT = 0
    TEMPLATE_COPY = 1
    TEMPLATE_BRIDGE = 2

    def __init__(
        self,
        max_seq_len,
        num_samples=100000,
        topic_count=8,
        entities_per_topic=8,
        shared_entity_count=16,
        verb_count=12,
        function_token_count=12,
        noise_token_count=32,
        seed=0,
        pad_token_id=0,
        min_token_id=1,
        topic_zipf_alpha=1.1,
        noise_rate=0.25,
        ambiguity_rate=0.35,
        copy_rate=0.25,
        bridge_rate=0.25,
        min_span_units=2,
        max_span_units=8,
        padding=False,
        return_metadata=False,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if topic_count < 2:
            raise ValueError("topic_count must be at least 2")
        if entities_per_topic < 1:
            raise ValueError("entities_per_topic must be positive")
        if shared_entity_count < 1:
            raise ValueError("shared_entity_count must be positive")
        if verb_count < 1 or function_token_count < 1 or noise_token_count < 1:
            raise ValueError("verb/function/noise token counts must be positive")
        if topic_zipf_alpha <= 0:
            raise ValueError("topic_zipf_alpha must be positive")
        for name, value in [
            ("noise_rate", noise_rate),
            ("ambiguity_rate", ambiguity_rate),
            ("copy_rate", copy_rate),
            ("bridge_rate", bridge_rate),
        ]:
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if min_span_units < 1 or max_span_units < min_span_units:
            raise ValueError("span unit bounds are invalid")

        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.topic_count = topic_count
        self.entities_per_topic = entities_per_topic
        self.shared_entity_count = shared_entity_count
        self.verb_count = verb_count
        self.function_token_count = function_token_count
        self.noise_token_count = noise_token_count
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.min_token_id = min_token_id
        self.topic_zipf_alpha = topic_zipf_alpha
        self.noise_rate = noise_rate
        self.ambiguity_rate = ambiguity_rate
        self.copy_rate = copy_rate
        self.bridge_rate = bridge_rate
        self.min_span_units = min_span_units
        self.max_span_units = max_span_units
        self.padding = padding
        self.return_metadata = return_metadata

        cursor = self.min_token_id
        self.topic_tokens = list(range(cursor, cursor + self.topic_count))
        cursor += self.topic_count
        self.private_entity_tokens = []
        for _topic_idx in range(self.topic_count):
            topic_entities = list(range(cursor, cursor + self.entities_per_topic))
            self.private_entity_tokens.append(topic_entities)
            cursor += self.entities_per_topic
        self.shared_entity_tokens = list(range(cursor, cursor + self.shared_entity_count))
        cursor += self.shared_entity_count
        self.verb_tokens = list(range(cursor, cursor + self.verb_count))
        cursor += self.verb_count
        self.function_tokens = list(range(cursor, cursor + self.function_token_count))
        cursor += self.function_token_count
        self.noise_tokens = list(range(cursor, cursor + self.noise_token_count))
        cursor += self.noise_token_count
        self.vocab_upper_bound = cursor
        self.topic_weights = [1.0 / ((idx + 1) ** self.topic_zipf_alpha) for idx in range(self.topic_count)]

    def __len__(self):
        return self.num_samples

    def _rng(self, *items):
        seed = self.seed
        for item in items:
            seed = (seed * 1000003 + int(item)) % (2**32)
        return random.Random(seed)

    def _append(self, tokens, metadata, token_id, role, topic_id, entity_id, span_id, template_id):
        tokens.append(int(token_id))
        if metadata is not None:
            metadata.append([
                int(role),
                int(topic_id),
                int(entity_id),
                int(span_id),
                int(template_id),
            ])

    def _sample_topic(self, rng):
        return rng.choices(range(self.topic_count), weights=self.topic_weights, k=1)[0]

    def _sample_entity(self, rng, topic_id):
        local_entity = rng.randrange(self.entities_per_topic)
        if rng.random() < self.ambiguity_rate:
            shared_idx = rng.randrange(self.shared_entity_count)
            return self.shared_entity_tokens[shared_idx], self.topic_count * self.entities_per_topic + shared_idx
        return self.private_entity_tokens[topic_id][local_entity], topic_id * self.entities_per_topic + local_entity

    def _maybe_noise(self, rng, tokens, metadata, span_id, template_id, topic_id=-1):
        if rng.random() >= self.noise_rate:
            return
        noise_len = 1 + int(rng.random() < 0.35)
        for _ in range(noise_len):
            self._append(
                tokens,
                metadata,
                rng.choice(self.noise_tokens),
                self.ROLE_NOISE,
                topic_id,
                -1,
                span_id,
                template_id,
            )

    def _emit_statement(self, rng, tokens, metadata, topic_id, span_id):
        template_id = self.TEMPLATE_STATEMENT
        entity_token, entity_id = self._sample_entity(rng, topic_id)
        object_token, object_id = self._sample_entity(rng, topic_id)
        verb_token = self.verb_tokens[(topic_id + entity_id) % self.verb_count]
        self._append(tokens, metadata, self.topic_tokens[topic_id], self.ROLE_TOPIC, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, rng.choice(self.function_tokens), self.ROLE_FUNCTION, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, entity_token, self.ROLE_ENTITY, topic_id, entity_id, span_id, template_id)
        self._maybe_noise(rng, tokens, metadata, span_id, template_id, topic_id)
        self._append(tokens, metadata, verb_token, self.ROLE_VERB, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, object_token, self.ROLE_OBJECT, topic_id, object_id, span_id, template_id)
        self._append(tokens, metadata, rng.choice(self.function_tokens), self.ROLE_FUNCTION, topic_id, -1, span_id, template_id)

    def _emit_copy(self, rng, tokens, metadata, topic_id, span_id):
        template_id = self.TEMPLATE_COPY
        entity_token, entity_id = self._sample_entity(rng, topic_id)
        gap = rng.randint(1, 4)
        self._append(tokens, metadata, self.topic_tokens[topic_id], self.ROLE_TOPIC, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, entity_token, self.ROLE_ENTITY, topic_id, entity_id, span_id, template_id)
        for _ in range(gap):
            self._maybe_noise(rng, tokens, metadata, span_id, template_id, topic_id)
            self._append(tokens, metadata, rng.choice(self.function_tokens), self.ROLE_FUNCTION, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, self.verb_tokens[entity_id % self.verb_count], self.ROLE_VERB, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, entity_token, self.ROLE_COPY, topic_id, entity_id, span_id, template_id)

    def _emit_bridge(self, rng, tokens, metadata, topic_id, span_id):
        template_id = self.TEMPLATE_BRIDGE
        other_topic = self._sample_topic(rng)
        if other_topic == topic_id:
            other_topic = (topic_id + 1) % self.topic_count
        left_token, left_entity = self._sample_entity(rng, topic_id)
        right_token, right_entity = self._sample_entity(rng, other_topic)
        self._append(tokens, metadata, self.topic_tokens[topic_id], self.ROLE_TOPIC, topic_id, -1, span_id, template_id)
        self._append(tokens, metadata, left_token, self.ROLE_ENTITY, topic_id, left_entity, span_id, template_id)
        self._append(tokens, metadata, rng.choice(self.function_tokens), self.ROLE_FUNCTION, -1, -1, span_id, template_id)
        self._append(tokens, metadata, self.topic_tokens[other_topic], self.ROLE_TOPIC, other_topic, -1, span_id, template_id)
        self._append(tokens, metadata, right_token, self.ROLE_OBJECT, other_topic, right_entity, span_id, template_id)

    def _generate_tokens(self, index, with_metadata=False):
        rng = self._rng(1009, index)
        required_len = self.max_seq_len + 1
        tokens = []
        metadata = [] if with_metadata else None
        span_id = 0
        while len(tokens) < required_len:
            topic_id = self._sample_topic(rng)
            span_units = rng.randint(self.min_span_units, self.max_span_units)
            for _ in range(span_units):
                roll = rng.random()
                if roll < self.copy_rate:
                    self._emit_copy(rng, tokens, metadata, topic_id, span_id)
                elif roll < self.copy_rate + self.bridge_rate:
                    self._emit_bridge(rng, tokens, metadata, topic_id, span_id)
                else:
                    self._emit_statement(rng, tokens, metadata, topic_id, span_id)
                if len(tokens) >= required_len:
                    break
            span_id += 1

        tokens = tokens[:required_len]
        if metadata is not None:
            metadata = metadata[:required_len]

        if self.padding and len(tokens) < required_len:
            pad_len = required_len - len(tokens)
            tokens.extend([self.pad_token_id] * pad_len)
            if metadata is not None:
                metadata.extend([[-1, -1, -1, -1, -1] for _ in range(pad_len)])

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


class ControlledReusedTokenData(Dataset):
    """
    Synthetic dataset with controlled slot-level input/output reuse.

    Each hierarchy level builds fixed-size slots:
      slot = (u_1, ..., u_{slot_size - 1}, v)
      input = (u_1, ..., u_{slot_size - 1})
      output = v

    Slots are split into three disjoint structural groups:
      1. same input, different output
      2. different input, same output
      3. normal one-to-one

    Higher layers use lower-layer unit ids as their symbols. The generated
    top-layer units are recursively flattened back to raw token ids so the
    dataset remains compatible with causal LM training.
    """

    SLOT_TYPE_NORMAL = 0
    SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT = 1
    SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT = 2

    def __init__(
        self,
        max_seq_len,
        num_samples=100000,
        slot_size=4,
        num_hierarchy_layers=2,
        content_token_count=512,
        num_units_per_layer=512,
        seed=0,
        pad_token_id=0,
        min_token_id=1,
        same_input_diff_output_rate=0.3,
        same_input_diff_output_size=4,
        same_input_diff_output_distribution="zipf",
        same_input_diff_output_zipf_alpha=1.0,
        diff_input_same_output_rate=0.3,
        diff_input_same_output_size=4,
        diff_input_same_output_distribution="zipf",
        diff_input_same_output_zipf_alpha=1.0,
        top_sampling_distribution="zipf",
        top_sampling_zipf_alpha=1.0,
        padding=False,
        return_metadata=False,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.slot_size = slot_size
        self.num_hierarchy_layers = num_hierarchy_layers
        self.content_token_count = content_token_count
        self.num_units_per_layer = num_units_per_layer
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.min_token_id = min_token_id
        self.same_input_diff_output_rate = same_input_diff_output_rate
        self.same_input_diff_output_size = same_input_diff_output_size
        self.same_input_diff_output_distribution = same_input_diff_output_distribution
        self.same_input_diff_output_zipf_alpha = same_input_diff_output_zipf_alpha
        self.diff_input_same_output_rate = diff_input_same_output_rate
        self.diff_input_same_output_size = diff_input_same_output_size
        self.diff_input_same_output_distribution = diff_input_same_output_distribution
        self.diff_input_same_output_zipf_alpha = diff_input_same_output_zipf_alpha
        self.top_sampling_distribution = top_sampling_distribution
        self.top_sampling_zipf_alpha = top_sampling_zipf_alpha
        self.padding = padding
        self.return_metadata = return_metadata

        self._validate_args()
        self.units_by_layer = self._build_units()
        self.top_layer = self.num_hierarchy_layers - 1
        self.top_units = self.units_by_layer[self.top_layer]
        self.top_unit_sample_weights = self._build_top_unit_sample_weights()

    def __len__(self):
        return self.num_samples

    def _validate_args(self):
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if self.slot_size < 2:
            raise ValueError("slot_size must be at least 2")
        if self.num_hierarchy_layers < 1:
            raise ValueError("num_hierarchy_layers must be at least 1")
        if self.content_token_count < self.slot_size:
            raise ValueError("content_token_count must be >= slot_size")
        if self.num_units_per_layer < 1:
            raise ValueError("num_units_per_layer must be positive")
        if self.same_input_diff_output_rate < 0 or self.diff_input_same_output_rate < 0:
            raise ValueError("reuse rates must be non-negative")
        if self.same_input_diff_output_rate + self.diff_input_same_output_rate > 1.0:
            raise ValueError(
                "same_input_diff_output_rate + diff_input_same_output_rate must be <= 1"
            )
        if self.same_input_diff_output_size < 2:
            raise ValueError("same_input_diff_output_size must be at least 2")
        if self.diff_input_same_output_size < 2:
            raise ValueError("diff_input_same_output_size must be at least 2")

        valid_distributions = {"uniform", "zipf"}
        for name, value in [
            ("same_input_diff_output_distribution", self.same_input_diff_output_distribution),
            ("diff_input_same_output_distribution", self.diff_input_same_output_distribution),
            ("top_sampling_distribution", self.top_sampling_distribution),
        ]:
            if value not in valid_distributions:
                raise ValueError(f"{name} must be one of {sorted(valid_distributions)}")
        if self.same_input_diff_output_zipf_alpha <= 0:
            raise ValueError("same_input_diff_output_zipf_alpha must be positive")
        if self.diff_input_same_output_zipf_alpha <= 0:
            raise ValueError("diff_input_same_output_zipf_alpha must be positive")
        if self.top_sampling_zipf_alpha <= 0:
            raise ValueError("top_sampling_zipf_alpha must be positive")

    def _rng(self, *items):
        seed = self.seed
        for item in items:
            seed = (seed * 1000003 + int(item)) % (2**32)
        return random.Random(seed)

    def _num_group_slots(self, rate, group_size):
        requested = int(self.num_units_per_layer * rate)
        return (requested // group_size) * group_size

    def _split_symbols(self, symbols, rng, same_output_count, diff_output_count, normal_output_count):
        shuffled = list(symbols)
        rng.shuffle(shuffled)

        num_symbols = len(shuffled)
        required_count = same_output_count + diff_output_count + normal_output_count
        if required_count > num_symbols:
            raise ValueError(
                "controlled disjoint generation needs more unique output symbols than available; "
                "increase content_token_count/num_units_per_layer or reduce rates/sizes"
            )

        extra_count = num_symbols - required_count
        total_rate = self.same_input_diff_output_rate + self.diff_input_same_output_rate
        normal_rate = max(0.0, 1.0 - total_rate)
        rate_sum = total_rate + normal_rate

        same_extra = int(extra_count * (self.same_input_diff_output_rate / rate_sum))
        diff_extra = int(extra_count * (self.diff_input_same_output_rate / rate_sum))
        normal_extra = extra_count - same_extra - diff_extra

        same_count = same_output_count + same_extra
        diff_count = diff_output_count + diff_extra
        normal_count = normal_output_count + normal_extra

        same_symbols = shuffled[:same_count]
        diff_symbols = shuffled[same_count:same_count + diff_count]
        normal_symbols = shuffled[same_count + diff_count:same_count + diff_count + normal_count]

        return same_symbols, diff_symbols, normal_symbols

    def _group_weights(self, size, distribution, alpha):
        if distribution == "uniform":
            return [1.0] * size
        return [1.0 / ((idx + 1) ** alpha) for idx in range(size)]

    def _sample_prefix(self, rng, symbols, used_prefixes):
        if len(symbols) < 1:
            raise ValueError("cannot sample prefix from an empty symbol set")

        for _ in range(10000):
            prefix = tuple(rng.choice(symbols) for _ in range(self.slot_size - 1))
            if prefix not in used_prefixes:
                used_prefixes.add(prefix)
                return prefix
        raise ValueError("failed to sample a unique prefix; increase symbol count")

    def _sample_outputs(self, rng, symbols, size, used_outputs):
        candidates = [symbol for symbol in symbols if symbol not in used_outputs]
        if len(candidates) < size:
            raise ValueError(
                "not enough output symbols for disjoint controlled groups; "
                "increase content_token_count/num_units_per_layer or reduce rates/sizes"
            )
        outputs = rng.sample(candidates, size)
        used_outputs.update(outputs)
        return outputs

    def _build_units(self):
        units_by_layer = []
        layer_symbols = list(range(self.min_token_id, self.min_token_id + self.content_token_count))

        for layer_idx in range(self.num_hierarchy_layers):
            rng = self._rng(17, layer_idx)
            layer_units = self._build_layer_units(layer_idx, layer_symbols, rng)
            units_by_layer.append(layer_units)
            layer_symbols = list(range(len(layer_units)))

        return units_by_layer

    def _build_layer_units(self, layer_idx, symbols, rng):
        same_slots = self._num_group_slots(
            self.same_input_diff_output_rate,
            self.same_input_diff_output_size,
        )
        diff_slots = self._num_group_slots(
            self.diff_input_same_output_rate,
            self.diff_input_same_output_size,
        )
        normal_slots = self.num_units_per_layer - same_slots - diff_slots
        same_output_count = same_slots
        diff_output_count = diff_slots // self.diff_input_same_output_size
        normal_output_count = normal_slots
        same_symbols, diff_symbols, normal_symbols = self._split_symbols(
            symbols,
            rng,
            same_output_count,
            diff_output_count,
            normal_output_count,
        )

        units = []
        used_prefixes = set()
        used_outputs = set()
        group_id = 0

        same_groups = same_slots // self.same_input_diff_output_size
        same_weights = self._group_weights(
            self.same_input_diff_output_size,
            self.same_input_diff_output_distribution,
            self.same_input_diff_output_zipf_alpha,
        )
        for _ in range(same_groups):
            prefix = self._sample_prefix(rng, same_symbols, used_prefixes)
            outputs = self._sample_outputs(
                rng,
                same_symbols,
                self.same_input_diff_output_size,
                used_outputs,
            )
            for output, weight in zip(outputs, same_weights):
                units.append({
                    "symbols": tuple(prefix) + (output,),
                    "slot_type": self.SLOT_TYPE_SAME_INPUT_DIFF_OUTPUT,
                    "group_id": group_id,
                    "variant_weight": float(weight),
                })
            group_id += 1

        diff_groups = diff_slots // self.diff_input_same_output_size
        diff_weights = self._group_weights(
            self.diff_input_same_output_size,
            self.diff_input_same_output_distribution,
            self.diff_input_same_output_zipf_alpha,
        )
        for _ in range(diff_groups):
            outputs = self._sample_outputs(rng, diff_symbols, 1, used_outputs)
            output = outputs[0]
            for weight in diff_weights:
                prefix = self._sample_prefix(rng, diff_symbols, used_prefixes)
                units.append({
                    "symbols": tuple(prefix) + (output,),
                    "slot_type": self.SLOT_TYPE_DIFF_INPUT_SAME_OUTPUT,
                    "group_id": group_id,
                    "variant_weight": float(weight),
                })
            group_id += 1

        normal_outputs = self._sample_outputs(rng, normal_symbols, normal_slots, used_outputs)
        for output in normal_outputs:
            prefix = self._sample_prefix(rng, normal_symbols, used_prefixes)
            units.append({
                "symbols": tuple(prefix) + (output,),
                "slot_type": self.SLOT_TYPE_NORMAL,
                "group_id": group_id,
                "variant_weight": 1.0,
            })
            group_id += 1

        rng.shuffle(units)
        for unit_idx, unit in enumerate(units):
            unit["unit_id"] = unit_idx
            unit["layer_idx"] = layer_idx
        return units

    def _build_top_unit_sample_weights(self):
        weights = [float(unit["variant_weight"]) for unit in self.top_units]
        if self.top_sampling_distribution == "uniform":
            return weights

        zipf_weights = [
            1.0 / ((idx + 1) ** self.top_sampling_zipf_alpha)
            for idx in range(len(self.top_units))
        ]
        rng = self._rng(7919)
        rng.shuffle(zipf_weights)
        return [weight * zipf_weight for weight, zipf_weight in zip(weights, zipf_weights)]

    def _flatten_unit(
        self,
        layer_idx,
        unit_idx,
        output,
        metadata=None,
        ancestor_unit_ids=None,
        ancestor_slot_type_ids=None,
        ancestor_group_ids=None,
    ):
        if ancestor_unit_ids is None:
            ancestor_unit_ids = [-1] * self.num_hierarchy_layers
        if ancestor_slot_type_ids is None:
            ancestor_slot_type_ids = [-1] * self.num_hierarchy_layers
        if ancestor_group_ids is None:
            ancestor_group_ids = [-1] * self.num_hierarchy_layers

        ancestor_unit_ids = list(ancestor_unit_ids)
        ancestor_slot_type_ids = list(ancestor_slot_type_ids)
        ancestor_group_ids = list(ancestor_group_ids)

        unit = self.units_by_layer[layer_idx][unit_idx]
        ancestor_unit_ids[layer_idx] = int(unit["unit_id"])
        ancestor_slot_type_ids[layer_idx] = int(unit["slot_type"])
        ancestor_group_ids[layer_idx] = int(unit["group_id"])

        if layer_idx == 0:
            for local_pos, token_id in enumerate(unit["symbols"]):
                output.append(int(token_id))
                if metadata is not None:
                    metadata["unit_ids_by_layer"].append(list(ancestor_unit_ids))
                    metadata["slot_type_ids_by_layer"].append(list(ancestor_slot_type_ids))
                    metadata["group_ids_by_layer"].append(list(ancestor_group_ids))
                    metadata["position_in_base_slot"].append(int(local_pos))
            return

        for child_idx in unit["symbols"]:
            self._flatten_unit(
                layer_idx - 1,
                int(child_idx),
                output,
                metadata,
                ancestor_unit_ids,
                ancestor_slot_type_ids,
                ancestor_group_ids,
            )

    def _generate_tokens(self, index, with_metadata=False):
        rng = self._rng(1009, index)
        required_len = self.max_seq_len + 1
        tokens = []
        metadata = None
        if with_metadata:
            metadata = {
                "unit_ids_by_layer": [],
                "slot_type_ids_by_layer": [],
                "group_ids_by_layer": [],
                "position_in_base_slot": [],
            }

        while len(tokens) < required_len:
            unit_idx = rng.choices(
                range(len(self.top_units)),
                weights=self.top_unit_sample_weights,
                k=1,
            )[0]
            self._flatten_unit(self.top_layer, unit_idx, tokens, metadata)

        tokens = tokens[:required_len]
        if metadata is not None:
            for key in metadata:
                metadata[key] = metadata[key][:required_len]

        if self.padding and len(tokens) < required_len:
            pad_len = required_len - len(tokens)
            tokens.extend([self.pad_token_id] * pad_len)
            if metadata is not None:
                metadata["unit_ids_by_layer"].extend([[-1] * self.num_hierarchy_layers for _ in range(pad_len)])
                metadata["slot_type_ids_by_layer"].extend([[-1] * self.num_hierarchy_layers for _ in range(pad_len)])
                metadata["group_ids_by_layer"].extend([[-1] * self.num_hierarchy_layers for _ in range(pad_len)])
                metadata["position_in_base_slot"].extend([-1] * pad_len)

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        if metadata is None:
            return token_tensor

        metadata_tensors = {
            "unit_ids_by_layer": torch.tensor(metadata["unit_ids_by_layer"], dtype=torch.long),
            "slot_type_ids_by_layer": torch.tensor(metadata["slot_type_ids_by_layer"], dtype=torch.long),
            "group_ids_by_layer": torch.tensor(metadata["group_ids_by_layer"], dtype=torch.long),
            "position_in_base_slot": torch.tensor(metadata["position_in_base_slot"], dtype=torch.long),
        }
        return token_tensor, metadata_tensors

    def get_metadata(self, index):
        token_ids, metadata = self._generate_tokens(index, with_metadata=True)
        metadata = dict(metadata)
        metadata["token_ids"] = token_ids
        return metadata

    def get_layer_units(self, layer_idx):
        return self.units_by_layer[layer_idx]

    def __getitem__(self, index):
        if self.return_metadata:
            token_ids, metadata = self._generate_tokens(index, with_metadata=True)
            sliced_metadata = {
                key: value[:-1]
                for key, value in metadata.items()
            }
            return token_ids[:-1], token_ids[1:], len(token_ids), sliced_metadata

        token_ids = self._generate_tokens(index)
        return token_ids[:-1], token_ids[1:], len(token_ids)
