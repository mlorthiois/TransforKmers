from .tokenizer import DNATokenizer

from typing import Optional, Any, Tuple
import math
from dataclasses import dataclass
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForKmerLM(DataCollatorForLanguageModeling):
    tokenizer: DNATokenizer
    mlm_probability: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            self.mask_list = torch.arange(
                start=-math.floor((self.tokenizer.k - 1) / 2),
                end=math.ceil((self.tokenizer.k - 1) / 2) + 1,
            )
        except AttributeError:
            raise Exception(
                "This tokenizer does not have a kmer size set. Use a trained DNATokenizer."
            )
        self.mlm_probability = self.mlm_probability / len(self.mask_list)

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        mask_list = self.mask_list
        tokenizer = self.tokenizer

        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
                "Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        for batch_id, kmer_pos in (masked_indices == True).nonzero():
            seq_len = len(inputs[batch_id])
            for padding in mask_list:
                padded_pos = kmer_pos + padding

                # +1 for [CLS], and -1 for [SEP]
                if not padded_pos < 1 and not padded_pos > seq_len - 1:
                    masked_indices[batch_id, padded_pos] = True

        labels[~masked_indices] = -100

        # 80% with [MASK]
        indices_replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
        mask_idx = indices_replace & masked_indices
        inputs[mask_idx] = tokenizer.mask_token_id

        # 10% with random kmer
        random_idx = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replace
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[random_idx] = random_words[random_idx]

        return inputs, labels
