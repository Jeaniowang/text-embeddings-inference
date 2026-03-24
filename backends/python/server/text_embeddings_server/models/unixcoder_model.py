import os
import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from opentelemetry import trace

from collections import defaultdict
import numpy as np
from loguru import logger

from text_embeddings_server.models.pooling import DefaultPooling

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score, TokenEmbedding

tracer = trace.get_tracer(__name__)

is_unixcode = os.getenv("IS_UNIXCODE", None)

class UniXcoderModel(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str,
        trust_remote: bool = False,
    ):
        self.config = RobertaConfig.from_pretrained(model_path)
        self.config.is_decoder = True
        model = (
            RobertaModel.from_pretrained(model_path, config=self.config)
            .to(dtype)
            .to(device)
            .eval()
        )

        self.hidden_size = model.config.hidden_size 

        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = model.config.pad_token_id + 1
        if hasattr(model.config, "max_seq_length"):
            self.max_input_length = model.config.max_seq_length
        else:
            self.max_input_length = (
                model.config.max_position_embeddings - position_offset
            )

        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
                
        super(UniXcoderModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        return self._process_unixcode(batch, kwargs)

    def _process_unixcode(self, batch: PaddedBatch, kwargs: dict):
        tokens_ids = []
        mode = "<encoder-only>"
        mode_id = self.tokenizer.convert_tokens_to_ids(mode)
        for tokens_id in batch.input_ids:
            tokens_id = tokens_id[:self.max_input_length - 4]
            tokens_id = tokens_id.tolist()[1:-1]
            tokens_id = [self.tokenizer.cls_token_id, mode_id, self.tokenizer.sep_token_id] + tokens_id + [self.tokenizer.sep_token_id]

            tokens_ids.append(tokens_id)
        tokens_ids = torch.tensor(tokens_ids).to(self.device)
        mask = tokens_ids.ne(self.config.pad_token_id)
        attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2).to(self.device)
        token_embeddings = self.model(tokens_ids, attention_mask=attention_mask)[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return [
            Embedding(
                values=sentence_embeddings[i]
            )
            for i in range(len(batch))
        ]

