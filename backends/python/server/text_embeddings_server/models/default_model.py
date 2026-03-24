import os
import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModel, AutoConfig, AutoTokenizer
from opentelemetry import trace

from collections import defaultdict
import numpy as np
from loguru import logger

from text_embeddings_server.models.pooling import DefaultPooling

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score, TokenEmbedding

tracer = trace.get_tracer(__name__)

IS_CAUSAL = os.getenv("IS_CAUSAL", "").lower()

class DefaultModel(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote)
            .to(dtype)
            .to(device)
            .eval()
        )
        self.hidden_size = model.config.hidden_size
        
        self.pool = pool

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

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        if self.pool == "splade":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.config = AutoConfig.from_pretrained(model_path)
            self.vocab_size = self.config.vocab_size
        
            self.sparse_linear = torch.nn.Linear(self.hidden_size, 1).to(device).to(dtype)
            sparse_model_path = os.path.join(model_path, "sparse_linear.pt")
            sparse_state_dict = torch.load(sparse_model_path, map_location="cpu", weights_only=True)
            self.sparse_linear.load_state_dict(sparse_state_dict)
        else:
            self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)
                
        super(DefaultModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids

        if self.pool == "splade":
            return self._process_splade(batch, kwargs)
        else:
            return self._process_default(batch, kwargs)

    def _process_splade(self, batch: PaddedBatch, kwargs: dict):
        with torch.no_grad():
            last_hidden_state = self.model(**kwargs, return_dict=True).last_hidden_state
        sparse_vecs = torch.relu(self.sparse_linear(last_hidden_state))
        token_weights = sparse_vecs.squeeze(-1)
        unused_tokens = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id}
        all_token_weights = torch.zeros((len(batch), self.vocab_size), dtype=token_weights.dtype, device=token_weights.device)
        
        for i in range(len(batch)):
            all_token_weights[i, batch.input_ids[i]] = token_weights[i]
        
        all_token_weights[:, list(unused_tokens)] = 0
        
        embeddings = [
            Embedding(
                values=all_token_weights[i].detach().cpu().numpy().tolist()
            )
            for i in range(len(batch))
        ]
        
        return embeddings
    
    def _process_default(self, batch: PaddedBatch, kwargs: dict):
        if IS_CAUSAL in ["false", "0"]:
            with torch.no_grad():
                output = self.model(**kwargs, is_causal=False)
        else:
            with torch.no_grad():
                output = self.model(**kwargs)
            
        embedding = self.pooling.forward(output, batch.attention_mask)
        cpu_results = embedding.view(-1).tolist()
        
        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
    
    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch):
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids
        output = self.model(**kwargs)
        embedding = output[0].contiguous()
        cpu_results = embedding.view(-1).tolist()
        embedding_result=[]
        for i in range(len(batch)):
            embedding_tmp=[
                Embedding(values=cpu_results[(j+i * batch.max_length) * self.hidden_size :
                                             (j + 1 + i * batch.max_length) * self.hidden_size])
                for j in range(batch.input_ids.size()[1])
            ]
            token_embeddings=TokenEmbedding(embeddings=embedding_tmp)
            embedding_result.append(token_embeddings)

        return embedding_result
    
    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        pass
