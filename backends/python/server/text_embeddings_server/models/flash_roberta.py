import json
import torch
import torch_npu
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from typing import Type, List, Union
from safetensors import safe_open
from safetensors.torch import save_file
from loguru import logger

from transformers.models.roberta import RobertaConfig
from transformers import AutoTokenizer
from opentelemetry import trace
from text_embeddings_server.models import Model

from text_embeddings_server.models.flash_bert import (BertEncoder, 
                                                      FastLayerNorm)

from text_embeddings_server.models.types import FlashBatch, PaddedBatch, Embedding, Score, TokenEmbedding


tracer = trace.get_tracer(__name__)

TOKEN_TYPE_SHIFT = 30

class RobertaEmbeddings:
    def __init__(self, prefix, handle, device, dtype, config: RobertaConfig):
        self.config = config
        self.word_embeddings_weight = (
            handle.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        )
        self.token_type_embeddings_weight = (
            handle.get_tensor(f"{prefix}.token_type_embeddings.weight")
            .to(dtype)
            .to(device)
        )

        if config.position_embedding_type == "absolute":
            self.position_embeddings_weight = (
                handle.get_tensor(f"{prefix}.position_embeddings.weight")
                .to(dtype)
                .to(device)
            )
        else:
            raise NotImplementedError(
                "FlashBert only supports absolute position embeddings"
            )

        self.layer_norm = FastLayerNorm(
            f"{prefix}.LayerNorm", handle, device, dtype, config
        )

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeds = nn.functional.embedding(
            token_type_ids, self.token_type_embeddings_weight
        )
        position_embeds = nn.functional.embedding(
            position_ids, self.position_embeddings_weight, padding_idx = self.config.pad_token_id
        )

        inputs_embeds += position_embeds
        
        embeddings, _ = self.layer_norm.forward(inputs_embeds, token_type_embeds)
        return embeddings

def replace_roberta_positions(
    input_ids: torch.Tensor, position_ids: torch.Tensor, padding_idx: int
) -> None:
    position_ids += padding_idx + 1
    

class FlashRobertaModel(nn.Module):
    def __init__(self, handle, device, dtype, config: RobertaConfig, prefix: str = ""):
        super().__init__()
        self.embeddings = RobertaEmbeddings(f"{prefix}embeddings", handle, device, dtype, config)
        self.encoder = BertEncoder(f"{prefix}encoder", handle, device, dtype, config)
        self.padding_idx: int = config.pad_token_id

    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):        
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s, attn_mask)
        if mask is not None:
            outputs = encoder_outputs[mask]
            return outputs
        return encoder_outputs
    
    
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, handle, device, dtype, model_config):
        super().__init__()
        config = model_config
        
        self.classifier_dense_weight = (
            handle.get_tensor(f"classifier.dense.weight").to(dtype).to(device)
        )
        self.classifier_dense_bias = (
            handle.get_tensor(f"classifier.dense.bias").to(dtype).to(device)
        )
           
        self.classifier_out_proj_weight = (
            handle.get_tensor(f"classifier.out_proj.weight").to(dtype).to(device)
        )
        self.classifier_out_proj_bias = (
            handle.get_tensor(f"classifier.out_proj.bias").to(dtype).to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLSPool has already been applied in `pooling`
        x = F.linear(
            x, self.classifier_dense_weight, self.classifier_dense_bias
        )
        x = torch.tanh(x)
        x = F.linear(
            x, self.classifier_out_proj_weight, self.classifier_out_proj_bias
        )
        return x
 
    
class RobertaForSequenceClassification(nn.Module):
    def __init__(self, handle, device, dtype, config: RobertaConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        
        self.roberta = FlashRobertaModel(handle, device, dtype, config, prefix)
        self.classifier = RobertaClassificationHead(handle, device, dtype, config)
    
    def forward(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            cu_seqlens,
            max_s,
            mask=None,
            attn_mask=None,
            ):

        sequence_output = self.roberta(input_ids,
            token_type_ids,
            position_ids,
            cu_seqlens,
            max_s,
            mask=None,
            attn_mask=None)
        embedding = sequence_output[cu_seqlens[:-1]]
        logits =  self.classifier(embedding)
        return logits


class FlashRoberta(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        config = RobertaConfig.from_pretrained(model_path)

        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings
            
        safe_weight_path = model_path / "model.safetensors"
        bin_weight_path = model_path / "pytorch_model.bin"
        if not safe_weight_path.exists() and not bin_weight_path.exists():
            logger.error(f"pytorch_model.bin and model.safetensors do not exist")
            raise FileNotFoundError(f"pytorch_model.bin and model.safetensors do not exist")
        if not safe_weight_path.exists():
            logger.info(f"model.safetensors does not exist, translate pytorch_model.bin to model.safetensors")
            stat_dict = torch.load(bin_weight_path.as_posix(), map_location=torch.device('cpu'))
            save_file(stat_dict, safe_weight_path.as_posix())
            
        with safe_open(model_path / "model.safetensors", framework="pt") as f:
            if config.architectures[0].endswith("Classification"):
                model = RobertaForSequenceClassification(f, device, dtype, config, "roberta.")
            else:    
                model = FlashRobertaModel(f, device, dtype, config)
                
        self.device = device
        self.dtype = dtype
        self.hidden_size = config.hidden_size

        super(FlashRoberta, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Union[FlashBatch, PaddedBatch]:
        # for hpu devices, we use PaddedBatch as we do not have real varlen fwd yet
        return FlashBatch if self.device.type != "hpu" else PaddedBatch
    
    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        if not isinstance(batch, FlashBatch):
            raise ValueError("only FlashBatch is supported")

        cu_seqlens = batch.cu_seqlens
        mask = None
        attn_mask = None
        max_input_lens = batch.max_s
        
        all_embedding = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
        embedding = all_embedding[cu_seqlens[:-1]]
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
        
    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: Union[FlashBatch, PaddedBatch]):
        if not isinstance(batch, FlashBatch):
            raise ValueError("only FlashBatch is supported")

        cu_seqlens = batch.cu_seqlens
        mask = None
        attn_mask = None
        max_input_lens = batch.max_s
        batch_size = batch.size
        
        all_embedding = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
        
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        cpu_results = all_embedding.tolist()
        embedding_result=[]
        for i in range(batch_size):
            embedding_tmp=[
                Embedding(values=cpu_results[j])
                for j in range(seqlens.tolist()[i])
            ]
            token_embeddings=TokenEmbedding(embeddings=embedding_tmp)
            embedding_result.append(token_embeddings)

        return embedding_result
    
    @tracer.start_as_current_span("predict")
    def predict(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Score]:
        if not isinstance(batch, FlashBatch):
            raise ValueError("only FlashBatch is supported")

        cu_seqlens = batch.cu_seqlens
        mask = None
        attn_mask = None
        max_input_lens = batch.max_s

        logits = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
    
        all_scores = logits.tolist()
        return [Score(values=scores) for scores in all_scores]
