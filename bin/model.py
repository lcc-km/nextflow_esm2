#!/usr/bin/env python
import os
import torch
import torch.nn as nn
from transformers import EsmModel, EsmPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

os.environ["TRANSFORMERS_SUPPRESS_TORCH_LOAD_WARNING"] = "1"


class EsmForMeanPoolingSequenceClassification(EsmPreTrainedModel):
    """
    ESM-2 sequence classification/regression model with mean pooling.
    Designed for predicting protein mutation effects (e.g., DMS data).
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esm = EsmModel(config, add_pooling_layer=False)
        
        # Simple MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Extract features from ESM backbone
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
        
        # Mask-aware Mean Pooling (excluding padding tokens)
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        # sum_embeddings = torch.sum(sequence_output * input_mask_expanded, dim=1)
        
        # sum_mask = input_mask_expanded.sum(dim=1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        
        # mean_pooled = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_size)

        # # Classification/regression head
        # logits = self.classifier(mean_pooled)

        cls_pooled = sequence_output[:, 0, :]

        logits = self.classifier(cls_pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)
        
        # HuggingFace-compatible output structure
        class CustomOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return CustomOutput(logits=logits)


def build_esm2_model(
    model_name: str = None,
    num_labels: int = 1,
    freeze_encoder_layers: int = 0,
    cache_dir: str = None
):
    """
    Build and configure the ESM-2 model with Mean Pooling.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", None))

    model = EsmForMeanPoolingSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )

    # Freeze lower encoder layers if requested
    if freeze_encoder_layers > 0:
        layers_to_freeze = [f"layer.{i}." for i in range(freeze_encoder_layers)]
        for name, param in model.esm.named_parameters():
            if any(layer_identifier in name for layer_identifier in layers_to_freeze):
                param.requires_grad = False

    return model