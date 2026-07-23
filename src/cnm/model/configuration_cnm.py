"""Configuration for CNM-BERT models.

This module defines CNMConfig, which extends HuggingFace's BertConfig
with additional parameters for the structural embedding components.
"""

from typing import Optional

from transformers import BertConfig
from transformers.configuration_utils import PretrainedConfig


class CNMConfig(BertConfig):
    """Configuration for CNM-BERT model.

    Extends BertConfig with parameters for:
    - Tree-MLP structural encoder
    - Component/operator vocabularies
    - Fusion layer settings
    - Auxiliary loss configuration
    """

    model_type = "cnm-bert"

    def __init__(
        self,
        # Standard BERT parameters
        vocab_size: int = 21128,  # bert-base-chinese vocab size
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        # CNM-specific parameters
        struct_dim: int = 256,
        tree_hidden_dim: int = 512,
        max_tree_depth: int = 6,
        component_vocab_size: int = 5000,
        operator_vocab_size: int = 16,
        aux_loss_weight: float = 0.1,
        fusion_type: str = "concat_project",
        struct_dropout: float = 0.1,
        use_struct_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            **kwargs,
        )

        # CNM-specific parameters
        self.struct_dim = struct_dim
        self.tree_hidden_dim = tree_hidden_dim
        self.max_tree_depth = max_tree_depth
        self.component_vocab_size = component_vocab_size
        self.operator_vocab_size = operator_vocab_size
        self.aux_loss_weight = aux_loss_weight
        self.fusion_type = fusion_type
        self.struct_dropout = struct_dropout
        self.use_struct_embeddings = use_struct_embeddings

    @classmethod
    def from_bert_config(
        cls,
        bert_config: BertConfig,
        struct_dim: int = 256,
        tree_hidden_dim: int = 512,
        max_tree_depth: int = 6,
        component_vocab_size: int = 5000,
        operator_vocab_size: int = 16,
        aux_loss_weight: float = 0.1,
        **kwargs,
    ) -> "CNMConfig":
        """Create CNMConfig from an existing BertConfig."""
        return cls(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            hidden_act=bert_config.hidden_act,
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer_range=bert_config.initializer_range,
            layer_norm_eps=bert_config.layer_norm_eps,
            pad_token_id=bert_config.pad_token_id,
            struct_dim=struct_dim,
            tree_hidden_dim=tree_hidden_dim,
            max_tree_depth=max_tree_depth,
            component_vocab_size=component_vocab_size,
            operator_vocab_size=operator_vocab_size,
            aux_loss_weight=aux_loss_weight,
            **kwargs,
        )

    @classmethod
    def base_chinese(cls, **kwargs) -> "CNMConfig":
        """Get config for BERT-base-chinese scale."""
        return cls(
            vocab_size=21128,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            **kwargs,
        )

    @classmethod
    def large_chinese(cls, **kwargs) -> "CNMConfig":
        """Get config for BERT-large scale."""
        return cls(
            vocab_size=21128,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            **kwargs,
        )
