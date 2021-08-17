from collections import OrderedDict
from packaging import version

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter

from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)


CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig)])
MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM")])
TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)),
    ]
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LayoutLMv2Config, LayoutLMv2ForTokenClassification), (LayoutXLMConfig, LayoutXLMForTokenClassification)]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LayoutLMv2Config, LayoutLMv2ForRelationExtraction), (LayoutXLMConfig, LayoutXLMForRelationExtraction)]
)


if version.parse(transformers.__version__) < version.parse("4.9.0"):
    from transformers.models.auto.modeling_auto import auto_class_factory
    AutoModelForTokenClassification = auto_class_factory(
        "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
    )

    AutoModelForRelationExtraction = auto_class_factory(
        "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
    )
else:
    from transformers.models.auto.modeling_auto import auto_class_update, _BaseAutoModelClass
    class AutoModelForTokenClassification(_BaseAutoModelClass):
        _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


    class AutoModelForRelationExtraction(_BaseAutoModelClass):
        _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    AutoModelForRelationExtraction = auto_class_update(AutoModelForRelationExtraction, head_doc="relation extraction")
