"""
⭐SN ComfyUI AceStep 1.5 LoRA Trainer — Version 1.0 (Beta)

Custom nodes for training audio LoRAs on the AceStep 1.5 DiT model.

Nodes:
  - ⭐SN AceStep 1.5 Dataset Builder: Scan audio, auto-label with LLM, create .txt metadata
  - ⭐SN AceStep 1.5 Dataset Preprocessor: Encode audio+text to .pt tensor files
  - ⭐SN AceStep 1.5 LoRA Trainer: Flow matching LoRA training on DiT decoder
  - ⭐SN AceStep Loss Graph: Real-time training loss visualization (observer)
"""

from .nodes.dataset_builder import AceStep15DatasetBuilder
from .nodes.preprocessor import AceStep15DatasetPreprocessor
from .nodes.trainer import AceStep15LoRATrainer
from .nodes.loss_graph import AceStepLossGraph

NODE_CLASS_MAPPINGS = {
    "AceStep15DatasetBuilder": AceStep15DatasetBuilder,
    "AceStep15DatasetPreprocessor": AceStep15DatasetPreprocessor,
    "AceStep15LoRATrainer": AceStep15LoRATrainer,
    "AceStep_Loss_Graph": AceStepLossGraph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStep15DatasetBuilder": "⭐SN AceStep 1.5 Dataset Builder (Beta)",
    "AceStep15DatasetPreprocessor": "⭐SN AceStep 1.5 Dataset Preprocessor (Beta)",
    "AceStep15LoRATrainer": "⭐SN AceStep 1.5 LoRA Trainer (Beta)",
    "AceStep_Loss_Graph": "⭐SN AceStep Loss Graph (Beta)",
}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
