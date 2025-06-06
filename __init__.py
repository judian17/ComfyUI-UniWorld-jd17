# ComfyUI-UniWorldEncoder/__init__.py

# Import mappings from all node files
from .uniworld_encoder_node import NODE_CLASS_MAPPINGS as encoder_mappings, NODE_DISPLAY_NAME_MAPPINGS as encoder_display_mappings
from .siglip2 import NODE_CLASS_MAPPINGS as siglip_mappings, NODE_DISPLAY_NAME_MAPPINGS as siglip_display_mappings
from .uniworld_sampler import NODE_CLASS_MAPPINGS as sampler_mappings, NODE_DISPLAY_NAME_MAPPINGS as sampler_display_mappings

# Merge the mappings from all nodes
NODE_CLASS_MAPPINGS = {**encoder_mappings, **siglip_mappings, **sampler_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**encoder_display_mappings, **siglip_display_mappings, **sampler_display_mappings}

# Expose the final merged mappings for ComfyUI to discover
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("### Loading: ComfyUI-UniWorldEncoder Nodes (Encoder, Siglip, Sampler) ###")
