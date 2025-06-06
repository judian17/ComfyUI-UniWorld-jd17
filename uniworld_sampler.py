import torch
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Helper function copied directly from the original FLUX/UniWorld pipeline
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """Calculates the mu parameter for the FlowMatchEulerDiscreteScheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

class UniWorldScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise, latent):
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        
        total_steps = int(steps / denoise) if denoise < 1.0 else steps
        
        latents_tensor = latent["samples"]
        # The original code uses the sequence length of the *packed* latents
        image_seq_len = latents_tensor.shape[1] if latents_tensor.numel() > 0 else 256
        
        # Get the base model config to initialize our own scheduler instance, ensuring compatibility.
        # The config is on the core model object (model.model), not the model_sampling wrapper.
        model_config_object = model.model.model_config
        
        # The model_config is an object, not a dict. Access attributes directly and safely.
        scheduler_config = model_config_object.scheduler_config if hasattr(model_config_object, "scheduler_config") else {}
        
        try:
            scheduler_instance = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        except Exception as e:
            print(f"UniWorld Scheduler: Could not instantiate FlowMatchEulerDiscreteScheduler from model config. Using default. Error: {e}")
            scheduler_instance = FlowMatchEulerDiscreteScheduler()


        # This is the core logic from the original pipeline
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=scheduler_instance.config.get("base_image_seq_len", 256),
            max_seq_len=scheduler_instance.config.get("max_image_seq_len", 4096),
            base_shift=scheduler_instance.config.get("base_shift", 0.5),
            max_shift=scheduler_instance.config.get("max_shift", 1.16),
        )
        print(f"UniWorld Scheduler: Calculated mu={mu} for latent sequence length {image_seq_len}")

        # Set timesteps with the custom mu on our own scheduler instance
        scheduler_instance.set_timesteps(total_steps, mu=mu, device="cpu")
        
        sigmas = scheduler_instance.sigmas

        if denoise < 1.0:
            sigmas = sigmas[-(steps + 1):]

        return (sigmas,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "UniWorldScheduler": UniWorldScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniWorldScheduler": "UniWorld Scheduler (for FLUX)",
}
