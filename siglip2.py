import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class UniWorldSiglipEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/UniWorld"

    def _comfy_image_to_pil(self, image_tensor: torch.Tensor):
        # Converts ComfyUI's [B, H, W, C] float tensor to a list of PIL Images
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        pil_images = []
        for i in range(image_tensor.shape[0]):
            img_np = image_tensor[i].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np, 'RGB'))
        return pil_images

    def encode(self, clip_vision, image):
        """
        This encoder performs the EXACT preprocessing required by the SigLIP model
        used in the UniWorld project, based on its preprocessor_config.json.
        It bypasses ComfyUI's generic `encode_image` to ensure correctness.
        """
        # 1. Define the exact transforms from the config file
        # size: 512x512, resample: 2 (BICUBIC), mean/std: 0.5
        # ComfyUI's IMAGE tensor is already in the [0, 1] range.
        transforms = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(), # Converts PIL image in range [0, 255] to [0, 1] tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 2. Convert ComfyUI tensor to PIL and apply transforms
        # We first convert to PIL because torchvision transforms are most consistent with PIL input
        pil_images = self._comfy_image_to_pil(image)
        
        # Process images in a batch
        processed_tensors = [transforms(p) for p in pil_images]
        pixel_values = torch.stack(processed_tensors).to(clip_vision.load_device)

        # 3. Manually call the vision model's forward pass
        # We need to access the underlying model from the ComfyUI wrapper
        vision_model = clip_vision.model
        
        # Ensure model is on the correct device and in eval mode
        vision_model.to(clip_vision.load_device).eval()

        # Get the model's output. The ComfyUI wrapper returns a tuple.
        outputs_tuple = vision_model(pixel_values)
        
        # Unpack the tuple: (last_hidden_state, pooled_output)
        last_hidden_state = outputs_tuple[0]
        pooled_output = outputs_tuple[1]

        # 4. Return in the format ComfyUI expects
        vision_output = {
            "last_hidden_state": last_hidden_state,
            "image_embeds": pooled_output
        }
        
        print("UniWorld Siglip Encoder: Successfully encoded image with custom, project-specific preprocessing.")
        return (vision_output,)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "UniWorldSiglipEncoder": UniWorldSiglipEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniWorldSiglipEncoder": "UniWorld Siglip Encoder"
}
