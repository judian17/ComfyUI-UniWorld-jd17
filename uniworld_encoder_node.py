import torch
import os
from PIL import Image
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import traceback # For detailed import error logging
import gc # For model unloading

# --- UniWorld/Hugging Face Imports ---
# Attempting to import dependencies with detailed error logging.

print("### UniWorld Encoder Node: Attempting to import dependencies... ###")

AutoProcessor = None
BitsAndBytesConfig = None
UnivaQwen2p5VLForConditionalGeneration = None

try:
    from transformers import AutoProcessor as HFAutoProcessor
    AutoProcessor = HFAutoProcessor
    print("Successfully imported AutoProcessor from transformers.")
except ImportError as e:
    print("ERROR: Failed to import AutoProcessor from transformers.")
    print(traceback.format_exc())
except Exception as e:
    print(f"ERROR: An unexpected error occurred while importing AutoProcessor: {e}")
    print(traceback.format_exc())

try:
    from transformers import BitsAndBytesConfig as HFBitsAndBytesConfig
    BitsAndBytesConfig = HFBitsAndBytesConfig
    print("Successfully imported BitsAndBytesConfig from transformers.")
except ImportError as e:
    print("ERROR: Failed to import BitsAndBytesConfig from transformers. This is required for NF4 precision.")
    print(traceback.format_exc())
except Exception as e:
    print(f"ERROR: An unexpected error occurred while importing BitsAndBytesConfig: {e}")
    print(traceback.format_exc())

try:
    # Assuming 'univa' folder is placed alongside this node file.
    from .univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration as UniWorldVLM
    UnivaQwen2p5VLForConditionalGeneration = UniWorldVLM
    print("Successfully imported UnivaQwen2p5VLForConditionalGeneration from .univa package.")
except ImportError as e:
    print("ERROR: Failed to import UnivaQwen2p5VLForConditionalGeneration from .univa package.")
    print("Ensure the 'univa' folder is correctly placed in the same directory as this node file (e.g., custom_nodes/YourNodeFolder/univa) and is a valid package.")
    print(traceback.format_exc())
except Exception as e: 
    print(f"ERROR: An unexpected error occurred while trying to import from .univa package: {e}")
    print(traceback.format_exc())

# --- End UniWorld/Hugging Face Imports ---

# --- Helper functions and constants from qwen_vl_utils, adapted for ComfyUI node ---
import math
from PIL import Image

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if min(height, width) == 0: # Avoid division by zero
        return 0, 0
    if max(height, width) / min(height, width) > MAX_RATIO:
        print(f"Warning: Image aspect ratio ({max(height, width) / min(height, width)}) exceeds max ({MAX_RATIO}). Using default resize logic.")
        side = int(math.sqrt(min_pixels))
        return side, side

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")

def fetch_image_from_pil(pil_image, ele):
    image = to_rgb(pil_image)
    width, height = image.size
    min_pixels = ele.get("min_pixels", MIN_PIXELS)
    max_pixels = ele.get("max_pixels", MAX_PIXELS)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image

def extract_vision_info(conversations: list[dict]) -> list[dict]:
    vision_infos = []
    # Adapted for the simpler conversation structure in the node
    for message in conversations:
        if isinstance(message.get("content"), list):
            for ele in message["content"]:
                if ele.get("type") == "image":
                    vision_infos.append(ele)
    return vision_infos

def process_vision_info_for_node(conversations: list[dict]):
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info and isinstance(vision_info["image"], Image.Image):
            image_inputs.append(fetch_image_from_pil(vision_info["image"], vision_info))
    if len(image_inputs) == 0:
        image_inputs = None
    return image_inputs, None # No video support

class UniWorldEncoderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uniworld_model_dir": ("STRING", {"default": "path/to/your/uniworld_hf_model_directory", "multiline": False}),
                "model_precision": (["bf16", "nf4"], {"default": "bf16"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "no_joint_with_t5": ("BOOLEAN", {"default": False}),
                "unload_model_after_use": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "t5_conditioning": ("CONDITIONING",),
                "siglip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/UniWorld"

    _loaded_models = {} # Key: (model_dir_path, model_precision), Value: {'vlm': model, 'processor': processor, 'dtype': effective_compute_dtype, 'image_token_str': str, 'image_token_id': int}

    def __init__(self):
        self.device = model_management.text_encoder_device()

    def _load_uniworld_model_and_processor(self, model_dir_path, model_precision):
        if AutoProcessor is None or UnivaQwen2p5VLForConditionalGeneration is None:
            if model_precision == "nf4" and BitsAndBytesConfig is None:
                raise ImportError("BitsAndBytesConfig is required for NF4 precision but not loaded. Check transformers/bitsandbytes installation.")
            raise ImportError("Core UniWorld/Transformers libraries (AutoProcessor or UnivaQwen2p5VLForConditionalGeneration) not loaded. Cannot proceed.")
        
        cache_key = (model_dir_path, model_precision)
        if cache_key in self._loaded_models:
            cached_data = self._loaded_models[cache_key]
            return cached_data['vlm'], cached_data['processor'], cached_data['dtype'], cached_data['image_token_str'], cached_data['image_token_id']

        if not os.path.isdir(model_dir_path):
            raise FileNotFoundError(f"UniWorld model directory not found: {model_dir_path}")

        print(f"Loading UniWorld model and processor from: {model_dir_path} with precision: {model_precision}")
        
        load_kwargs = {"trust_remote_code": True}
        effective_compute_dtype = model_management.text_encoder_dtype(self.device) 

        if model_precision == "nf4":
            if BitsAndBytesConfig is None: 
                raise ImportError("BitsAndBytesConfig is required for NF4 precision but was not imported successfully.")
            
            if effective_compute_dtype not in [torch.bfloat16, torch.float16]:
                print(f"Warning: ComfyUI's default dtype {effective_compute_dtype} not ideal for NF4 compute_dtype. Defaulting to torch.bfloat16.")
                effective_compute_dtype = torch.bfloat16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=effective_compute_dtype, 
                bnb_4bit_use_double_quant=False
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = effective_compute_dtype 
            print(f"NF4 quantization enabled. Compute dtype: {effective_compute_dtype}")

        elif model_precision == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
            effective_compute_dtype = torch.bfloat16 
            print("BF16 precision enabled.")
        else: 
            load_kwargs["torch_dtype"] = effective_compute_dtype
            print(f"Using ComfyUI environment dtype: {effective_compute_dtype}")

        try:
            processor = AutoProcessor.from_pretrained(model_dir_path, trust_remote_code=True)
            vlm = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
                model_dir_path,
                **load_kwargs
            )
            
            if "quantization_config" not in load_kwargs or not load_kwargs.get("device_map"):
                 vlm.to(self.device)
            vlm.eval()

            image_token_id = vlm.config.image_token_id
            image_token_str = processor.tokenizer.decode([image_token_id])
            print(f"UniWorld Node: VLM config image_token_id: {image_token_id}")
            print(f"UniWorld Node: Decoded image_token_id to string: '{image_token_str}'")
            
            self._loaded_models[cache_key] = {
                'vlm': vlm, 
                'processor': processor, 
                'dtype': effective_compute_dtype, 
                'image_token_str': image_token_str,
                'image_token_id': image_token_id
            }
            print(f"UniWorld VLM and processor loaded successfully for {model_dir_path} ({model_precision}).")
            return vlm, processor, effective_compute_dtype, image_token_str, image_token_id
        except Exception as e:
            print(f"Error loading UniWorld model/processor from {model_dir_path} ({model_precision}): {e}")
            print(traceback.format_exc())
            raise e

    def _comfy_image_to_pil(self, image_tensor: torch.Tensor):
        if image_tensor is None:
            return []
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        pil_images = []
        for i in range(image_tensor.shape[0]):
            img_np = image_tensor[i].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np, 'RGB'))
        return pil_images

    def encode(self, uniworld_model_dir: str, model_precision: str, prompt: str, no_joint_with_t5: bool, unload_model_after_use: bool,
               image: torch.Tensor = None, t5_conditioning: list = None, siglip_vision_output: dict = None):
        
        vlm, processor, current_dtype, _, _ = self._load_uniworld_model_and_processor(uniworld_model_dir, model_precision)

        # 1. Construct the conversation in the format the model expects
        pil_images = self._comfy_image_to_pil(image)
        
        if not prompt and not pil_images:
            print("Warning: UniWorldEncoderNode received no prompt or image. Returning empty conditioning.")
            dummy_emb_dim = 4096; dummy_seq_len = 77 
            dummy_cond = torch.zeros((1, dummy_seq_len, dummy_emb_dim), device=self.device, dtype=current_dtype)
            dummy_pooled = torch.zeros((1, 4096), device=self.device, dtype=current_dtype)
            return ([[dummy_cond, {"pooled_output": dummy_pooled}]], )

        content = []
        if pil_images:
            for pil_img in pil_images:
                # The 'image' key holds the actual PIL object for local processing
                content.append({"type": "image", "image": pil_img})
        if prompt:
            content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]

        # 2. Process vision info using the logic from the original repository
        # This performs the crucial 'smart_resize' on the images
        image_inputs, _ = process_vision_info_for_node(conversation)
        if image_inputs:
            print(f"UniWorld Node: Processed {len(image_inputs)} image(s) with smart_resize. New size(s): {[img.size for img in image_inputs]}")

        # 3. Apply chat template to the structured conversation to get the final text
        chat_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        # Mimic original repo: drop the system prompt part
        if "<|im_start|>system" in chat_text:
            chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
        print(f"UniWorld Node: Text after apply_chat_template: '{chat_text}'")

        # 4. Use the main processor to tokenize text and prepare final inputs
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        print(f"UniWorld Node: Processor output input_ids: {inputs.input_ids}")
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            print(f"UniWorld Node: Processor output pixel_values shape: {inputs.pixel_values.shape}")

        # 5. Handle SigLIP embeddings (logic remains the same)
        siglip_embeds_for_vlm = None
        if siglip_vision_output is not None:
            if isinstance(siglip_vision_output, dict) and "last_hidden_state" in siglip_vision_output:
                siglip_embeds_for_vlm = siglip_vision_output["last_hidden_state"]
            elif torch.is_tensor(siglip_vision_output):
                siglip_embeds_for_vlm = siglip_vision_output
            
            if siglip_embeds_for_vlm is not None:
                siglip_embeds_for_vlm = siglip_embeds_for_vlm.to(self.device, dtype=current_dtype)
                print("UniWorld Node: Using siglip_hidden_states for VLM.")

        # 6. Forward pass through the VLM to get denoise_embeds
        with torch.no_grad():
            vlm_forward_kwargs = {
                "input_ids": inputs.input_ids.long(),
                "pixel_values": getattr(inputs, 'pixel_values', None),
                "attention_mask": inputs.attention_mask,
                "output_type": "denoise_embeds"
            }
            if hasattr(inputs, 'image_grid_thw'):
                vlm_forward_kwargs["image_grid_thw"] = getattr(inputs, 'image_grid_thw', None)
            if siglip_embeds_for_vlm is not None:
                vlm_forward_kwargs["siglip_hidden_states"] = siglip_embeds_for_vlm
            
            lvlm_embeds = vlm(**vlm_forward_kwargs)
            if lvlm_embeds.dtype != current_dtype:
                lvlm_embeds = lvlm_embeds.to(dtype=current_dtype)

        # 7. Combine with T5 conditioning (logic remains the same)
        t5_prompt_embeds = None
        pooled_prompt_embeds = None
        expected_pooled_dim = 4096

        if t5_conditioning is not None and not no_joint_with_t5:
            if len(t5_conditioning) > 0:
                t5_cond_item = t5_conditioning[0]
                t5_prompt_embeds = t5_cond_item[0].to(self.device, dtype=current_dtype)
                if isinstance(t5_cond_item[1], dict) and "pooled_output" in t5_cond_item[1]:
                    pooled_prompt_embeds = t5_cond_item[1].get("pooled_output").to(self.device, dtype=current_dtype)

        if pooled_prompt_embeds is None:
            batch_size = lvlm_embeds.shape[0]
            pooled_prompt_embeds = torch.zeros((batch_size, expected_pooled_dim), device=self.device, dtype=current_dtype)

        if t5_prompt_embeds is not None and not no_joint_with_t5:
            final_prompt_embeds = torch.cat([t5_prompt_embeds, lvlm_embeds], dim=1)
        else:
            final_prompt_embeds = lvlm_embeds

        # 8. Format and return the final conditioning
        output_conditioning = []
        for i in range(final_prompt_embeds.shape[0]):
            cond_slice = final_prompt_embeds[i:i+1]
            pooled_slice = pooled_prompt_embeds[i:i+1]
            output_conditioning.append([cond_slice, {"pooled_output": pooled_slice}])
            
        if unload_model_after_use:
            print(f"UniWorld Node: Unloading model from VRAM for {uniworld_model_dir} ({model_precision}).")
            cache_key = (uniworld_model_dir, model_precision)
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
            del vlm, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("UniWorld Node: Model unloaded.")

        return (output_conditioning,)


class UniWorld_T5_CLIP_Encoder:
    """
    A specialized text encoder for UniWorld that accurately reproduces the prompt encoding logic
    from the original repository's `denoiser_prompt_embedding_flux.py`.
    It bypasses ComfyUI's generic prompt parsing and uses a direct tokenization method.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_sequence_length": ("INT", {"default": 512, "min": 77, "max": 4096}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/UniWorld"

    def _encode_prompt_with_clip(self, text_encoder, tokenizer, prompt, device):
        # The ComfyUI tokenizer is a wrapper; the actual callable tokenizer is in the .tokenizer attribute
        text_inputs = tokenizer.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # ComfyUI's model wrappers expect a list of lists of ints, not a tensor.
        tokens_as_list = text_input_ids.tolist()
        # The encode method returns a tuple (last_hidden_state, pooled_output)
        _, pooled_output = text_encoder.encode(tokens_as_list)
        # Get dtype from the model's parameters as it doesn't have a .dtype attribute
        model_dtype = next(text_encoder.parameters()).dtype
        return pooled_output.to(dtype=model_dtype, device=device)

    def _encode_prompt_with_t5(self, text_encoder, tokenizer, prompt, max_sequence_length, device):
        # The ComfyUI tokenizer is a wrapper; the actual callable tokenizer is in the .tokenizer attribute
        text_inputs = tokenizer.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # ComfyUI's model wrappers expect a list of lists of ints, not a tensor.
        tokens_as_list = text_input_ids.tolist()
        # The T5 model wrapper returns a tuple where the first element is the embeds
        prompt_embeds = text_encoder(tokens_as_list)[0]
        # Get dtype from the model's parameters as it doesn't have a .dtype attribute
        model_dtype = next(text_encoder.parameters()).dtype
        return prompt_embeds.to(dtype=model_dtype, device=device)

    def encode(self, flux_clip, prompt, max_sequence_length):
        # This node is designed to work with the output of a DualCLIPLoader set to 'flux' type.
        # The structure is confirmed from comfy/text_encoders/flux.py
        if not (hasattr(flux_clip.tokenizer, 'clip_l') and hasattr(flux_clip.tokenizer, 't5xxl')):
            raise TypeError("The provided CLIP object is not a valid FLUX dual CLIP. It must have 'clip_l' and 't5xxl' tokenizers.")
        
        if not (hasattr(flux_clip.cond_stage_model, 'clip_l') and hasattr(flux_clip.cond_stage_model, 't5xxl')):
            raise TypeError("The provided CLIP object is not a valid FLUX dual CLIP. It must have 'clip_l' and 't5xxl' encoders.")

        # Correctly access the individual tokenizers and encoders
        clip_tokenizer = flux_clip.tokenizer.clip_l
        t5_tokenizer = flux_clip.tokenizer.t5xxl
        
        clip_encoder = flux_clip.cond_stage_model.clip_l
        t5_encoder = flux_clip.cond_stage_model.t5xxl
        
        # A robust way to get the device from the model's parameters
        device = next(t5_encoder.parameters()).device
        
        # Handle empty prompt, which is a valid use case for image-only editing
        if not prompt:
            prompt = ""
        
        print(f"UniWorld T5/CLIP Encoder: Encoding prompt '{prompt}' with max_sequence_length {max_sequence_length}")

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder=clip_encoder,
            tokenizer=clip_tokenizer,
            prompt=[prompt],
            device=device,
        )

        prompt_embeds = self._encode_prompt_with_t5(
            text_encoder=t5_encoder,
            tokenizer=t5_tokenizer,
            prompt=[prompt],
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        return ([[prompt_embeds, {"pooled_output": pooled_prompt_embeds}]], )


NODE_CLASS_MAPPINGS = {
    "UniWorldEncoderNode": UniWorldEncoderNode,
    "UniWorld_T5_CLIP_Encoder": UniWorld_T5_CLIP_Encoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniWorldEncoderNode": "UniWorld Encoder (Qwen2.5VL)",
    "UniWorld_T5_CLIP_Encoder": "UniWorld T5/CLIP Encoder",
}
