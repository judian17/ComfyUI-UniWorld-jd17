import torch
from safetensors.torch import load_file, save_file
import json
import os
from collections import defaultdict
import argparse
import gc # For garbage collection

def extract_nf4_weights_incrementally(uniworld_hf_model_dir, output_dir, target_component):
    """
    Processes sharded NF4 quantized UniWorld model weights incrementally 
    for a specific component (flux or vlm).
    target_component: "flux" or "vlm"
    Weights are preserved in their original (quantized) format.
    """
    print(f"Processing NF4 weights for component: {target_component} from: {uniworld_hf_model_dir}")
    
    index_path = os.path.join(uniworld_hf_model_dir, "model.safetensors.index.json")
    component_state_dict = {}
    flux_prefix = "denoise_tower.denoiser."

    # Known suffixes for bitsandbytes quantization states. Order might matter if one is a substring of another.
    # More specific (longer) suffixes should come first.
    known_quant_suffixes = [
        ".quant_state.bitsandbytes__nf4", 
        ".quant_state.bitsandbytes__fp4", # If other quant types might be present
        ".SCB", # Older bitsandbytes state format for 4bit
        ".absmax", 
        ".quant_map", 
    ]

    if not os.path.exists(index_path):
        print(f"model.safetensors.index.json not found. Attempting to process as single file for {target_component}.")
        single_model_path = None
        found_files = [f for f in os.listdir(uniworld_hf_model_dir) if f.endswith(".safetensors") and not f.endswith(".index.json")]
        if len(found_files) == 1:
            single_model_path = os.path.join(uniworld_hf_model_dir, found_files[0])
        else:
            print(f"Error: Expected a single .safetensors file or an index.json for NF4 model. Found: {found_files}")
            return False

        if not (single_model_path and os.path.exists(single_model_path)):
            print(f"Error: Single NF4 model file not found at {single_model_path}")
            return False

        print(f"Loading full NF4 model from single file for {target_component} processing: {single_model_path}")
        try:
            full_state_dict = load_file(single_model_path, device="cpu")
            for key, tensor in full_state_dict.items():
                base_key_for_prefix_check = key
                quant_suffix = ""
                for suffix in known_quant_suffixes:
                    if key.endswith(suffix):
                        base_key_for_prefix_check = key[:-len(suffix)]
                        quant_suffix = suffix
                        break
                
                is_flux_tensor_base = base_key_for_prefix_check.startswith(flux_prefix)

                if target_component == "flux" and is_flux_tensor_base:
                    new_base_key = base_key_for_prefix_check[len(flux_prefix):]
                    final_key_for_flux_dict = new_base_key + quant_suffix
                    component_state_dict[final_key_for_flux_dict] = tensor
                elif target_component == "vlm" and not is_flux_tensor_base:
                    component_state_dict[key] = tensor
            del full_state_dict
            gc.collect()
        except Exception as e:
            print(f"Error processing single NF4 model file {single_model_path}: {e}")
            return False
    else: # Sharded model
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        if "weight_map" not in index_data:
            print(f"Error: 'weight_map' not found in {index_path}.")
            return False

        shards_content = defaultdict(list)
        for key, filename in index_data["weight_map"].items():
            shards_content[filename].append(key)
        
        for shard_filename, keys_in_shard_from_index in shards_content.items():
            shard_path = os.path.join(uniworld_hf_model_dir, shard_filename)
            if not os.path.exists(shard_path):
                print(f"Warning: Shard file '{shard_path}' listed in index not found. Skipping.")
                continue
            
            print(f"  Processing NF4 shard for {target_component}: {shard_path}")
            try:
                shard_state_dict = load_file(shard_path, device="cpu")
                # Iterate over keys actually in the shard, but use the original full key from index for prefix checking
                for original_key in keys_in_shard_from_index:
                    if original_key not in shard_state_dict:
                        # This case should ideally be handled by the outer loop over index_data["weight_map"].items()
                        # if we load all shards first, but for incremental, we check here.
                        # However, the current loop structure is over shard_filename, then keys_in_shard_from_index.
                        # So, original_key is the one from index.
                        print(f"Warning: Key '{original_key}' (from index) not found in loaded shard '{shard_filename}'. Skipping.")
                        continue
                    
                    tensor = shard_state_dict[original_key]
                    
                    base_key_for_prefix_check = original_key
                    quant_suffix = ""
                    for suffix in known_quant_suffixes:
                        if original_key.endswith(suffix):
                            base_key_for_prefix_check = original_key[:-len(suffix)]
                            quant_suffix = suffix
                            break
                    
                    is_flux_tensor_base = base_key_for_prefix_check.startswith(flux_prefix)

                    if target_component == "flux" and is_flux_tensor_base:
                        new_base_key = base_key_for_prefix_check[len(flux_prefix):]
                        final_key_for_flux_dict = new_base_key + quant_suffix
                        component_state_dict[final_key_for_flux_dict] = tensor
                    elif target_component == "vlm" and not is_flux_tensor_base:
                        component_state_dict[original_key] = tensor 
                
                del shard_state_dict
                if 'tensor' in locals(): del tensor 
                gc.collect()
            except Exception as e:
                print(f"Error processing NF4 shard {shard_path}: {e}")
                continue
    
    if not component_state_dict:
        print(f"No weights found for NF4 component: {target_component}. Skipping save.")
        return True

    if target_component == "flux":
        output_filename = "uniworld_finetuned_flux_transformer_nf4.safetensors"
    elif target_component == "vlm":
        output_filename = "uniworld_vlm_and_projectors_nf4.safetensors"
    else:
        print(f"Error: Unknown target component '{target_component}'")
        return False
        
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        print(f"Saving NF4 {target_component} weights ({len(component_state_dict)} tensors) to: {output_path}")
        save_file(component_state_dict, output_path)
        del component_state_dict
        gc.collect()
        print(f"Successfully saved {output_filename}")
        return True
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and separate UniWorld NF4 quantized model weights incrementally.")
    parser.add_argument("model_dir", type=str, help="Path to the UniWorld NF4 quantized Hugging Face model directory.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where separated NF4 model files will be saved.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Provided model directory does not exist: {args.model_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting NF4 Flux Transformer extraction ---")
    if not extract_nf4_weights_incrementally(args.model_dir, args.output_dir, "flux"):
        print("Failed to process NF4 Flux Transformer weights.")
        return
    
    print("\n--- Starting NF4 VLM and Projectors extraction ---")
    if not extract_nf4_weights_incrementally(args.model_dir, args.output_dir, "vlm"):
        print("Failed to process NF4 VLM and Projector weights.")
        return
        
    print("\nNF4 Weight separation and saving process complete.")

if __name__ == '__main__':
    main()
    # Example usage from command line:
    # python extract_uniworld_nf4_weights.py path/to/uniworld-nf4-hf-model path/to/output-nf4-separated-models
