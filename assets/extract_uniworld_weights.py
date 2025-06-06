import torch
from safetensors.torch import load_file, save_file
import json
import os
from collections import defaultdict
import argparse
import gc # For garbage collection

def process_and_save_weights_incrementally(uniworld_hf_model_dir, output_dir, target_component):
    """
    Processes shards incrementally for a specific component (flux or vlm).
    target_component: "flux" or "vlm"
    """
    print(f"Processing for component: {target_component}")
    
    index_path = os.path.join(uniworld_hf_model_dir, "model.safetensors.index.json")
    component_state_dict = {}
    flux_prefix = "denoise_tower.denoiser."

    if not os.path.exists(index_path):
        # Handle single large file case (less memory efficient for this function's design)
        print(f"model.safetensors.index.json not found. Attempting to process as single file for {target_component}.")
        single_model_path = None
        found_files = [f for f in os.listdir(uniworld_hf_model_dir) if f.endswith(".safetensors") and not f.endswith(".index.json")]
        if len(found_files) == 1:
            single_model_path = os.path.join(uniworld_hf_model_dir, found_files[0])
        else:
            print(f"Error: Expected a single .safetensors file or an index.json. Found: {found_files}")
            return False

        if not (single_model_path and os.path.exists(single_model_path)):
            print(f"Error: Single model file not found at {single_model_path}")
            return False

        print(f"Loading full model from single file for {target_component} processing: {single_model_path}")
        # This part is still memory intensive for single large files.
        # For very large single files, this script would need a different approach (e.g. memory mapping if supported by safetensors for selective loading)
        try:
            full_state_dict = load_file(single_model_path, device="cpu")
            for key, tensor in full_state_dict.items():
                try:
                    tensor_bf16 = tensor.to(dtype=torch.bfloat16)
                except Exception as e:
                    print(f"Warning: Could not convert tensor '{key}' to bfloat16. Skipping. Error: {e}")
                    continue

                is_flux_tensor = key.startswith(flux_prefix)
                if target_component == "flux" and is_flux_tensor:
                    new_key = key[len(flux_prefix):]
                    component_state_dict[new_key] = tensor_bf16
                elif target_component == "vlm" and not is_flux_tensor:
                    component_state_dict[key] = tensor_bf16
            del full_state_dict
            gc.collect()
        except Exception as e:
            print(f"Error processing single model file {single_model_path}: {e}")
            return False
    else: # Sharded model
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        if "weight_map" not in index_data:
            print(f"Error: 'weight_map' not found in {index_path}.")
            return False

        # Invert weight_map for easier shard processing
        shards_content = defaultdict(list)
        for key, filename in index_data["weight_map"].items():
            shards_content[filename].append(key)
        
        for shard_filename, keys_in_shard in shards_content.items():
            shard_path = os.path.join(uniworld_hf_model_dir, shard_filename)
            if not os.path.exists(shard_path):
                print(f"Warning: Shard file '{shard_path}' listed in index not found. Skipping.")
                continue
            
            print(f"  Processing shard for {target_component}: {shard_path}")
            try:
                shard_state_dict = load_file(shard_path, device="cpu")
                for key in keys_in_shard: # Iterate only over keys supposed to be in this shard
                    if key not in shard_state_dict:
                        print(f"Warning: Key '{key}' (from index) not found in shard '{shard_filename}'. Skipping.")
                        continue
                    
                    tensor = shard_state_dict[key]
                    try:
                        tensor_bf16 = tensor.to(dtype=torch.bfloat16)
                    except Exception as e:
                        print(f"Warning: Could not convert tensor '{key}' to bfloat16. Skipping. Error: {e}")
                        continue

                    is_flux_tensor = key.startswith(flux_prefix)
                    if target_component == "flux" and is_flux_tensor:
                        new_key = key[len(flux_prefix):]
                        component_state_dict[new_key] = tensor_bf16
                    elif target_component == "vlm" and not is_flux_tensor:
                        component_state_dict[key] = tensor_bf16
                
                del shard_state_dict # Explicitly delete to free memory
                del tensor # Explicitly delete to free memory
                if 'tensor_bf16' in locals(): del tensor_bf16 # Explicitly delete to free memory
                gc.collect() # Force garbage collection
            except Exception as e:
                print(f"Error processing shard {shard_path}: {e}")
                continue
    
    # Save the processed component
    if not component_state_dict:
        print(f"No weights found for component: {target_component}. Skipping save.")
        return True # Not a failure if component is empty

    if target_component == "flux":
        output_filename = "uniworld_finetuned_flux_transformer_bf16.safetensors"
    elif target_component == "vlm":
        output_filename = "uniworld_vlm_and_projectors_bf16.safetensors"
    else:
        print(f"Error: Unknown target component '{target_component}'")
        return False
        
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        print(f"Saving {target_component} weights ({len(component_state_dict)} tensors) to: {output_path}")
        save_file(component_state_dict, output_path)
        del component_state_dict # Free memory after saving
        gc.collect()
        print(f"Successfully saved {output_filename}")
        return True
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and separate UniWorld model weights incrementally.")
    parser.add_argument("model_dir", type=str, help="Path to the UniWorld Hugging Face model directory.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where separated model files will be saved.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Provided model directory does not exist: {args.model_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting Flux Transformer extraction ---")
    if not process_and_save_weights_incrementally(args.model_dir, args.output_dir, "flux"):
        print("Failed to process Flux Transformer weights.")
        return
    
    print("\n--- Starting VLM and Projectors extraction ---")
    if not process_and_save_weights_incrementally(args.model_dir, args.output_dir, "vlm"):
        print("Failed to process VLM and Projector weights.")
        return
        
    print("\nWeight separation and saving process complete.")

if __name__ == '__main__':
    main()
    # Example usage from command line:
    # python extract_uniworld_weights.py path/to/uniworld-hf-model path/to/output-separated-models
