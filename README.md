# ComfyUI-UniWorld-jd17
Unofficial ComfyUI implementation of the image editing part of [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1).

## [中文版说明](README-zh.md)

## Introduction

The workflow and results are shown in the figure below.

![Example](assets/example.png)

The image editing feature is probably implemented :)

**Note: Please ensure that `transformers==4.50.0` in your ComfyUI environment (I tested that `4.52.3` causes errors, and I'm not sure about lower versions).**

### Weight Extraction
You can use [extract_uniworld_weights.py](assets/extract_uniworld_weights.py) to extract the Flux model weights and the remaining weights from the [UniWorld](https://huggingface.co/LanguageBind/UniWorld-V1) model. Example:

```python
python "path/to/extract_uniworld_weights.py" "path/to/uniworld-32" "path/to/uniworld-16-extracted"
```

The extracted Flux weights can be loaded directly using ComfyUI's `UnetLoader`. The remaining weights are used for text encoding in the `UniWorld Encoder` node (this requires modifying the `config.json` file accordingly).

### NF4 Quantized Model Support
Special thanks to [wikeeyang](https://huggingface.co/wikeeyang) for providing the [UniWorld-V1-NF4](https://huggingface.co/wikeeyang/UniWorld-V1-NF4) model.

You can use [extract_uniworld_nf4_weights.py](assets/extract_uniworld_nf4_weights.py) to extract the NF4 quantized Qwen2.5VL weights. Example:

```python
python "path/to/extract_uniworld_nf4_weights.py" "path/to/uniworld-v1-nf4" "path/to/uniworld-v1-nf4-extracted"
```

These weights are also used for text encoding in the `UniWorld Encoder` node (this requires modifying the `config.json` file accordingly).

## Downloads
Here are the download links for some pre-processed model weights:

**Flux weights from UniWorld-V1:**

*   [UniWorld-v1-flux-comfy-fp8](https://www.modelscope.cn/models/ahaha2024/UniWorld-v1-flux-comfy-fp8/)
*   [UniWorld-v1-flux-bf16](https://www.modelscope.cn/models/ahaha2024/UniWorld-v1-flux-bf16)

**Qwen2.5VL weights from UniWorld-V1:**

*   [UniWorld-v1-qwen2.5vl-nf4](https://www.modelscope.cn/models/ahaha2024/UniWorld-v1-qwen2.5vl-nf4/)
*   [UniWorld-v1-qwen2.5vl-bf16](https://www.modelscope.cn/models/ahaha2024/UniWorld-v1-qwen2.5vl-bf16)

Alternatively, you can download the official models and process them using the provided scripts. [config.json for bf16](assets/bf16/config.json) is the configuration file for the bf16 version of the UniWorld Encoder model, and [config.json for nf4](assets/nf4/config.json) is for the nf4 version.

## Additional Notes

I am not an expert in coding. The nodes in this project were primarily developed with the help of Gemini. Based on the output results, it seems to be working as intended. If you find any areas for improvement, please feel free to point them out!
