

<div align="center">
<h1> GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling </h1>


</div>

<div align="center">
<p align="center">
💜 <a href="https://knowledgexlab.github.io/gdibench.github.io/"><b>HomePage</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/GDIBench">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2505.00063">Paper</a>&nbsp&nbsp
</p>


</div>


## 📰 News

The rapid advancement of multimodal large language models (MLLMs) has profoundly impacted the document domain, creating a wide array of application scenarios. This progress highlights the need for a comprehensive benchmark to evaluate these models' capabilities across various document-specific tasks. However, existing benchmarks often fail to locate specific model weaknesses or guide systematic improvements. To bridge this gap, we introduce a General Document Intelligence Benchmark (GDI-Bench), featuring 2.3k images across 9 key scenarios and 19 document-specific tasks. By decoupling visual complexity and reasoning complexity, the GDI-Bench structures graded tasks that allow performance assessment by difficulty, aiding in model weakness identification and optimization guidance. We evaluate various open-source and closed-source models on GDI-Bench, conducting decoupled analyses in the visual and reasoning domains, revealing their strengths and weaknesses. To address the diverse tasks and domains in the GDI-Bench, we propose a GDI-Model that mitigates catastrophic forgetting during the supervised fine-tuning (SFT) process through an intelligence-preserving training strategy, thereby reinforcing the inherent weaknesses of the base model. Our model achieves state-of-the-art performance on previous benchmarks and the GDI-Bench. Both our benchmark and models are or will be open-sourced on https://huggingface.co/GDIBench.



## 🎯 GDI-Bench

To assist Multimodal Large Language Models (MLLMs) in locating their weaknesses within the document domain and to further guide model optimization, we first constructed a benchmark. GDI-Bench decouples task complexity into two distinct dimensions—visual complexity and reasoning complexity—and establishes a graded mechanism.

![image-20250528-155238](https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/20250528-155238.jpg)

GDI-Bench decouples document understanding complexity into visual complexity (V0-V2) and reasoning complexity (R0-R2) dimensions. The visual complexity dimension is operationalized through a hierarchical categorization of document images into three levels: V0 (plain text), V1 (formal representations), and V2 (explanatory representations). In parallel, the dimension of reasoning complexity is characterized by three categories: R0 (Full Page Structured Extraction), R1 (Information Extraction), and R2 (Reasoning).



## 🚀 Layer-wise Adaptive Freezing-Tuning

### 1. Environment Setup

The environment configuration aligns with the fine-tuning setup of **InternVL3**.
Ensure all dependencies match the original fine-tuning environment.

### 2. Freezing Strategy Generation

Generate the parameter freezing strategy using the following command, where `k` is the number of global unfrozen parameters:

```
python param_mask_layer.py \
  --ref_model_path /your/path/to/base/model \
  --expert_model_path /your/path/to/expert/model \
  --k 80000000
```

### 3. Fine-tuning Configuration

1. **Replace Training Script**
   Substitute `InternVL/internvl_chat/train/internvl_chat_finetune.py` with our open-sourced [`internvl_chat_finetune.py`](https://github.com/KnowledgeXLab/GDI-Bench/blob/main/internvl_chat_finetune.py).

2. **Inject Freezing Strategy**
   Add the `--load_freeze` flag to your training command, pointing to the generated strategy JSON:

   ```
   --load_freeze '/your/path/to/freezing/strategy.json'
   ```

   Example full command:

   ```
   python internvl_chat_finetune.py \
     ... # other flags \
     --load_freeze '/path/to/strategy.json'
   ```



## 🐳 Docker Image for GDI Inference

### 1. Run Inference with Docker

```
# Step 1: Pull the pre-built image
docker pull yxizhong/gdi:latest

# Step 2: Run inference (map custom paths + GPU enabled)
docker run --rm --gpus device=0 \
  -v /path/to/your/models:/app/my_model \
  -v /path/to/your/images:/app/my_images \
  yxizhong/gdi:latest \
  python GDI_inference.py \
    --model_dir /app/my_model \
    --image_dir /app/my_images \
    --prompt "your_custom_prompt_here"
```

### 2. Prompt Examples

| **Task Type**                  | Image Example                                                | **Prompt Example**                                           |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Exam Analysis**              | <img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/Exam%20Analysis.png" width="200"> | `"Generate JSON from Problem-solving Questions question 13 with keys: 题号 (number), 题目 (question), 答案 (answer)"` |
| **Handwritten Answer Parsing** | <img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/Handwritten%20Answer%20Parsing.png" width="200"> | `"Extract question 15 from Problem-solving Questions to markdown format"` |
| **Author Metadata Extraction** | <img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/Author%20Metadata%20Extraction.png" width="200"> | `"Convert author metadata to JSON: {"Author Information": [{"Name": "...", "Affiliation": "...", "Role": "First Author/Corresponding Author/Author"}]}"` |
| **Reference Extraction**       | <img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/bibtex.jpg" width="200"> | `"Extract all references on this page and output in BibTeX format"` |
| **Table Extraction**           | <img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/exm1_table_inf.png" width="200"> | `"Extract the table in this image, convert to LaTeX code"`   |





## 📊 Quantatitive Performance

Performance of various open-source and closed-source models on GDI-Bench at different levels of reasoning complexity. The GDI-Model is fine-tuned based on the InternVL3-8B model.

![multi_subplot_analysis_alpha](https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/multi_subplot_analysis_alpha.png)

The performance of different open-source and closed-source large models and different training methods on GDI-Bench.

![GDI-bench](https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/GDI-bench.jpg)



## Citation

If you find the provided dataset or model useful for your research, consider citing them as:

```
@article{gdibench,
  title={GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling},
  author={Li, Siqi and Shen, Yufan and Chen, Xiangnan and Chen, Jiayi and Ju, Hengwei and Duan, Haodong and Mao, Song and Zhou, Hongbin and Zhang, Bo and Fu, Bin and others},
  journal={arXiv preprint arXiv:2505.00063},
  year={2025}
}
```

