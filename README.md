

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


## 📊 Quantatitive Performance

Performance of various open-source and closed-source models on GDI-Bench at different levels of reasoning complexity. The GDI-Model is fine-tuned based on the InternVL3-8B model.

![multi_subplot_analysis_alpha](.\image\multi_subplot_analysis_alpha.png)

The performance of different open-source and closed-source large models and different training methods on GDI-Bench.

![GDI-bench](.\image\GDI-bench.jpg)

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

