

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

# Step 2: Download pre-trained model from HuggingFace
git lfs install
git clone https://huggingface.co/GDIBench/GDI-Model/ /app/my_model

# Step 3: Start interactive Docker container
docker run -it --rm --gpus device=0 \
  -v /path/to/my_model:/app/my_model \
  -v /path/to/my_images:/app/my_images \
  yxizhong/gdi:latest \
  /bin/bash
  
# Step 4: Activate the Python environment
conda activate gdi

# Step 5: Execute inference command
python GDI_inference.py \
  --model_dir /app/my_model \
  --image_dir /app/my_images \
  --prompt "your custom prompt here"
```

### 2. Prompt Examples

| Image Example                                                | Task Type             | Prompt Example                                           | Response                                                     |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Handwritten Answer Paper**<img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/Exam%20Analysis.png" width="200"><img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/Handwritten%20Answer%20Parsing.png" width="200"> | Exam Analysis              | `Generate JSON from Multiple-Choices Questions question 9 with keys: 题号 (number), 题目 (question), 答案 (answer)` | `{    "number": "9",    "question": "已知数列\(\{ a_{n}\}\)满足\(a_{n}=3n - 10\)，则以下说法正确的是（  ）\nA. 数列\(\{ a_{n}\}\)为单调递减数列\nB. \(a_{2}a_{5}>0\)\nC. \(a_{3}+a_{7}=2a_{5}\)\nD. \(\vert a_{n}\vert\)不单调",    "answer": "D" }` |
|                                                              | Handwritten Answer Parsing | `"Extract question 15 from Problem-solving Questions to markdown format"` | ` 15.（15分）已知$F_1,F_2$分别为椭圆$C:\frac{x^2}{a^2}+\frac{y^2}{b^2}=1(a>b>0)$的左、右焦点，且椭圆经过点$(2,0)$和点$(1,e)$，其中$e$为椭圆的离心率.   (1)求椭圆$C$的方程；   (2)若倾斜角为$30^\circ$的直线$l$经过点$F_2$，且与$C$交于$M,N$两点（$M$点在$N$点的上方），求$\frac{\|MF_2\|}{\|NF_2\|}$的值.   解：(1) 由题$(2,0)$和点$(1,e)$   $\therefore \frac{2^2}{a^2}+\frac{0^2}{b^2}=1$，$\frac{1^2}{a^2}+\frac{e^2}{b^2}=1$   又$e=\frac{c}{a}$，$a^2=b^2+c^2$，由$\frac{4}{a^2}=1$得$a = 2$   则$\frac{1}{4}+\frac{(\frac{c}{2})^2}{b^2}=1$，且$b^2 + c^2 = 4$，解得$b^2 = 3$，$c = 1$   $\therefore$ 椭圆$C$的方程为$\frac{x^2}{4}+\frac{y^2}{3}=1$   (2) '` |
| **Printed Exam Paper**<img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/exam2.png" width="200"> | Exam Analysis              | `Extract question 8 of the multiple-choice questions and organize them into json format, including keys: 题号 (number), 题目 (question), 答案 (answer), 解析(analysis)` | `{    "number": "8",    "question": "（6分）一束由红、蓝两单色光组成的复色光从空气射向半球形玻璃的上表面入射入射角为θ，穿过玻璃后折射到光屏上。已知该半球形玻璃对红、蓝光的折射率分别为n₁和n₂，且n₁<n₂。则在光穿过玻璃的过程中（　　）\nA. 若改变θ，当红、蓝光都能射出玻璃时，它们的折射角仍都为θ\nB. 若θ增大，则红、蓝光可能都不会从下表面射出\nC. 若θ等于0，红、蓝光传播时间分别为t₁和t₂，则t₁>t₂\nD. 若θ等于0，红光的上、下表面侧得的入射角都为90°",    "answer": "B",    "analysis": "【分析】根据结合关系，结合折射定律\(n = \frac{\sin\theta}{\sin\alpha}\)，且\(n = \frac{c}{v}\)，求出光在玻璃砖中传播的时间与折射率的关系，从而根据数学知识进行比较."}` |
| **Paper**<img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/author.png" width="200"><img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/table.png" width="200"><img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/chart.png" width="200"><img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/bibtex2.jpg" width="200"> | Author Metadata Extraction | `"Convert author metadata to JSON: {"Author Information": [{"Name": "...", "Affiliation": "...", "Role": "First Author/Corresponding Author/Author"}]}"` | `{'Author Information': [{'Name': 'Victor Mundan', 'Affiliation': 'School of Nursing Sciences, University of Nairobi, P.O. Box 19676 (00202) KNH, Nairobi, Kenya; Armed Forces Memorial Hospital, Nairobi, Kenya', 'Role': 'First Author'}, {'Name': 'Margaret Muiva', 'Affiliation': 'School of Nursing Sciences, University of Nairobi, P.O. Box 19676 (00202) KNH, Nairobi, Kenya', 'Role': 'Author'}, {'Name': 'Samuel Kimani', 'Affiliation': 'School of Nursing Sciences, University of Nairobi, P.O. Box 19676 (00202) KNH, Nairobi, Kenya', 'Role': 'Corresponding Author'}]}` |
|                                                              | Table Extraction           | `"Extract all references on this page and output in BibTeX format"` | `\\begin{tabular}{lccccc}\n\\hline\nCOD\\textsubscript{cr} & pH & Turbidity & S & $\\zeta$-potential & Water \\\\\n(mg/L) & & (NTU) & (mg/L) & (mV) & Temperature \\\\\n& & & & & ($^{\\circ}$C) \\\\\n\\hline\n310–740 & 5–8 & 100–500 & 400–800 & $-18.4$–$-22.6$ & 15–25 \\\\\n\\hline\n\\end{tabular}` |
|                                                              | Paragraph Extraction       | `"Extract the paragraph beginning with \'In order to\'".`    | `In order to research the effect of the pulse direct current field on membrane flux, the change in membrane flux was measured at three different electric fields: 20 V/cm steady direct current field, 20 V/cm electric field with 10 pulse duration, 30 min pulse interval, and no appending electric field. These three kinds of electric fields were applied separately for 90 min. 2.1 can be seen from Fig. 2, that the sewage membrane flux in pulsed direct current field varies between the values for membrane flux in continuous direct current field and no electric field, under the same electric field strength. ` |
|                                                              | Chart Extraction           | ` "Extract raw chart data from specified PDF coordinates ('123', '65', '453', '222') and output as Markdown table with the first row fixed as header. Do not add any extra explanations or comments."` | `"\|Characteristic \|Population in millions \|\n\|--- \|--- \|\n\|2026* \|55.57 \|\n\|2025* \|54.38 \|\n\|2024* \|53.22 \|\n\|2023* \|52.06 \|\n\|2022* \|50.92 \|\n\|2021* \|49.8 \|\n\|2020* \|48.89 \|\n\|2019* \|47.6 \|\n\|2018 \|46.5 \|\n\|2017 \|45.4 \|\n\|2016 \|44.3 \|"` |
|                                                              | Reference Extraction       | `"Extract all references on this page and output in BibTeX format"` | `\```bibtex @article{Farris2000,    author = {Farris, K.B. and Stenton, S.B. and Samnani, M.},    title = {How satisfied are your patients?},    journal = {CPJ},    year = {2000},    pages = {133:32 - 36} } @article{Giese2000,    author = {Giese, J.L. and Cote, J.A.},    title = {Defining consumer satisfaction},    journal = {Academy of Marketing Science Review},    volume = {2000},    number = {No.1},    year = {2000},    note = {Available from: <http://www.amsreview.org/giese01 - 2000.pdf>},    accessed = {01.20.06} } @article{Donabedian1988,    author = {Donabedian, A.},    title = {The quality of care. How can it be assessed?},    journal = {JAMA},    year = {1988},    volume = {260},    pages = {1743 - 1748} } @article{Kucukarslan2002,    author = {Kucukarslan, S. and Schommer, J.C.},    title = {Patients' expectations and satisfaction with pharmacy services},    journal = {J Am Pharm Assoc},    year = {2002},    volume = {42},    pages = {489 - 496} } @article{Johnson1999,    author = {Johnson, J.A. and Coons, S.J. and Hays, R.D. and Pickard, A.S.},    title = {Health status and satisfaction with pharmacy services},    journal = {J Manag Care},    year = {1999},    pages = {51:63 - 170} } @article{Mackinnon2002,    author = {Mackinnon, G.E. and Mahrous, H.},    title = {Assessing consumers' interest in health care services offered in community pharmacies},    journal = {J Am Pharm Assoc},    year = {2002},    pages = {42:512 - 515} } @article{Kamei2002,    author = {Kamei, M. and Teshima, K. and Fukushima, N. and Nakamura, T.},    title = {Investigation of patients' demand for community pharmacies: relationship between pharmacy services and patient satisfaction},    journal = {Yakugaku Zasshi},    year = {2001},    pages = {121:215 - 220} } @article{Aharony1995,    author = {Aharony, L. and Strasser, S.},    title = {Patient satisfaction: What we know about and what we still need to explore},    journal = {Med Care Rev},    year = {1995},    pages = {50:49 - 79} } @article{Oparah2004,    author = {Oparah, A.C. and Enaore, E.F.O. and Akoria, A.O.},    title = {Assessment of patient satisfaction with pharmaceutical services in a Nigerian teaching hospital},    journal = {Int J Pharm Pract},    year = {2004},    pages = {12:7 - 12} } @article{Oparah2005,    author = {Oparah, A.C. and Elerekian, A.E.},    title = {Attitudes of Nigerian pharmacists towards pharmaceutical care},    journal = {Pharm World Sci},    year = {2005},    pages = {27:208 - 214} } @article{Larson2002,    author = {Larson, L.N. and Rovers, J.P. and Mackeigan, L.D.},    title = {Patient satisfaction with pharmaceutical care: update of a validated instrument},    journal = {J Am Pharm Assoc},    year = {2002},    pages = {42:44 - 50} } ``` ` |
| **Newspaper**<img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/20250107-185752.jpg"><img src="https://github.com/KnowledgeXLab/GDI-Bench/blob/main/image/newspaper.png"> | Title Extraction           | `An automated system requires newspaper headlines - extract each on separate lines with newline divisions.` | `浙江省农作物病虫害防治条例1月1日实施\n湖北省农业厅四个结合推进公开承诺\n从首创《村规民约》起步——广西宜州市合寨村“村民自治”30年纪实\n兰州市为农村五保户发物价补贴\n庄户剧团活跃乡间\n合肥市2010年农业执法挽回农民经济损失千万元\n庆元县农民收入香菇效益占一半以上\n灌云县80%农村五保老人实现集中供养\n科学示范带动安全用药\n饮水思源 倍感党亲` |
|                                                              | Information Extraction     | `Extract the header information of the newspaper and output it in JSON format.` | `{'日期': '2024年1月30日', '见习编辑': '张缘成', '新闻热线': '01084395098', 'E-mail': 'nmrbdianshang@126.com'}` |





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

