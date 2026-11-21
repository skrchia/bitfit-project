# BitFit 论文复现：GLUE RTE + BERT-base-cased

本仓库是自然语言处理项目实践（G10 组）的最终代码与结果。我们在 **GLUE RTE** 上，用 **BERT-base-cased** 对照复现了 ACL 2022 短文 [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://aclanthology.org/2022.acl-short.1/)，并补充了 **bias 子集消融** 与 **BitFit 学习率单因素实验**。

- **原论文**：Ben Zaken, Goldberg, Ravfogel. ACL 2022 (Short).  
- **官方代码**：https://github.com/benzakenelad/BitFit  
- **本组复现定位**：固定任务（RTE）与模型（bert-base-cased）后的**对照复现与消融**，而非与原文相同超参下的逐点数值复现。

## 复现了什么

在 GLUE RTE（训练 2,490 / 验证 277 / 测试 3,000）上，同一脚本、同一数据划分下横向比较 5 种可训练参数集合：

| 方法 | 可训练参数 | 参数量 | 学习率 |
|---|---|---|---|
| Full FT | 全部参数 | 108,311,810 | 2e-5 |
| BitFit | encoder 全部 bias + 分类头 | 104,450 | 5e-4 |
| Partial BitFit (query+intermediate) | query / intermediate / pooler bias + 分类头 | 48,386 | 5e-3 |
| Partial BitFit (query-only) | query bias + 分类头 | 10,754 | 5e-3 |
| Classifier-only | 仅分类头 | 1,538 | 1e-3 |

**消融**：bias 子集从“全部 bias”逐步收缩到“单一 bias”；BitFit 学习率单因素扫描（1e-4 / 5e-4 / 1e-3，seed=42）。

## 主结果

RTE 验证集，3 个种子（42 / 123 / 2026）的均值 ± 标准差。多数类基线为 52.71%。

| 方法 | 最优验证准确率 (%) | 可训练参数 | 参数占比 | 时间 (s) | 显存 (GiB) |
|---|---|---|---|---|---|
| Full FT | 68.11 ± 0.91 | 108,311,810 | 100% | 260.5 | 3.734 |
| BitFit | 66.67 ± 1.10 | 104,450 | 0.0964% | 158.5 | 2.486 |
| Partial (query+intermediate) | 63.90 ± 2.25 | 48,386 | 0.0447% | 155.1 | 2.482 |
| Partial (query-only) | 59.81 ± 1.50 | 10,754 | 0.0099% | 153.8 | 2.482 |
| Classifier-only | 56.44 ± 0.42 | 1,538 | 0.0014% | 81.0 | 2.118 |

**核心结论**：BitFit 以 0.0964% 的可训练参数，达到与全量微调统计上不可区分的性能（配对 McNemar 逐种子 p = 1.000 / 0.885 / 0.349）；bias 子集越小，性能越低；仅分类头接近多数类基线，说明只训分类头不足以利用预训练表示。

## 环境与安装

```bash
pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

`requirements.txt` 锁定版本：

```txt
torch==2.14.1+cu130
transformers==5.22.1
datasets==4.8.5
accelerate==1.15.0
tokenizers==0.23.3
numpy==2.4.6
pandas==2.2.3
tqdm==4.67.3
```

运行环境：Python 3.13 / CUDA 13.0 / Tesla T4。模型使用 Hugging Face `bert-base-cased` 默认 revision，实验期间未变更。

## 如何复现

仓库中包含一个自包含的 Jupyter Notebook，**只需顺序运行单元格**即可完成全部实验。Notebook 已支持断点续跑：已完成的 run 会自动跳过，中断后重跑不会覆盖已有结果。

1. 打开 Notebook；
2. 将 `ROOT` 修改为你自己的 Google Drive 路径（或本地路径）；
3. 顺序运行全部单元格。

全部 17 个 run（主实验 5×3=15，学习率扫描新增 2 个）在 T4 上约 45 分钟完成。结果会自动写入 `results/` 目录。

## 结果文件

| 文件 | 内容 |
|---|---|
| `summary.csv` | 17 个 run 的完整训练记录 |
| `three_seed_statistics.csv` | 5 种方法 × 3 种子的均值与标准差 |
| `parameter_audit.csv` | 可训练参数、实际更新参数、冻结参数是否变化 |
| `gradient_audit.csv` | 首 batch 梯度审计（缺失 / 全零 / 非有限） |
| `bitfit_lr_sweep_summary.csv` | BitFit 学习率扫描 |
| `bitfit_lr_sweep_error_changes.csv` | 相对 5e-4 的错误样本变化 |
| `predictions/*.csv` | 每个 run 的验证集与测试集逐条预测 |

**测试集**：GLUE 版 RTE 的 test 标签不公开，本仓库只保存 3,000 条测试集预测，不报告测试集准确率。

## 审计：如何证明实现的是 BitFit

本仓库不依赖参数名宣称实现，而是在 17 个 run 上完成三重检查：

1. **梯度审计**：首 batch 反传后，所有可训练张量均存在梯度，缺失 0 个，非有限 0 个；
2. **参数差异审计**：训练前保存全参数快照，训练后逐张量比较，冻结张量变化数为 0，可训练张量全部发生变化；
3. **更新集合规则断言**：按方法规则核对实际更新的参数名，违规即抛出异常。

审计逻辑已集成在 Notebook 中，运行结束时会自动断言。

## 引用

```bibtex
@inproceedings{ben-zaken-etal-2022-bitfit,
  title     = "{B}it{F}it: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
  author    = "Ben Zaken, Elad and Goldberg, Yoav and Ravfogel, Shauli",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
  year      = "2022",
  url       = "https://aclanthology.org/2022.acl-short.1/"
}
```

## 团队

G10 组：谢升均（组长）、蔡嘉迅、梁又文、哈拉哈提·乌热拉力。  
项目实践课程，2026 年 11 月。

---

### 关于目录结构

如果你的仓库不是 Notebook 单文件，可以补充一个精简的目录树：

```text
.
├── README.md
├── requirements.txt
├── notebook/
│   └── BitFit_main.ipynb
├── results/
│   ├── summary.csv
│   ├── three_seed_statistics.csv
│   ├── parameter_audit.csv
│   ├── gradient_audit.csv
│   └── predictions/
└── report/
    └── BitFit_最终报告.pdf
```
