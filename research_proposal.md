# MMG-MPP: 多视图分子图表征预训练框架 — 实现研究方案

## 摘要

MMG-MPP 提出一种四视图分子预训练框架，将原子图（ME）、联结树（MT）、片段图（FM）、3D构象图（MG）四种分子表征通过门控残差融合（GatedFusion）整合，采用化学感知对比学习（Chem-Aware InfoNCE）在 ZINC+ChEMBL 精选数据集上预训练。下游微调时使用 Focal Loss + Label Smoothing + TaskSharedHead + 数据集感知超参策略。在 11 个 MoleculeNet 基准上评估。

---

## 1. 问题陈述

分子性质预测（MPP）面临三个核心挑战：

1. **标注稀缺**：MoleculeNet 半数数据集不足 2000 分子，从头训练 GNN 无法有效学习
2. **表征碎片化**：原子图（2D拓扑）、联结树（子结构层级）、功能团（化学语义）提供互补信息，现有方法只用单一视图
3. **可解释性缺失**：黑盒 GNN 难以与化学家领域知识对齐

## 2. 方法

### 2.1 四视图分子表征

给定分子 SMILES $m$，通过 RDKit 构建：

**ME — 原子图** $\mathcal{G}_A = (V_A, E_A)$
- 节点：148 维原子特征（类型、杂化、手性、电荷）
- 边：11 维键特征（类型、共轭、环信息）

**MT — 联结树** $\mathcal{G}_J = (V_J, E_J)$
- 环检测 + 非环原子合并 → 子结构簇
- 簇间共价键为边，捕获分子骨架层级

**FM — 片段图** $\mathcal{G}_F = (V_F, E_F)$
- 22 种 SMARTS 预定义功能团匹配
- BRICS 键断裂处理剩余原子
- 提供化学语义对齐

**MG — 3D 距离编码
- ETKDG 构象 + Gasteiger 电荷
- 距离矩阵 + Coulomb 互作用 + 高斯模糊 → 小型 CNN → (B, 512)

### 2.2 编码器

```
ME: GINWithVirtualNode(4×GINEConv, edge_dim=512)
    → GRUCell 虚拟节点 → Transformer(4层, 8头) → (B, 512)

MT: JunctionTreeEncoder(4×GINEConv, edge_dim=512) → MLP → (B, 512)

FM: FragmentEncoder(4×GINEConv, edge_dim=512 + FlashAttention) → (B, 512)

MG: 手工radius_graph + 高斯距离展开 + 3层连续滤波 → (B, 512)
```

### 2.3 融合模块

**GatedFusion** — 三阶段逐维门控残差融合：

$$\mathbf{w}_1 = \sigma(\text{FC}([\mathbf{z}_{ME} \| \mathbf{z}_{MT}]))$$
$$\mathbf{h}_{GT} = \mathbf{w}_1 \odot \text{FC}(\mathbf{z}_{ME}) + (1-\mathbf{w}_1) \odot \text{FC}(\mathbf{z}_{MT}) + \mathbf{z}_{ME} + \mathbf{z}_{MT}$$

$$\mathbf{w}_2 = \sigma(\text{FC}([\mathbf{h}_{GT} \| \mathbf{z}_{FM}]))$$
$$\mathbf{h}_{GTF} = \mathbf{w}_2 \odot \text{FC}(\mathbf{h}_{GT}) + (1-\mathbf{w}_2) \odot \text{FC}(\mathbf{z}_{FM}) + \mathbf{h}_{GT} + \mathbf{z}_{FM}$$

Stage 3（可选）：$\mathbf{h}_{out} = \text{Gate}(\mathbf{h}_{GTF}, \mathbf{z}_{MG})$

备选模式：`cnn_attention` / `cross_attention`。

### 2.4 预训练

**Chem-Aware InfoNCE**：

片段嵌入 $\mathbf{z}_{FM}$ 为锚点，多视图和 $\mathbf{z}_{ME}+\mathbf{z}_{MT}$ 为正样本。Tanimoto 相似度加权负样本：

$$w_{ij} = \frac{1 - \text{Tanimoto}(m_i, m_j)}{\sum_{k \neq i} (1 - \text{Tanimoto}(m_i, m_k))} \cdot (B-1)$$

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_i \log \frac{\exp(\mathbf{z}_{FM}^i \cdot \mathbf{z}_{GT}^i / \tau)}{\exp(\mathbf{z}_{FM}^i \cdot \mathbf{z}_{GT}^i / \tau) + \sum_{j \neq i} w_{ij} \exp(\mathbf{z}_{FM}^i \cdot \mathbf{z}_{GT}^j / \tau)}$$

| 参数 | 值 |
|------|-----|
| batch_size | 128 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| epochs | 100 |
| warmup | 10 epochs (线性) |
| info_nce_tau | 0.07 |
| dropout | 0.1 |
| grad_clip | 5.0 |
| scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| 早停 | 验证驱动, patience=20 |
| 断点 | latest_checkpoint.pt auto-resume |

### 2.5 微调

#### 数据集分层超参

| 层级 | 分子数 | 数据集 | Dropout | LR | WD | Epochs | Patience |
|------|--------|--------|---------|-----|-----|--------|----------|
| Tiny | <1500 | SIDER, ClinTox, FreeSolv | 0.5-0.7 | 5e-5~1e-4 | 1e-3~5e-3 | 120 | 40 |
| Small | 1500-2500 | BBBP, BACE, ESOL | 0.3-0.7 | 1e-4~5e-4 | 1e-4~1e-3 | 100 | 30 |
| Medium | 4000-8000 | Tox21, Lipo, QM7 | 0.2-0.5 | 1e-4~1e-3 | 1e-5~1e-4 | 80 | 25-30 |
| Large | >20000 | HIV, QM8 | 0.1-0.3 | 1e-4~5e-4 | 1e-5~1e-4 | 80 | 15-20 |

#### Loss 策略

- **多标签**（Tox21, SIDER, ClinTox, QM8）：TaskSharedHead（任务嵌入+自注意力共享学习）+ Focal Loss (γ=2) + pos_weight
- **单标签**（BBBP, BACE, HIV）：Label Smoothing (0.05)

#### 评估协议

- Scaffold Split (80/10/10)，5 轮独立实验（seed 42-46）
- 分类：ROC-AUC (mean ± std )
- 回归：RMSE (ESOL/FreeSolv/Lipo), MAE (QM7/QM8)
- 从 checkpoint config 重建架构，确保预训练-微调一致

### 2.6 Lamole 可解释性   

```
预测 → Attention+Gradient Hadamard积 → 片段级重要性
     → 功能团标注映射 → 原子级热力图
     → 化学家标注 mask → Marginal Loss 对齐
```

## 3. 实验设计

### 3.1 数据集

| 类型 | 数据集 | 分子数 | 任务数 | 指标 |
|------|--------|--------|--------|------|
| 分类 | BBBP | 2,050 | 1 | ROC-AUC |
| 分类 | Tox21 | 7,831 | 12 | ROC-AUC |
| 分类 | SIDER | 1,427 | 27 | ROC-AUC |
| 分类 | ClinTox | 1,484 | 2 | ROC-AUC |
| 分类 | BACE | 1,522 | 1 | ROC-AUC |
| 分类 | HIV | 41,127 | 1 | ROC-AUC |
| 回归 | ESOL | 1,128 | 1 | RMSE |
| 回归 | FreeSolv | 642 | 1 | RMSE |
| 回归 | Lipo | 4,200 | 1 | RMSE |
| 回归 | QM7 | 6,834 | 1 | MAE |
| 回归 | QM8 | 21,786 | 12 | MAE |

### 3.2 预训练数据

- ZINC 250K + ChEMBL 500K → 合并 749K → Tanimoto(>0.3) 精选 → ~450K
- 过滤：≤100 原子，RDKit 可解析

### 3.3 消融实验

| 类别 | 实验 | 变量 | 开关 |
|------|------|------|------|
| 架构 | 视图消融 | ME→ME+MT→ME+FM→ME+MT+FM | 分支开关 |
| 架构 | **3D视图消融** | 3-view vs 4-view (+MG) | `use_mph` |
| 架构 | 融合消融 | Gated vs CNN-Attn vs Cross-Attn | `fusion_mode` |
| 架构 | 边特征消融 | GINConv vs GINEConv (三路) | 替换 conv 类型 |
| 训练 | Chem-Aware NCE | True vs False | `chem_aware_nce` |
| 训练 | Focal Loss | BCE→BCE+pos_weight→Focal(γ=2) | `focal_gamma` |
| 训练 | **TaskSharedHead** | 独立MLP→任务嵌入+自注意力共享 | `num_tasks>1` |
| 数据 | 预训练规模 | 100K→200K→450K | 子采样 |
| 数据 | 分子筛选 | 随机 vs Tanimoto精选 | CSV 切换 |
| 基线 | **无预训练** | 随机初始化 vs 预训练 | `--scratch` |

### 3.4 对标基线

GROVER, GraphMVP, MolCLR, GINE (有监督), AttentiveFP

## 4. 技术栈

| 组件 | 选型 |
|------|------|
| 图神经网络 | PyTorch Geometric (GINEConv, GINConv) |
| Transformer | PyTorch nn.TransformerEncoder |
| 注意力 | FlashAttention (F.scaled_dot_product_attention) |
| 化学信息学 | RDKit (SMILES, SMARTS, BRICS, Morgan FP) |
| 预训练 | InfoNCE (Chem-Aware) |
| GPU | NVIDIA RTX 3090 (24GB) |
| 参数规模 | 31.9M |

## 5. 时间线

| 阶段 | 内容 | 预估 |
|------|------|------|
| 预训练 | ~450K 分子, batch=128, ~70 epochs (早停) | ~90 小时 (首epoch 6h+后续~1.4h/epoch) |
| 微调 | 11 数据集, 每数据集 5 轮 | ~12 小时 |
| 消融 | 9组实验 (架构/训练/数据) | ~48 小时 |
| 可解释性 | Lamole 归因 | ~2 小时 |
| **总计** | | **~152 小时 (~6.3 天)** |

