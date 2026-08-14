# MMG-MPP 结构框架图

## 整体架构

```mermaid
graph TB
    SMILES["SMILES 分子"]

    subgraph Views["四视图构建"]
        AG["🔵 原子图(ME)<br/>GINEConv edge_dim=512<br/>节点=原子 边=化学键"]
        JT["🟢 联结树(MT)<br/>GINEConv edge_dim=512<br/>节点=子结构簇 边=簇间键"]
        FG["🟠 片段图(FM)<br/>GINEConv edge_dim=512<br/>节点=功能团 边=片段间键"]
        D3["🔴 3D构象图(MG)<br/>手工radius_graph<br/>ETKDG+Gasteiger"]
    end

    subgraph Encoders["编码器 31.9M — 三路统一GINEConv"]
        ME_E["ME: GIN-Transformer<br/>4×GINEConv + GRU虚拟节点<br/>4×Transformer(8头) → (B,512)"]
        MT_E["MT: Tree Encoder<br/>4×GINEConv → MLP → (B,512)"]
        FM_E["FM: Fragment Encoder<br/>4×GINEConv + FlashAttention<br/>→ (B,512)"]
        MG_E["MG: 3D图编码器<br/>高斯距离展开+连续滤波<br/>3层消息传递 → (B,512)"]
    end

    subgraph Fusion["GatedFusion 3-Stage"]
        S1["Stage1: ME+MT<br/>gate₁=σ(FC(concat)) 双残差"]
        S2["Stage2: h_GT+FM<br/>gate₂=σ(FC(concat)) 双残差"]
        S3["Stage3: h_GTF+MG<br/>gate₃=σ(FC(concat)) 双残差"]
        S1 --> S2 --> S3
    end

    subgraph Pretrain["预训练 InfoNCE-only"]
        CL["Chem-Aware InfoNCE<br/>FM锚点 | ME+MT+MG正样本<br/>Tanimoto加权 τ=0.07<br/>batch=128 patience=20"]
    end

    subgraph Finetune["下游微调 11数据集"]
        LOSS["Focal(γ=2)+LabelSmooth(0.05)+pos_weight<br/>多标签: TaskSharedHead(任务嵌入+自注意力)"]
        TASKS["6分类 | 5回归"]
    end

    SMILES --> AG & JT & FG & D3
    AG --> ME_E
    JT --> MT_E
    FG --> FM_E
    D3 --> MG_E
    ME_E & MT_E & FM_E & MG_E --> S1
    S3 --> CL
    S3 --> LOSS --> TASKS
```

## 关键设计

| 组件 | 选型 |
|------|------|
| ME/MT/FM图卷积 | **GINEConv (三路统一 edge_dim=512)** |
| MG图卷积 | 手工radius_graph + 高斯滤波 |
| 虚拟节点 | GRUCell (仅ME) |
| FM注意力 | FlashAttention O(N) |
| 融合 | GatedFusion 3-stage (ME+MT→+FM→+MG) |
| 预训练Loss | Chem-Aware InfoNCE-only |
| 分类Loss | Focal(γ=2) / LabelSmooth(0.05) |
| 早停 | 验证驱动 patience=20 |
| 断点 | latest_checkpoint.pt auto-resume |
| 参数 | **31.9M** |
| 预训练数据 | ZINC+ChEMBL Tanimoto精选 ~450K |
| 数据集 | **11** (6分类+5回归) |
