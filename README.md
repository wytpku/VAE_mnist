# VAE_mnist

在 MNIST 上从零实现并训练一个变分自编码器（VAE），并通过 β-VAE 做了一组关于 disentanglement 的消融实验。这是我自学 Stanford [CS236 (Deep Generative Models)](https://deepgenerativemodels.github.io/) 过程中的配套练习项目。

## 项目结构

```
VAE_mnist/
├── vae.py                    # Encoder / Decoder / VAE / 重参数化 / vae_loss
├── train.py                  # 第一阶段训练脚本（基础 VAE，argparse + checkpoint + 可视化 + loss CSV）
├── train_beta.py             # 第二阶段训练脚本（支持 --beta / --seed，用于 β-VAE 消融实验）
├── visualize.py               # 可视化工具：reconstruction / random sample / latent interpolation / latent traversal
├── requirement.txt           # 依赖列表
├── data/MNIST/raw/           # MNIST 数据（首次运行自动下载）
├── checkpoints/              # 各轮实验保存的模型权重
├── report/                   # 各轮实验保存的图片与 loss_log.csv
├── report_1.zip              # 第一轮实验的完整产出打包
├── 第一轮实验报告.docx        # 第一轮实验报告（基础 VAE）
├── 第二轮实验报告.docx        # 第二轮实验报告（β-VAE 消融实验）
├── beta_vae_notes.pdf        # disentanglement / β-VAE 学习笔记
└── README.md
```

## 环境配置

```bash
pip install -r requirement.txt
```

## 使用方法

**第一阶段：训练基础 VAE**（MLP encoder/decoder，784→400→20→400→784，标准 ELBO，即 β=1）

```bash
python train.py --epochs 10
```

**第二阶段：β-VAE 消融实验**（固定随机种子，仅改变 β，观察重建质量与隐空间正则化的 trade-off）

```bash
python train_beta.py --beta 0.5 --epochs 5 --seed 42
python train_beta.py --beta 1.0 --epochs 5 --seed 42
python train_beta.py --beta 2.0 --epochs 5 --seed 42
python train_beta.py --beta 4.0 --epochs 5 --seed 42
```

每轮训练都会在 `report/` 下保存 reconstruction grid、random sample grid、latent interpolation grid、latent traversal grid，以及记录 loss 的 CSV；`checkpoints/` 下保存每个 epoch 的模型权重。

## 实验记录

### 第一轮：基础 VAE

Adam 优化器，lr=1e-3，batch_size=128，训练 10 个 epoch。

| epoch | total_loss | recon_loss | kl_loss |
|---|---|---|---|
| 1 | 214.18 | 201.39 | 12.79 |
| 10 | 111.12 | 86.36 | 24.75 |

- `recon_loss` 显著下降，重建质量肉眼可见提升；`kl_loss` 上升但增速明显放缓，训练收敛健康。
- 随机采样（生成）质量提升不如重建图稳定、锐利，推测与隐空间的聚合后验未完全覆盖先验分布（hole problem）有关。
- 详见 [`第一轮实验报告.docx`](./第一轮实验报告.docx)。

### 第二轮：β-VAE 消融实验（β ∈ {0.5, 1.0, 2.0, 4.0}）

其余超参数与第一轮一致，每组训练 5 个 epoch，seed=42。

| β | recon_loss（epoch 5） | kl_loss（未加权） | 坍缩维度数 / 20 |
|---|---|---|---|
| 0.5 | 78.4 | 33.4 | 0 |
| 1.0 | 84.7 | 24.8 | 0 |
| 2.0 | 97.7 | 16.1 | 1 |
| 4.0 | 118.8 | 8.9 | 9 |

- `recon_loss` 随 β 单调上升，原始 `kl_loss` 随 β 单调下降，符合 β-VAE 理论预期。
- β=4 时近一半隐空间维度坍缩（对 traversal 不敏感），可用自由度骤降；综合权衡后选定 β≈2 作为后续重点观察对象。
- 详见 [`第二轮实验报告.docx`](./第二轮实验报告.docx)。

## Disentanglement / β-VAE 笔记

关于 disentanglement 直觉、β-VAE 的信息瓶颈视角、重建质量与正则化的 trade-off、posterior collapse 等内容，整理在 [`beta_vae_notes.pdf`](./beta_vae_notes.pdf) 中。

## 后续计划

- 对 β≈2 附近做更细粒度的扫描，并对未坍缩的活跃维度做定性检验（观察逐维度扫描时是否只改变单一语义属性）
- 尝试 KL warm-up / 退火、free bits 等技术，缓解 β 增大对重建质量的负面影响
- 视情况尝试将 MLP encoder/decoder 替换为卷积结构

## 致谢

本项目的代码调试与实验分析部分，借助 Claude（Anthropic）提供的讲解与辅助完成。

