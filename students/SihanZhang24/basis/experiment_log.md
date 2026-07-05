# SURF-2026 实验记录：basis4,11,12的完成模型与消融实验

> **实验环境总览**
> - **平台**: AutoDL 算力云
> - **GPU**: RTX 5090 (32 GB) × 1卡
> - **PyTorch 环境**: PyTorch 2.8.0; Python 3.12 (Ubuntu 22.04); CUDA 12.8
> - **随机种子 (seed)**: 42
> - **训练配置**: epochs = 60, batch_size = 32, learning_rate = 0.001, grad_weight = 0.1
> - **数据集划分**: 训练集 3500 / 验证集 300 / 测试集 100


## 完整模型 (Full Model) 实验结果

### Basis 4 (Full Model)

**基础配置**
- 训练目标: 第 4 个基函数 (Zero-based Index: 3)
- 运行模式: `full`
- 随机种子: `42`

**最终测试集指标**
| 指标 | 值 |
|:---|:---|
| MSE | 0.01462 |
| RMSE | 0.12092 |
| MAE | 0.03943 |
| **R²** | **0.9173** |
| Relative L2 | 0.29005 |
| Grad MSE | 0.05238 |
| Max Abs Error | 1.75944 |
| n_samples | 100 |

**运行信息**
- 训练耗时: ~700 s
- 输出目录: `outputs/teaching_single_basis/basis_04/`
- 模型权重: `best_model/basis_04_best.pt`

---

### Basis 11 (Full Model)

**基础配置**
- 训练目标: 第 11 个基函数 (Zero-based Index: 10)
- 运行模式: `full`
- 随机种子: `42`

**最终测试集指标**
| 指标 | 值 |
|:---|:---|
| MSE | 0.00292 |
| RMSE | 0.05399 |
| MAE | 0.02356 |
| **R²** | **0.8943** |
| Relative L2 | 0.32700 |
| Grad MSE | 0.01312 |
| Max Abs Error | 0.83834 |
| n_samples | 100 |

**运行信息**
- 训练耗时: 381 s
- 输出目录: `outputs/teaching_single_basis/basis_11/`
- 模型权重: `best_model/basis_11_best.pt`

---

### Basis 12 (Full Model)

**基础配置**
- 训练目标: 第 12 个基函数 (Zero-based Index: 11)
- 运行模式: `full`
- 随机种子: `42`

**最终测试集指标**
| 指标 | 值 |
|:---|:---|
| MSE | 0.00191 |
| RMSE | 0.04366 |
| MAE | 0.01813 |
| **R²** | **0.9616** |
| Relative L2 | 0.19360 |
| Grad MSE | 0.00804 |
| Max Abs Error | 0.64456 |
| n_samples | 100 |

**运行信息**
- 训练耗时: 302 s
- 输出目录: `outputs/teaching_single_basis/basis_12/`
- 模型权重: `best_model/basis_12_best.pt`


## 消融实验 (Ablation Study) 结果

### 消融实验一：移除 FNO 模块 (no_fno)

> **实验目的**: 验证 FNO (Fourier Neural Operator) 预处理层对模型性能的贡献。
> **代码修改**: `use_fno=False` (将 FNO 替换为 Identity 层)

#### Basis 4 (full_no_fno)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00959 | 下降 34.4% |
| RMSE | 0.09791 | 下降 19.0% |
| **R²** | **0.9457** | **提升 0.0284** |
| Relative L2 | 0.23237 | 下降 19.9% |
| Max Abs Error | 1.79313 | — |

- 训练耗时: ~267 s
- 输出目录: `outputs/teaching_single_basis/basis_04/ablation_no_fno/`

#### Basis 11 (full_no_fno)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00265 | 下降 9.0% |
| RMSE | 0.05149 | 下降 4.6% |
| **R²** | **0.9038** | **提升 0.0095** |
| Relative L2 | 0.30877 | 下降 5.6% |
| Max Abs Error | 0.68106 | — |

- 训练耗时: ~265 s
- 输出目录: `outputs/teaching_single_basis/basis_11/ablation_no_fno/`

#### Basis 12 (full_no_fno)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00251 | 上升 31.7% |
| RMSE | 0.05010 | 上升 14.8% |
| **R²** | **0.9494** | **下降 0.0122** |
| Relative L2 | 0.22468 | 上升 16.0% |
| Max Abs Error | 0.63486 | — |

- 训练耗时: ~124 s
- 输出目录: `outputs/teaching_single_basis/basis_12/ablation_no_fno/`

---

### 消融实验二：移除注意力模块 (no_attention)

> **实验目的**: 验证 Attention Gate (注意力门控) 对模型性能的贡献。
> **代码修改**: 将 `UpAtt` (带注意力的上采样) 替换为普通 `Up` 模块

#### Basis 4 (full_no_attention)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.01365 | 下降 6.6% |
| RMSE | 0.11685 | 下降 3.4% |
| **R²** | **0.9227** | **提升 0.0054** |
| Relative L2 | 0.27662 | 下降 4.6% |
| Max Abs Error | 1.88087 | — |

- 训练耗时: ~343 s
- 输出目录: `outputs/teaching_single_basis/basis_04/ablation_no_attention/`

#### Basis 11 (full_no_attention)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00249 | 下降 14.5% |
| RMSE | 0.04992 | 下降 7.5% |
| **R²** | **0.9096** | **提升 0.0153** |
| Relative L2 | 0.30149 | 下降 7.8% |
| Max Abs Error | 0.69097 | — |

- 训练耗时: ~345 s
- 输出目录: `outputs/teaching_single_basis/basis_11/ablation_no_attention/`

#### Basis 12 (full_no_attention)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00164 | 下降 13.7% |
| RMSE | 0.04055 | 下降 7.1% |
| **R²** | **0.9669** | **提升 0.0053** |
| Relative L2 | 0.17962 | 下降 7.2% |
| Max Abs Error | 0.63991 | — |

- 训练耗时: ~342 s
- 输出目录: `outputs/teaching_single_basis/basis_12/ablation_no_attention/`

---

### 消融实验三：移除梯度损失 (no_gradient_loss)

> **实验目的**: 验证梯度匹配项 (gradient loss) 对模型性能的贡献。
> **代码修改**: `GRAD_WEIGHT = 0` (移除梯度损失项)
> **注**: 由于梯度损失被移除，`grad_mse` 不适用，标记为 "未计算"

#### Basis 4 (full_no_gradient_loss)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.01159 | 下降 20.7% |
| RMSE | 0.10766 | 下降 11.0% |
| **R²** | **0.9344** | **提升 0.0171** |
| Relative L2 | 0.25712 | 下降 11.4% |
| Grad MSE | 未计算 | — |
| Max Abs Error | 1.75206 | — |

- 训练耗时: ~314 s
- 输出目录: `outputs/teaching_single_basis/basis_04/ablation_no_gradient_loss/`

#### Basis 11 (full_no_gradient_loss)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00310 | 上升 6.5% |
| RMSE | 0.05572 | 上升 3.2% |
| **R²** | **0.8874** | **下降 0.0069** |
| Relative L2 | 0.33664 | 上升 2.9% |
| Grad MSE | 未计算 | — |
| Max Abs Error | 0.67747 | — |

- 训练耗时: 378 s
- 输出目录: `outputs/teaching_single_basis/basis_11/ablation_no_gradient_loss/`

#### Basis 12 (full_no_gradient_loss)

| 指标 | 值 | 与完整模型对比 |
|:---|:---|:---|
| MSE | 0.00205 | 上升 7.8% |
| RMSE | 0.04532 | 上升 3.8% |
| **R²** | **0.9586** | **下降 0.0030** |
| Relative L2 | 0.20078 | 上升 3.7% |
| Grad MSE | 未计算 | — |
| Max Abs Error | 0.64081 | — |

- 训练耗时: ~324 s
- 输出目录: `outputs/teaching_single_basis/basis_12/ablation_no_gradient_loss/`


## 消融实验总结与讨论

### R² 指标对比汇总

| 模型变体 | Basis 4 | Basis 11 | Basis 12 |
|:---|:---|:---|:---|
| **完整模型 (Baseline)** | **0.9173** | **0.8943** | **0.9616** |
| 去除 FNO | 0.9457 ✅ (+0.0284) | 0.9038 ✅ (+0.0095) | 0.9494 ❌ (-0.0122) |
| 去除注意力 | 0.9227 ✅ (+0.0054) | 0.9096 ✅ (+0.0153) | 0.9669 ✅ (+0.0053) |
| 去除梯度损失 | 0.9344 ✅ (+0.0171) | 0.8874 ❌ (-0.0069) | 0.9586 ❌ (-0.0030) |

> **结论**:
> - **注意力模块 (Attention)**：移除后 R² 在三个基函数上均有提升，表明当前任务中注意力机制对性能有轻微抑制作用，可能是该架构在 128×128 网格上带来了过参数化，或注意力门控的额外参数在样本量不足以支撑其表达力。
> - **FNO 模块**：对 Basis 4 和 Basis 11 有轻微负贡献，但对 Basis 12 有正贡献，表明 FNO 的作用与基函数频率特性相关。
> - **梯度损失 (Gradient Loss)**：在 Basis 4 上有明显正贡献，但在 Basis 11 和 Basis 12 上移除后性能略有提升，表明梯度损失的效果与基函数本身的空间振荡特性相关。
> - **整体趋势**：所有消融变体在 Basis 12 上均达到了 R² > 0.94，最低为 0.9494，说明该基函数更容易学习；而 Basis 11 的 R² 相对较低（0.8874–0.9096），表明其空间模式更为复杂，是更具挑战性的预测目标。

> **重要说明**：
> - 部分消融实验（如 Basis 12 no_fno）显示出与完整模型相近甚至更优的性能，这可能意味着该基函数的学习主要依赖 U-Net 的卷积表达能力，FNO 的频域建模优势在其空间模式上未能充分体现。
> - 上述指标对比以 `summary_basis_XX.json` 中的测试集结果为准。本报告中的 R² 值均来自测试集（Test Set），确保了不同模型变体之间的公平性。

---

*报告生成日期: 2026-07-05*