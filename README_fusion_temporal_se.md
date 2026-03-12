# fusion_temporal_se 使用说明

## 这次改动做了什么

- 在 `model_factory.py` 中新增了 `fusion_temporal_se` 模型。
- 在 `fusion.py` 中新增了 `--model fusion_temporal_se` 选项。
- `fusion.py` 现在会按模态分别做通道选择：
  - EEG 使用 `TDPSD` Fisher 分数；
  - fNIRS 使用 `HbO/HbR` 成对的 `mean/std/slope` Fisher 分数；
  - 之后再把选中的通道重排为：`EEG 在前，fNIRS 在后`。
- 模型内部使用双分支结构：
  - EEG 分支单独编码 EEG 通道特征；
  - fNIRS 分支单独编码 fNIRS 通道特征；
  - 两个分支的 embedding 在融合头中拼接，再做分类。

## 版本控制建议

- 旧模型 `shallow`、`temporal_se` 保持不变，可以直接做对照实验。
- 新模型使用新的模型名 `fusion_temporal_se`，不会覆盖旧实验逻辑。
- 推荐先在小样本上验证，再做完整批量实验。

## 模型结构

`fusion_temporal_se` 的核心流程如下：

1. 输入张量形状为 `(batch, channels, time)`。
2. `fusion.py` 先分别完成 EEG / fNIRS 的 Fisher 通道筛选，再按模态把选中通道重排成：
   - 前半部分：EEG 通道
   - 后半部分：fNIRS 通道
3. 模型内部切分为两条分支：
   - `eeg_x = x[:, :eeg_channels, :]`
   - `fnirs_x = x[:, eeg_channels:, :]`
4. 两条分支分别通过 `TemporalSEEncoder` 提取 embedding。
5. 在特征维拼接：`torch.cat([eeg_emb, fnirs_emb], dim=1)`。
6. 通过融合头分类。

## 运行方式

### 1. 使用旧的通用模型

```bash
python fusion.py --model temporal_se --files_limit 1
```

### 2. 使用新的双分支融合模型

```bash
python fusion.py --model fusion_temporal_se --files_limit 1
```

### 3. 使用共享配置文件中的默认训练参数

```bash
python fusion.py --model fusion_temporal_se
```

### 4. 临时覆盖共享配置

```bash
python fusion.py --model fusion_temporal_se --batch_size 64 --epochs 50
```

### 5. 显式指定两个模态各自保留多少通道

```bash
python fusion.py --model fusion_temporal_se --top_k_eeg 32 --top_k_fnirs 20
```

说明：
- `--top_k_fnirs` 必须是偶数，因为 fNIRS 会按 HbO/HbR 成对保留。
- 如果不传这两个参数，脚本会根据当前总 `top_k` 按模态规模自动分配。

### 6. 通过 all.py 批量运行 fusion 专属模型

```bash
python all.py --scripts fusion --model_fusion fusion_temporal_se --top_k_eeg 32 --top_k_fnirs 20 --files_limit 1
```

如果要同时跑三个模态，但只让 fusion 使用新模型，可以这样：

```bash
python all.py --scripts eeg fnirs fusion --model_eeg temporal_se --model_fnirs temporal_se --model_fusion fusion_temporal_se --top_k_eeg 32 --top_k_fnirs 20
```

## 对照实验建议

建议按下面顺序做实验：

1. 先跑 baseline：

```bash
python fusion.py --model temporal_se --files_limit 1
```

2. 再跑新模型：

```bash
python fusion.py --model fusion_temporal_se --files_limit 1
```

3. 确认日志、输出目录、收敛曲线都正常后，再去掉 `--files_limit 1` 跑完整实验。

## 输出与日志

- 日志仍然保存在：`<output_root>/Logs/`
- Fusion 结果仍然保存在：`<output_root>/ResFusion/`
- 当使用 `fusion_temporal_se` 时，`selected_channels.csv` 中会额外保存：
  - `ordered_idx`
  - `ordered_name`
  - `modality`

这两个字段表示实际送入双分支模型时使用的通道顺序。

## 注意事项

- `fusion_temporal_se` 要求本轮选中的通道里同时包含 EEG 和 fNIRS；如果某一轮筛选后只剩单一模态，脚本会报错并提示。
- 当前 fusion 的通道选择不再共用单一特征，而是：EEG 用 TDPSD，fNIRS 用 pair-wise `mean/std/slope`。
- `all.py` 现在已经支持按脚本分别指定模型：`--model_eeg`、`--model_fnirs`、`--model_fusion`。
- `all.py` 也已经支持把 `--top_k_eeg`、`--top_k_fnirs` 传给 fusion 脚本。
