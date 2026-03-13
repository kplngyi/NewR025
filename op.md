# all.py 训练使用说明

## 作用

`all.py` 是统一训练入口，用来批量启动以下脚本：

- `eeg.py`
- `fnirs.py`
- `fusion.py`

它支持：

- 只运行指定模态脚本
- 同时运行多个脚本
- 扫描不同的 `batch_size` 和 `epochs`
- 串行或并行运行
- 给不同脚本单独指定模型
- 给 fusion 单独指定 EEG / fNIRS 保留通道数

## 基本行为

`all.py` 会根据 `--scripts` 选择要运行的脚本，并把共享参数转发给对应子脚本。

默认脚本集合为：

```bash
fusion eeg fnirs
```

默认模型参数为：

```bash
--model temporal_se
```

输出目录会按本轮实验参数自动创建，例如：

```text
runs/
  bs32_ep50/
    eeg/
    fnirs/
    fusion/
```

如果没有在 `all.py` 中显式传 `batch_size` 或 `epochs`，子脚本会回退到 `config.yaml` 中的默认值，对应目录名会写成：

```text
bs_config_ep_config
```

## 常用命令

### 1. 运行全部脚本

```bash
python all.py
```

### 2. 只跑 EEG

```bash
python all.py --scripts eeg
```

### 3. 只跑 fNIRS

```bash
python all.py --scripts fnirs
```

### 4. 只跑 fusion

```bash
python all.py --scripts fusion
```

### 5. 同时跑 EEG 和 fNIRS

```bash
python all.py --scripts eeg fnirs
```

## 模型选择

### 1. 三个脚本统一使用同一个模型

```bash
python all.py --model temporal_se
```

可选值为：

- `shallow`
- `temporal_se`
- `fusion_temporal_se`

说明：

- `eeg.py` 只支持 `shallow`、`temporal_se`
- `fnirs.py` 只支持 `shallow`、`temporal_se`
- `fusion.py` 支持 `shallow`、`temporal_se`、`fusion_temporal_se`

因此，如果你要同时跑全部脚本，通常更推荐统一使用：

```bash
python all.py --model temporal_se
```

### 2. 为不同脚本分别指定模型

```bash
python all.py \
  --scripts eeg fnirs fusion \
  --model_eeg shallow \
  --model_fnirs temporal_se \
  --model_fusion fusion_temporal_se
```

如果设置了 `--model_eeg`、`--model_fnirs` 或 `--model_fusion`，这些脚本级参数会覆盖全局 `--model`。

### 3. shallow 的行为

当前代码中，当模型为 `shallow` 时：

- 不启用 early stopping
- 不启用 `load_best`

因此 `shallow` 训练会直接跑满设定的 `max_epochs`。

## 扫描 batch_size 和 epochs

### 1. 指定单组参数

```bash
python all.py --batch_sizes 32 --epochs_list 50
```

### 2. 扫描多个 batch_size

```bash
python all.py --batch_sizes 16 32 64
```

### 3. 扫描多个 epochs

```bash
python all.py --epochs_list 50 100
```

### 4. 同时扫描 batch_size 和 epochs

```bash
python all.py --batch_sizes 16 32 --epochs_list 50 100
```

上面的命令会依次生成 4 组实验：

- `bs16_ep50`
- `bs16_ep100`
- `bs32_ep50`
- `bs32_ep100`

## 串行与并行运行

### 1. 串行运行

默认是串行模式：

```bash
python all.py --scripts eeg fnirs fusion
```

它会按顺序一个个启动子脚本。

### 2. 并行运行

```bash
python all.py --scripts eeg fnirs fusion --parallel --gpus 0 1 2
```

并行模式下：

- 每个子脚本会分配一个 GPU id
- `all.py` 通过 `CUDA_VISIBLE_DEVICES` 约束子进程只看到自己那张卡
- 当 GPU 数量少于脚本数量时会直接报错

如果并行运行 EEG，`all.py` 还会把子脚本设备参数规范成 `cuda:0`，因为在该子进程里已经只暴露了一张卡。

## 设备参数

`all.py` 提供：

```bash
--device cuda
```

需要注意：

- 这个参数只会显式转发给 `eeg.py`
- `fnirs.py` 和 `fusion.py` 当前仍然由各自脚本内部自行判断 `torch.cuda.is_available()`
- 如果你的环境不是 CUDA 环境，它们会自动退回 CPU

例如：

```bash
python all.py --scripts eeg --device cpu
```

## 数据目录参数

三个模态的数据目录是分开传递的：

```bash
--data_dir_eeg PPEEG
--data_dir_fnirs PPfNIRS
--data_dir_fusion FusionEEG-fNIRS
```

示例：

```bash
python all.py \
  --scripts eeg fnirs fusion \
  --data_dir_eeg PPEEG \
  --data_dir_fnirs PPfNIRS \
  --data_dir_fusion FusionEEG-fNIRS
```

## 配置与输出参数

### 1. 指定配置文件

```bash
python all.py --config_path config.yaml
```

### 2. 指定项目根目录

```bash
python all.py --project_root /path/to/project
```

### 3. 指定输出根目录

```bash
python all.py --output_root runs
```

例如：

```bash
python all.py --output_root experiments
```

生成结果会放到类似目录：

```text
experiments/bs32_ep50/eeg/
experiments/bs32_ep50/fnirs/
experiments/bs32_ep50/fusion/
```

## 通道搜索相关参数

`all.py` 会把以下参数转发给子脚本：

- `--files_limit`
- `--top_k_step`
- `--min_top_k`
- `--early_stop_patience`
- `--early_stop_monitor`
- `--early_stop_threshold`

示例：

```bash
python all.py --files_limit 1 --top_k_step 10 --min_top_k 20
```

这适合先做小规模验证。

## fusion 专属参数

如果运行 `fusion.py`，可以额外传：

- `--top_k_eeg`
- `--top_k_fnirs`

示例：

```bash
python all.py --scripts fusion --top_k_eeg 32 --top_k_fnirs 20
```

说明：

- `--top_k_fnirs` 必须是偶数
- 因为 fNIRS 是按 HbO/HbR 成对保留
- 如果不传这两个参数，`fusion.py` 会按 EEG 和 fNIRS 通道规模自动分配

## 推荐使用方式

### 1. 先做小规模调试

```bash
python all.py --scripts eeg fnirs fusion --files_limit 1 --batch_sizes 32 --epochs_list 10
```

### 2. 做完整串行实验

```bash
python all.py --scripts eeg fnirs fusion --batch_sizes 32 --epochs_list 100
```

### 3. 做并行实验

```bash
python all.py \
  --scripts eeg fnirs fusion \
  --parallel \
  --gpus 0 1 2 \
  --batch_sizes 32 \
  --epochs_list 100
```

### 4. 让 fusion 使用专属模型

```bash
python all.py \
  --scripts eeg fnirs fusion \
  --model_eeg temporal_se \
  --model_fnirs temporal_se \
  --model_fusion fusion_temporal_se \
  --top_k_eeg 32 \
  --top_k_fnirs 20
```

## 注意事项

- `all.py` 只是调度器，不负责具体训练逻辑
- 真正训练仍在 `eeg.py`、`fnirs.py`、`fusion.py` 中完成
- 如果某个子脚本退出码非 0，`all.py` 会抛出异常并停止当前批次
- 并行模式下，`--gpus` 数量必须不少于 `--scripts` 数量
- 若不显式传 `batch_sizes` 或 `epochs_list`，子脚本会使用 `config.yaml` 默认值
- 若只给 fusion 指定 `--top_k_eeg` 或 `--top_k_fnirs` 其中一个，fusion 脚本会报错，必须成对提供

## 一个完整示例

```bash
python all.py \
  --scripts eeg fnirs fusion \
  --batch_sizes 32 \
  --epochs_list 100 \
  --output_root runs \
  --files_limit 2 \
  --model_eeg shallow \
  --model_fnirs temporal_se \
  --model_fusion fusion_temporal_se \
  --top_k_eeg 32 \
  --top_k_fnirs 20
```

这条命令的效果是：

- 运行 EEG、fNIRS、fusion 三个脚本
- 本轮实验使用 `batch_size=32`、`epochs=100`
- 每个脚本最多处理 2 个文件
- EEG 使用 `shallow`
- fNIRS 使用 `temporal_se`
- fusion 使用 `fusion_temporal_se`
- fusion 中显式保留 32 个 EEG 通道和 20 个 fNIRS 通道
- 输出保存到 `runs/bs32_ep100/` 下对应子目录
