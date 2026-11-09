# Memory & Codebase Knowledge Base: DVS Gesture Recognition

## 1. Executive Summary & Purpose
This repository explores, benchmarks, and analyzes different representations of event-based asynchronous neuromorphic data using the **IBM DVS128 Gesture Dataset** (`tonic.datasets.DVSGesture`, 11 gesture classes). 

The core research objective is understanding how different event stream discretization/encoding schemes affect the accuracy, computational cost, and convergence across three model paradigms:
1. **3D Convolutional Neural Networks (CNNs)**
2. **3D Residual Networks (ResNets)**
3. **3D Spiking Neural Networks (SNNs)** via `snntorch`

---

## 2. Dataset & Event Stream Representations

### Dataset
- **IBM DVS128 Gesture**: Recorded with an iniVation DVS128 event camera (128x128 resolution, 2 polarities ON/OFF).
- 11 action classes: hand clapping, right hand wave, left hand wave, right arm clockwise, right arm counter-clockwise, left arm clockwise, left arm counter-clockwise, arm roll, air drums, air guitar, other.
- Handled via the `tonic` neuromorphic library.

### Representations Implemented
1. `n_bins`: Discretizes the event duration into a constant number of temporal bins (default: `n_time_bins=100`).
2. `binary`: Discretizes into `n_time_bins=200` followed by binary temporal bit representation (`TT.ToBinaRep(n_frames=100, n_bits=2)`).
3. `time_window`: Aggregates events into fixed temporal duration slices (default: `time_window=10000` µs = 10 ms). Yields variable sequence lengths.
4. `spike_count`: Aggregates events by fixed event count slices (default: `event_count=1000` events per frame). Yields variable sequence lengths.
5. `timesurface`: Computes exponential decay of past event timestamps (`tau=30000`, `dt=10000`). Yields variable sequence lengths.

### Data Handling & Caching (`dataset/`, `utils/`, `cache_dataset.py`)
- **`dataset/dataset.py`**:
  - `DVSGestureData`: On-the-fly dataset wrapper transposing event arrays into shape `(C, T, H, W)` with PyTorch float32 tensors.
  - `CachedDVSGestureData`: Loads pre-computed sample `.pt` files from `./cache/<representation>/[train|test]/sample_<idx>.pt` to bypass costly on-the-fly transforms and prevent OOM.
- **`cache_dataset.py`**: Script to offline-cache the entire dataset for a chosen representation.
- **`compute_frame_count_stats.py`**: Calculates running mean, min, max, and histograms of frame counts across samples for variable-length representations.
- **`utils/pad_tensors.py`**:
  - `PadTensors`: Simple zero-padding along temporal dimension $T$ up to batch max.
  - `PadTensorsUpdated`: Normalizes temporal sequence length to a fixed `target_frames` parameter using zero-padding (if $T < \text{target}$) or downsampling/pooling (`max`, `avg`, `linear` trilinear interpolation if $T > \text{target}$).

---

## 3. Architecture & Model Definitions (`models/`)

### 1. Base 3D CNN (`models/cnn/model.py`)
- Class: `Gesture3DConvNet`
- Structure:
  - 3 Conv3D blocks: Conv3D(kernel=3, padding=1) -> ReLU -> MaxPool3D(2).
  - Channel progression: $2 \rightarrow 8 \rightarrow 16 \rightarrow 32$.
  - Dynamically infers flattened feature size via dummy input.
  - Classifier: Linear(flattened, 128) -> ReLU -> Linear(128, 11).

### 2. Scalable 3D CNN (`models/cnn/new_model.py`)
- Class: `Gesture3DConvNet_v2`
- Structure:
  - Configurable number of conv layers (`nr_conv_layers`), channels doubling each layer ($2 \rightarrow 4 \rightarrow 8 \rightarrow \dots \rightarrow 2^{\text{layers}+1}$).
  - Fixed spatial-temporal bottleneck using `nn.AdaptiveAvgPool3d((12, 16, 16))`.
  - Classifier: Linear($2^{\text{layers}+1} \times 12 \times 16 \times 16, 128$) -> ReLU -> Linear(128, 11).

### 3. 3D ResNet (`models/resnet/resnet.py`)
- Class: `ResNet3D` and `BasicBlock3D`
- Structure:
  - Stem: Conv3D(2 -> base_channels=16, kernel=3, padding=1) -> BatchNorm3d -> ReLU.
  - Residual blocks: `num_blocks` sequential 3D residual blocks with channel doubling ($16 \rightarrow 32 \rightarrow 64 \dots$) and $1\times 1 \times 1$ conv projection shortcuts.
  - Global temporal-spatial pooling: `nn.AdaptiveAvgPool3d((1, 1, 1))`.
  - Linear classifier directly from pooled channels to classes.

### 4. 3D Spiking Neural Network (`models/snn/snn_model.py`)
- Class: `Gesture3DCSNN` built using `snntorch`
- Structure:
  - 3 Conv3D feature extraction blocks: Conv3D -> `snn.Leaky(beta=beta, spike_grad=atan)` -> MaxPool3D(2).
  - Channels: $2 \rightarrow 8 \rightarrow 16 \rightarrow 32$.
  - Spatial-temporal pooling: `nn.AdaptiveAvgPool3d((12, 16, 16))`.
  - Permutes dimensions to `[time, batch, channels, height, width]`.
  - Classifier evaluated per time step: Flatten -> Linear($32 \times 16 \times 16, 128$) -> `snn.Leaky` -> Linear(128, 11) -> `snn.Leaky(output=True)`.
  - State reset (`utils.reset`) between passes.
  - Returns `spike_rec`, `mem_rec`, and softmax output probabilities over integrated spikes.

### 5. Transformer Stub (`models/transformer/`)
- Currently an empty placeholder (`__init__.py`).

---

## 4. Training Pipelines & Scripts

### Common Training Characteristics
- Framework: **PyTorch Lightning** (`lightning.pytorch`)
- Precision: 16-bit mixed precision (`precision="16-mixed"`)
- Optimizer: Adam (`lr=1e-3`)
- Callbacks: `EarlyStopping(monitor="val_loss", patience=2, mode="min")`
- Logging: `TensorBoardLogger` with epoch-level confusion matrix heatmaps saved to TensorBoard.
- Evaluation metrics: `train_acc`, `val_acc`, `test_acc`, confusion matrix via `torchmetrics`.

### Script Overview
- **`cnn_script.py`**:
  - Trains base `Gesture3DConvNet`.
  - Supports only uniform representations (`n_bins`, `binary`).
- **`new_cnn_script.py`**:
  - Trains `Gesture3DConvNet_v2`.
  - Supports all representations (`n_bins`, `binary`, `time_window`, `spike_count`, `timesurface`).
  - Integrates `PadTensorsUpdated` with target frame counts and pooling modes (`max`, `avg`, `linear`).
- **`resnet_script.py` / `resnet_script_nocache.py`**:
  - Trains `ResNet3D` using cached data (`resnet_script.py`) or raw tonic dataset (`resnet_script_nocache.py`).
  - Supports residual depth tuning (`--num-blocks`), target frame counts, and pooling modes.
- **`snn_script.py`**:
  - Trains `Gesture3DCSNN`.
  - Configurable surrogate decay factor `beta`.
  - Configurable loss objectives via `snntorch.functional`:
    - `count`: `ce_count_loss`
    - `latency`: `ce_temporal_loss`
    - `frequency`: `accuracy_rate`
    - `max_mem`: `ce_max_membrane_loss`
  - High GPU VRAM usage due to LIF neuron temporal rollouts; requires small batch size (e.g., 8).

---

## 5. Experiment Automation & Batch Execution

### Script Suites
- **`run_scripts/`**:
  - Shell automation for grid searches over seeds (`42, 13, 93, 45, 96, 6, 98, 59, 44`), pooling strategies (`max`, `avg`, `linear`), target frame counts (`25` to `800`), training set fractions (`0.1` to `0.9`), and SNN loss/beta variations.
  - Includes checkpoint-skipping logic checking for existing results in destination CSVs before initiating training runs.
- **`collect_scripts/`**: Shell scripts compiling experiment outputs and status.
- **`oar_run.sh`**: Wrapper script for HPC cluster job scheduling (OAR).

### Output Directories & Analysis
- **`results/`**: Log files from grid sweeps across CNN, ResNet, and SNN configurations.
- **`results_with_target_fc/`**: CSV files and plots comparing accuracies across target frame count settings and pooling modes.
- **`interpreted_results/`**: Jupyter notebooks analyzing SNN loss functions (latency, frequency, count, max membrane potential) and CNN batch sizes.
- **`frame_count_stats/`**: Notebooks evaluating distribution of frame counts for variable-length representations.

---

## 6. Key Developer Conventions & Caveats

1. **VRAM Constraints with SNN**: `Gesture3DCSNN` stores temporal unrolls of membrane states and spikes. Keep batch size $\le 8$ on typical GPUs.
2. **Device Variable Syntax**: Note differences across scripts: `cnn_script.py` uses `"gpu"` (Lightning translates this), while `snn_script.py` uses `"cuda"`.
3. **Cluster Hardcoding**: Notice `oar_run.sh` contains hardcoded user paths (`/home/rosvath/...`) that should be generalized for local execution.
4. **Caching Workflow**: For non-uniform representations (`time_window`, `spike_count`, `timesurface`), running `cache_dataset.py` first is strongly recommended to avoid DataLoader bottlenecks.
5. **Collate Function Requirement**: Any model training on variable-length representations must use `PadTensorsUpdated` with an explicit `--batch-fc` / `--desired_fc` argument.

---

## 7. Research Paper Context & Core Hypotheses

- **Paper Title**: *"Impact of frame-based representations of event-based data for gesture recognition"* (Róbert-Levente Osváth, Babeș-Bolyai University, in collaboration with CRIStAL / FOX research team, University of Lille; short paper presented at NCDL 2025 `hal-05330405`; experiments executed on **Grid'5000 / OAR** infrastructure).
- **Academic Scope & Pragmatic Constraints**:
  - Continuation conducted as a **5-credit Special Assignment in Computer Science at Aalto University**.
  - **Objective**: Achieve a clean, defensible deliverable covering the Project Plan milestones with **minimum engineering overhead and minimum Grid'5000 runtime**:
    1. Document the fixed-frame-count pooling results (`target_fc` sweep for memory-performance trade-off; omit the 10%–90% learning curve sweeps for the new dataset).
    2. Generalize findings to **DailyAction-DVS** as the primary second dataset (see rationale below).
    3. Test lightweight event-level preprocessing (event-level spatial/temporal downsampling & unsupervised denoising filter via `tonic.transforms.Denoise` / `Downsample` / `DropEvent`).
- **Established Findings on DVS128 Gesture**:
  1. **Dynamic vs. Fixed Frame Counts**: Dynamic frame count representations (`time_window`, `spike_count`, `timesurface`) consistently outperform fixed frame count representations (`n_bins`, `binary`), especially when training on small subsets of data (10%–50%), showing steeper learning curves.
  2. **Temporal Pooling Trade-off**: Applying 3D adaptive pooling (`avg`, `linear`, `max`) via `PadTensorsUpdated` to bound dynamic representations to a fixed `target_frames` (e.g., 150–600 frames) matches or exceeds unpooled baselines (up to 94% on individual runs, ~88–89% mean) while bounding GPU VRAM.
- **Core Hypothesis to Validate Next (Section VI Discussion)**:
  - On DVS128 Gesture, `time_window` performed best overall because recordings have a nearly uniform duration (~6s) and steady, continuous cyclic arm/hand motion.
  - **Hypothesis**: On datasets with **variable recording durations** or **chaotic/bursty motion** (short bursts of intense movement separated by inactive periods), `spike_count` (constant event count slicing) should outperform `time_window` and fixed-bin representations.

---

## 8. Candidate Datasets Evaluation (Next Research Phase)

None of the three candidate datasets are built into `tonic.datasets` (`tonic==1.6.0`), so a custom PyTorch `Dataset` wrapper converting raw files into Tonic's structured NumPy array (`dtype=[('x', int16), ('y', int16), ('t', int64), ('p', int8)]`) is required before passing to `tonic.transforms`.

### Comparative Overview

| Feature | **DVS128 Gesture** (Current) | **1. DailyAction-DVS** (Easiest Drop-In) | **2. THU-EACT-50-CHL** (Best `.npy` Benchmark) | **3. THU-EACT-50** (Standard) | **4. DailyDVS-200** (Largest Scale) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Publication** | CVPR 2017 (Amir et al.) | IJCAI 2021 (Liu et al.) | IEEE TPAMI 2023 (Gao et al.) | IEEE TPAMI 2023 (Gao et al.) | ECCV 2024 (Wang et al.) |
| **Sensor** | iniVation DVS128 | iniVation DVS128 | iniVation DAVIS346 | CeleX-V | DVXplorer Lite |
| **Spatial Resolution** | $128 \times 128$ | **$128 \times 128$** (Exact match) | $346 \times 260$ | $1280 \times 800$ | $320 \times 240$ |
| **Classes (`num_classes`)** | 11 | **12** | 50 | 50 | 200 |
| **Total Recordings** | 1,342 | **1,440** | 2,330 | 10,500 | 22,046 |
| **Subjects** | 29 (23 train / 6 test) | 15 (80/20 random split) | 18 (14 train / 4 test) | 105 (85 train / 20 test) | 47 (32 train / 8 val / 9 test) |
| **Raw File Format** | `.aedat` / Tonic `.npy` | `.aedat` (AEDAT 2.0) | **`.npy` (`[x, y, t, p]`)** | `.csv` (`[y, x, b, p, t]`) | `.aedat4` (DV container) |
| **Decompressed Size** | ~1.5 GB | **~1 GB** | **~4.6 GB** | ~332 GB (up to 1.1 TB) | ~50–100+ GB |
| **Download Hosting** | Tonic auto-download | Google Drive | Google Drive & Baidu | OneDrive & Baidu | Google Drive & Baidu |
| **Repo Link** | Built into `tonic` | `qianhuiliu/SNN-action-recognition` | `lujiaxuan0520/THU-EACT-50` | `lujiaxuan0520/THU-EACT-50` | `QiWang233/DailyDVS-200` |

### Detailed Assessment for Our Pipeline & Grid'5000

#### 1. **DailyAction-DVS** — *Easiest Drop-In Replacement & Fastest Execution*
- **Why it fits**:
  - Uses the **exact same $128 \times 128$ resolution**, 2 polarities, similar sample count (1,440 vs. 1,342), and nearly identical class count (12 vs. 11: *bend, climb, fall down, get up, jump, lie down, carry box, run, sit down, stand up, walk, pick up*).
  - Clip durations vary between **1 and 6 seconds** and include abrupt, bursty whole-body motions (*fall down, jump, sit down*) alongside continuous motions (*walk, run*), directly testing the paper's `spike_count` vs. `time_window` hypothesis.
  - Fits easily within Grid'5000 `/home` storage quotas and runs a full 9-seed sweep in the same walltime as DVS128 Gesture.
- **Implementation notes**:
  - Requires parsing AEDAT 2.0 files once during caching (or via `tonic.io.read_aedat` / `aedat`).
  - Uses an 80%/20% stratified split (no official subject split file), so seed-fixed splitting must be done prior to training.

#### 2. **THU-EACT-50-CHL (Challenging Subset)** — *Easiest Parser (`.npy`) & Strongest Benchmark*
- **Why it fits**:
  - While the standard `THU-EACT-50` is **332 GB of `.csv` files at $1280 \times 800$** (impractical for Grid'5000 quotas and 3D CNN VRAM without heavy downsampling), the official **`THU-EACT-50-CHL`** subset is **only 4.6 GB** and already pre-packaged as **`.npy` NumPy arrays** of shape `(N, 4)` (`[x, y, t, p]`, $p \in \{0, 1\}$).
  - Requires **zero external C++/Rust binary decoders** (`np.load()` only!).
  - Includes official subject-disjoint `train.txt` (14 subjects) and `test.txt` (4 subjects) across 50 diverse classes (whole-body, fine-grained, and two-player actions) with varying action speeds/magnitudes.
- **Implementation notes**:
  - Sensor size is `(346, 260, 2)`. Since `Gesture3DConvNet_v2`, `ResNet3D`, and `Gesture3DCSNN` use `AdaptiveAvgPool3d`, only `num_classes=50` needs to change in the model head; optionally prepend `TT.Downsample(spatial_factor=0.5)` to match $128 \times 128$ VRAM usage.
  - Note: Check timestamp units in `.npy` files (microseconds vs. normalized) before applying `time_window=10000` or `timesurface`.

#### 3. **DailyDVS-200** — *Most Comprehensive (Attribute-Annotated), Harder at Full Scale on Grid'5000*
- **Why it fits**:
  - Modern ECCV 2024 benchmark ($320 \times 240$, 200 classes, 22,046 clips) with **14 fine-grained attribute annotations** in `all_data.json` (including `"Complexity of Movement"`, `"Range of Motion"`, `"Move"` camera motion, `"Background Complexity"`, and `"Illumination"`).
  - Those 14 attributes allow directly isolating chaotic/complex movements vs. simple movements to prove the paper's hypothesis!
- **Challenges on Grid'5000**:
  - Requires `aedat` (`pip install aedat`, currently commented out in `requirements.txt`) or `dv` to decode `.aedat4` files.
  - 22,046 recordings is $16.4\times$ larger than DVS128 Gesture; caching all 5 representations generates $>110,000$ `.pt` files, which can exceed Grid'5000 NFS disk/inode quotas and OAR job walltime limits unless placed on node-local scratch (`/tmp`) or filtered to a targeted subset (e.g., 20–30 classes or specific movement-complexity attributes).

