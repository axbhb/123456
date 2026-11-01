Table 2 全量实验 GitHub 仓库说明（无权重 无数据集）

用途
在新的 Linux GPU 服务器上部署并继续运行 FrozenDrive、DriveDreamer-2、MagicDrive、Panacea+ 的 Clean、APGD、AdvDiff、Asymmetry、BAIT、PIL 实验。主运行入口不列出 Ours。所有 baseline 均为已说明的任务适配版本，不能称为原论文完整复现。
最终完成状态以 results/validation_matrix.csv 和 VALIDATION_zh.txt 为准。

数据范围
按用户要求移除了 mini 原始数据、四套 mini 合成训练图、攻击图、infos、片段计划和条件/几何缓存。本包不含训练或评估数据集，完整 nuScenes trainval 和对应全量生成条件须在目标服务器另行准备。
保留 20 个组合已通过 mini 验证的历史指标、审计和文本日志；这些是历史证据，不是可直接运行的 mini 数据。

文件布局
以 README.md 的展开目录为准。runtime/ 是源码与配置；code/five_methods/ 是主入口；models/ 是权重准备清单；prompts/ 是最新执行要求。没有资源压缩包和实际模型权重。

本次实际环境
GPU：两张 Tesla V100-PCIE-32GB；每个受控任务只用一张卡。
系统：Ubuntu 18.04，Linux 5.4.0-51，驱动 535.161.08，GCC 7.5.0。
检测、APGD/BAIT/PIL/Asymmetry、FrozenDrive/MagicDrive：Python 3.9.15，PyTorch 1.11.0+cu113，torchvision 0.12.0+cu113，MMCV-full 1.7.0，mmdet 2.28.2，mmdet3d 1.0.0rc6。
DD2/Panacea 生成：Python 3.9.16，PyTorch 2.2.2+cu121，torchvision 0.17.2。
Torch 的编译 CUDA 版本分别为 11.3 和 12.1；非交互 PATH 未找到 nvcc。不要把 Torch 编译版本直接当作系统 CUDA Toolkit 版本。
MagicDrive 使用随源码提供的 third_party/diffusers；DD2 使用随包 import_overlay 与 dreamer-models。pip 安装元数据不是全部运行上下文，源码优先顺序必须保留。详见 measured_versions.json 和运行时模块记录。
目标服务器的 GPU、系统和 CUDA 尚未确定。按用户要求，由目标服务器的 Codex 根据这些实测版本调整环境；本包不声称已验证其他硬件。

部署
先按 README.md 克隆仓库，将 runtime/. 复制到 ROOT，并将 code/five_methods/. 复制到 ROOT/table2_portable_20261101。执行 relocate_bundle.py --root ROOT --skip-pickles，按 models/weights_required.json 另行准备权重。

下文的 python 命令均指相应已激活的环境；检测、数据注册和核验使用检测环境。
随后按 environment/ 的实际版本与随包源码构建下列环境目录：
  ROOT/envs/streampetr_torch111_cu113/bin/python
  ROOT/envs/frozendrive/bin/python              # 同时用于 MagicDrive
  ROOT/envs/dd2_gpu/bin/python
  ROOT/envs/panacea_plus/bin/python
nksr_reconstruction 用于从原始点云重建新的 FrozenDrive 条件；本包不再含 mini 的已有条件。
*_core_versions.txt 和 *_installed_versions.txt 为版本信息；不要照抄原机器 @ file 的安装路径。重建 MMCV、mmdet3d、xformers 等扩展时按目标 GPU 架构和 Torch/CUDA 重新编译。保留所附源码中的兼容修改。
完成检测环境后重定位 pickle 元数据：

  /srv/table2/envs/streampetr_torch111_cu113/bin/python relocate_bundle.py --root /srv/table2

重定位只改路径文本及 infos 元数据，不转换模型张量权重。环境没有随 ZIP 复制为二进制 conda 环境。

运行正式实验
本包已移除 mini。不要直接使用脚本历史 mini 默认路径；必须显式传入下文的全量参数。新服务器先从真实全量数据选择少量帧复查兼容性，再展开正式矩阵。
设置 export TABLE2_USE_EXISTING_GUARD=0 以使用新服务器直接运行模式；不复制朋友服务器授权或记账文件。
Clean 与攻击须使用相同训练 seed、初始权重、增强和预算；Clean 每个生成器/训练 seed 运行一次。iterations 是计划迭代数，实际 optimizer 更新数另记。

全量数据准备
1. 另行准备完整 nuScenes trainval、地图、相机数据及原生 StreamPETR 2D/ego/temporal infos。真实评估数据需有标注。按完整场景划分并保留连续关键帧，训练与测试场景不重叠。
2. 用 prepare_full_dataset.py 注册训练、测试 infos，生成适用于该版本的检测配置；make_full_plan.py 生成 DD2/MagicDrive/Panacea 原生片段计划。
3. 按原生生成器流程准备全量 Clean 图像和原生条件。register_full_clean.py 可将 token/camera 格式的图像注册成 Clean 元数据；默认仿射矩阵来自 generator_layouts.json，若改变生成分辨率/裁剪则显式提供 --matrix。
4. FrozenDrive 还需要与全量数据相符的 native_clean_config.yaml、native_train_infos.pkl、native_generation_pairs.json、训练 occupancy/投影和 Clean 前序帧。只有 JPEG 图像不足以运行 AdvDiff/Asymmetry。原生数据和条件准备源码已附，目标服务器 Codex 需用实际全量路径配置，不能用 mini 条件冒充全量。
5. 在 run_experiment.py 显式设置 --dataset、--raw-dataset、--nuscenes-version v1.0-trainval、--clean-root、--plan-dir、--frozen-reference-clean、--iterations 和新的 --output。--clean-root 下按四个生成器名称建目录。

全量调用形态：
  python run_experiment.py --root ROOT --generators dd2 --methods clean apgd advdiff asymmetry bait pil --dataset ROOT/datasets/full_camera --raw-dataset /data/nuscenes --nuscenes-version v1.0-trainval --clean-root ROOT/outputs/full_clean --plan-dir ROOT/plans/full --frozen-reference-clean ROOT/outputs/full_clean/frozendrive --ratios .05 .10 .20 --seeds 0 1 2 --iterations 实际计划迭代数 --gpu 0 --output ROOT/outputs/full_experiments

全量迭代预算由实际实验设置决定，没有自动沿用 283。更换数据或方法设置时使用新的 Clean 和 baseline 输出目录。若需要场景级替换，另行实现并标注选择口径，不要把本入口的帧级比例写成场景比例。
AdvDiff 当前独立重采样每个选中关键帧所在的原生片段；所需计算显著多于图像扰动。BAIT/PIL 当前将训练 ROI 裁剪载入 CPU 内存。全量部署时按实际资源调整数据读取/调度，同时保持算法和配对配置的含义。

指标与核验
  python collect_results_portable.py --output 实验输出根目录
  TABLE2_ROOT=ROOT TABLE2_GENERATOR=dd2 TABLE2_CLEAN=对应Clean目录 TABLE2_OUTPUT=某个n/seed/iter输出目录 TABLE2_REPLACEMENT_COUNT=实际帧数 python audit_replacement.py apgd

collect_results_portable.py 另外输出 results_by_seed_group.csv/json，按相同设置汇总训练 seed，给出均值与样本标准差；只有一个 seed 时标准差留空。保留逐 seed 配对 ASR 的均值，以及明确命名的 ASR_from_means，两种口径不能混写。需要一并汇总 Clean 时传入 --include-clean-root。

每组保留 attack_status.json、pipeline_status.json、summary.json、training_audit.json、after/result.json 和原生各类别 AP/TP errors。summary 中的阶段耗时为进程墙钟时间；朋友服务器的受控记账排除锁等待，包含作业内 CPU 计算，不是 GPU 核函数活跃时间。
本次交叉验证临时训练检查点在评估、审计后为腾出空间清理；审计、指标和日志保留。生成器权重及初始 StreamPETR 权重均不在仓库中，需另行准备，正式实验入口默认保留新训练检查点。
mini 单种子结果只能作流程和初步效果观察；不能据小幅升降判断统计显著性。所有生成器原生条件和标注保持，不等于生成物体与标注的语义一致性得到形式保证。
