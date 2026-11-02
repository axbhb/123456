# 自动驾驶生成数据训练污染实验

四个生成器（DriveDreamer-2、FrozenDrive、MagicDrive、Panacea+）的 Clean 与 APGD、AdvDiff、Asymmetry、BAIT、PIL 任务适配代码、运行源码、部署记录及新服务器提示词。

**不含模型权重，不含本项目 mini 或完整 nuScenes 实验数据集，不上传 ZIP。**此前 20/20 组合完成的是 mini 端到端验证；全量实验尚未执行。

## 目录

- code/five_methods/：主要方法、数据注册、批量运行和指标汇总入口。
- runtime/：原生生成器、检测框架、第三方源码与模型配置，保留部署相对路径。
- models/：待另行准备的权重清单，没有实际张量权重。
- prompts/：新服务器 Codex 提示词，提供 Markdown 与 Word。
- environment/：原服务器实测依赖版本。
- results/：历史 mini 指标、耗时、审计和文本日志。
- protocol/：原始实验设置文档。
- METHODS_zh.txt 与 VALIDATION_zh.txt：适配方法和已验证范围。
- README_zh.txt 与 HANDOFF_FOR_CODEX_zh.txt：详细部署和交接说明。

## 新服务器部署

先读取 [完整提示词](prompts/CODEX_FULL_EXPERIMENT_zh.md)。目标服务器的 GPU、系统和 CUDA 尚未确定，需对照实际版本调整环境。

原环境：V100 32GB、Ubuntu 18.04、GCC 7.5；检测及 FrozenDrive/MagicDrive 使用 Python 3.9.15、PyTorch 1.11.0+cu113、MMCV-full 1.7.0、mmdet 2.28.2、mmdet3d 1.0.0rc6；DD2/Panacea 生成使用 Python 3.9.16、PyTorch 2.2.2+cu121。具体版本及源码优先顺序见 environment 与运行记录。没有复制 Conda 二进制环境。

    git clone https://github.com/axbhb/123456.git
    cd 123456
    export ROOT=/srv/table2
    mkdir -p "$ROOT" "$ROOT/table2_portable_20261101"
    cp -a runtime/. "$ROOT/"
    cp -a code/five_methods/. "$ROOT/table2_portable_20261101/"
    python3 relocate_bundle.py --root "$ROOT" --skip-pickles

然后按详细说明构建环境，将合法取得的权重放回 models/weights_required.json 所列路径（相对于 ROOT），另行准备完整 nuScenes trainval、地图、原生 infos、四套全量 Clean 图像及生成条件。不能直接沿用历史 mini 默认路径。

## 本轮全量实验要求

对相同生成器、攻击配置和攻击 seed，只生成一次 20% 中毒池；5% 和 10% 分别取其中嵌套的前四分之一和前二分之一，20% 使用全部。三个比例分别微调和评测，共享生成成本只计一次。

**现有 run_experiment.py 仍按各比例分别准备攻击数据。20% 池复用是本轮新增要求，尚未在这份代码中实现。目标服务器 Codex 需要先依据提示词调整入口、验证嵌套集合和缓存复用，再展开正式实验。**

所有 baseline 保留适配版本名称，不是原论文完整复现。历史 DD2 使用 14 个六视角关键帧；其余生成器各方法使用 1 帧替换验证运行性。运行通过不代表攻击必然有效。

## 第三方来源

保留各第三方目录已有 LICENSE、NOTICE 和来源记录。各组件遵循原有许可；本仓库不为第三方代码或模型授予额外权利。模型权重须按原发布渠道及相应条款另行获取。

第三方源码保留其自带的小型测试样本和演示文件；它们不是本项目训练/评估数据集。

## 精简 mini 数据与条件重建

另有按用户要求制作的404帧精简数据包，省略大投影/几何缓存和原始LiDAR，保留四套Clean图像、原始相机/历史图像、标注/标定和小型配对记录。原始LiDAR及lidarseg需在目标服务器另行取得。重建几何还需额外的NKSR ks.pth（另附第14个权重包），详情见 [重建说明](runtime/table2_clean_generators_20261030/frozendrive_conditions/README_REBUILD_zh.md)。

2026-11-02修复原生条件入口缺少参数初始化的问题，并实际CPU重建一个DD2片段（29.7秒通过）。该修复未改变攻击或训练算法。
