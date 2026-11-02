# 重建已省略的 FrozenDrive 和 Asymmetry 几何条件

这些文件是原服务器实际采用的 mini 几何构建脚本与配置，此前迁移包只携带已构建缓存，故没有包含这批脚本。现在精简数据不含大缓存，也不含原始 LiDAR，需在新服务器准备同源数据后重建。

## 必须准备的原始输入

1. 与283训练关键帧及其场景一致的nuScenes v1.0-mini LiDAR关键帧、必要sweeps和原始lidarseg语义标签。图像、标注和相机元数据不能代替实测点云。
2. NKSR预训练检查点ks.pth，放在ROOT/table2_clean_generators_20261030/frozendrive_conditions/checkpoints/ks.pth。该文件未上传GitHub，另见14_nksr_ks.zip。
3. 重建nksr/chamferdist/mmcv/open3d环境及扩展，使用environment中原版本作参考，按目标GPU和Torch/CUDA编译。
4. 原Occupancy投影源码在ROOT/frozendrive_deployment_20261029/source/Occupancy，可将本目录的projection_source链接到该目录。

## 原流程顺序

先检查输入和编译扩展，再从generate_occ_mini.py生成带法线和语义点的prepared.npz；reconstruct_prepared.py使用config800.yaml和NKSR生成稠密占据；project_uniscene_mini.py执行六相机投影；validate_condition_pilot.py检查几何、标签及相机结果。先完成第一帧pilot，再按prepare_all_conditions.py处理需要的训练帧。

保持voxel_size=0.125、800×800×64范围、NKSR detail_level=0.5、mise_iter=1、类别映射、原标定及投影规则。不要把不同来源的第三方GT占据直接称为原实验的NKSR重建结果。

run_nksr_pilot.py保留了原有编译/输入验证记录检查与资源协调锁，且按mini的283/121划分编写。新服务器需真正执行对应编译和数据验证，生成本机记录；不要复制旧机器的passed JSON冒充检查，也不要直接删除检查。源目录nksr_source、checkpoint及编译中间产物不在GitHub中，应按实际环境准备。脚本为原流程参考，不代表已在目标硬件上重跑验证。

## 与原生片段条件入口的区别

code/five_methods/prepare_conditions.py负责DD2、MagicDrive和Panacea的地图/框等原生条件，与NKSR检查点无关。已修复其参数与builder初始化，并在原服务器实际CPU重建DD2一个原生片段，约29.7秒通过。其他两种builder的本次新增迁移路径没有据此声称完成新硬件验证。
