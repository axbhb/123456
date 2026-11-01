# 模型权重需另行准备

按用户要求不上传任何模型权重。配置、tokenizer 等非权重文件保留在 runtime/assets 中。weights_required.json 记录排除路径和大小，路径相对于部署 ROOT。结合 runtime 中各原生项目 README 和许可证，从原发布渠道取得所需权重；重复别名可复用。

不要用随机初始化权重冒充原设置。
