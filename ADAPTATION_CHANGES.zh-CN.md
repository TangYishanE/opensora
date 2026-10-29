# 本地适配改动盘点

本文记录 OpenSora 工作区为 RoboTwin 2.0 和 RoboDojo 增加的数据、训练和评估接口，以及截至 2026-10-29 的验证边界，供后续审查和服务器部署参考。它是当前未提交工作区的快照；具体参数和命令以代码及两个任务文档为准。

盘点基线：分支 `feature/robodojo-adaptation`，HEAD `a50d1d1ced6d`。编写本文前有 5 个已跟踪文件修改、26 个未跟踪文件；本文本身也是新增文件。

## 处理流程与模块

适配流程从原始轨迹和来源清单开始，转换图像与动作、生成仅使用训练集统计量的清单，再读取时序窗口并训练动作条件视频模型，最后进行视频预测指标评估。下列模块分别承担这些步骤。

| 文件 | 修改内容与作用 |
| --- | --- |
| [opensora/robotwin_dataset.py](opensora/robotwin_dataset.py) | 增加 RoboTwin 轨迹窗口读取、图像和动作处理，供训练及评估复用。 |
| [opensora/robodojo_dataset.py](opensora/robodojo_dataset.py) | 在共用窗口接口上检查 RoboDojo 数据集标识，读取对应轨迹。 |
| [scripts/convert_robotwin.py](scripts/convert_robotwin.py)、[scripts/convert_robodojo.py](scripts/convert_robodojo.py) | 转换两个来源的轨迹、校验数据及划分，并为后续训练准备清单和统计量。RoboDojo 转换需要明确动作语义、时间对齐、图像解码和关节顺序约定。 |
| [opensora/action_stats.py](opensora/action_stats.py) | 计算及使用训练集动作分位数，检查数值和统计元数据，避免将评估集信息用于训练归一化。 |
| [opensora/action_encoder.py](opensora/action_encoder.py)、[opensora/models/stdit/stdit3.py](opensora/models/stdit/stdit3.py) | 将动作编码器独立出来并接入 STDiT；校验动作输入，针对动作嵌入层尺寸变化处理旧 checkpoint，其他不匹配仍需明确检查。 |
| [scripts/train_robotwin.py](scripts/train_robotwin.py)、[scripts/train_robodojo.py](scripts/train_robodojo.py) | 增加两个任务的训练入口；RoboDojo 复用训练流程。当前入口面向单 GPU 训练，保存模型和训练记录，不等于已完成目标域训练或支持完整断点续训。 |
| [opensora/robotwin_evaluation.py](opensora/robotwin_evaluation.py)、[scripts/evaluate_robotwin.py](scripts/evaluate_robotwin.py)、[scripts/evaluate_robodojo.py](scripts/evaluate_robodojo.py) | 增加离线视频预测评估及两个任务的入口；视频指标不能作为物理任务成功率。 |

RoboDojo 样本中使用的 `state_target` 表示记录的关节目标；尚未确认它与真实执行命令完全等价。动作顺序、帧率和图像颜色约定应随来源清单保存，部署时应与目标环境再次核对。

## 配置、测试与其他文件

| 文件 | 修改内容与作用 |
| --- | --- |
| [config/robotwin_source_manifest.example.json](config/robotwin_source_manifest.example.json)、[config/robodojo_source_manifest.example.json](config/robodojo_source_manifest.example.json) | 两个数据来源清单示例，明确输入和转换所需元数据。 |
| [config/train_robotwin_place_empty_cup.yaml](config/train_robotwin_place_empty_cup.yaml)、[config/train_robodojo_put_bottles_into_dustbin.yaml](config/train_robodojo_put_bottles_into_dustbin.yaml) | 首轮单任务训练配置。路径、数据和权重需要按目标机器准备。 |
| [tests/test_action_encoder.py](tests/test_action_encoder.py)、[tests/test_action_stats.py](tests/test_action_stats.py) | 动作编码与归一化相关单元测试。 |
| [tests/test_robotwin_dataset.py](tests/test_robotwin_dataset.py)、[tests/test_robodojo_dataset.py](tests/test_robodojo_dataset.py)、[tests/test_robotwin_training.py](tests/test_robotwin_training.py) | 数据转换、数据集和训练通路测试；部分测试使用替身或小模型。 |
| [requirements/requirements-robotwin-test.txt](requirements/requirements-robotwin-test.txt)、[requirements/requirements-robodojo-test.txt](requirements/requirements-robodojo-test.txt) | 两套测试依赖清单。 |
| [docker/robotwin-unit.Dockerfile](docker/robotwin-unit.Dockerfile)、[docker/robodojo-unit.Dockerfile](docker/robodojo-unit.Dockerfile) | 已有 CPU 单测镜像配置；不代表完整 GPU 训练或物理仿真环境。 |
| [docs/robotwin.md](docs/robotwin.md)、[docs/robodojo.md](docs/robodojo.md)、[README.md](README.md) | 增加任务相关使用说明和入口。 |
| [.gitignore](.gitignore)、[pyproject.toml](pyproject.toml) | 调整需要纳入版本管理的文档、测试文件及项目工具配置。 |
| [examples/run_val.sh](examples/run_val.sh) | 移除脚本中硬编码的凭据，改为从环境提供；旧值可能仍在 Git 历史中，推送前应由凭据持有人检查并轮换。 |

## 已有验证与未完成事项

- Windows 原生 CPU 接口回归记录为 **64 passed、2 deselected**；未选的两项依赖 Linux Bash 路径，测试使用明确记录的替身及小模型。这是历史测试结果，本文编写时未重新运行训练或仿真。
- 一条真实 RoboDojo `put_bottles_into_dustbin` 轨迹按记录的约定转换为 361 条转移、354 个训练窗口，并使用生产 ActionEncoder、RFlow 和训练步骤配合小模型完成一次 CPU 参数更新；只有一条训练轨迹，不能据此证明留出集效果或完整模型可用。
- RoboTwin 的 SAPIEN 任务 smoke 曾在软件渲染及本地降噪兼容设置下运行短轨迹；这不是 OpenSora 模型闭环成功率测试。Windows 原生 SAPIEN 的基础渲染和物理能力曾单独验证，完整任务仍未完成原生验收。
- RoboDojo 的 Windows 原生 Isaac Sim 启动时在 RTX 场景插件发生 `0xC0000005`，未完成真实任务步进和相机验证。尚需在服务器验证真实环境、目标域完整 checkpoint、CUDA 训练、留出集视频质量、奖励模型与物理任务成功率。

同级工作区的 `experiments/本机数据与接口复核报告.md`、`experiments/Windows原生验证报告.md` 和 `experiments/真实仿真接口测试报告.md` 保存了详细证据；这些文件不属于本仓库，单独复制本仓库时不会自动包含。
