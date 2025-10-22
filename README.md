# 机械臂智能抓取系统

## 项目简介

这是一个基于深度学习的机械臂抓取系统，使用GraspNet神经网络预测物体抓取姿态，并通过UR5机械臂在模拟环境中执行完整的抓取和放置任务。该项目整合了计算机视觉、深度学习、机器人运动规划等多个领域的技术。

## 系统架构

```
manipulator_old/
├── main.py                          # 主程序入口
├── graspnet-baseline/               # GraspNet基线模型
│   ├── models/                      # 神经网络模型定义
│   │   ├── graspnet.py             # GraspNet主模型
│   │   ├── backbone.py             # 骨干网络
│   │   └── modules.py              # 网络模块
│   ├── dataset/                     # 数据集处理
│   │   └── graspnet_dataset.py     # 数据加载器
│   ├── utils/                       # 工具函数
│   │   ├── collision_detector.py   # 碰撞检测
│   │   └── data_utils.py           # 数据处理工具
│   └── graspnetAPI/                 # GraspNet API
│       └── graspnetAPI/
│           └── grasp.py             # 抓取姿态表示
├── manipulator_grasp/               # 机械臂控制模块
│   ├── env/                         # 仿真环境
│   │   └── ur5_grasp_env.py        # UR5抓取环境
│   ├── arm/                         # 机械臂控制
│   │   └── motion_planning/        # 运动规划
│   │       ├── motion_parameter.py # 运动参数
│   │       └── trajectory_planning/ # 轨迹规划
│   ├── assets/                      # 3D模型资源
│   └── utils/                       # 工具函数
└── logs/                            # 训练日志和模型权重
    └── log_rs/
        └── checkpoint-rs.tar        # 预训练模型权重
```

## 核心功能模块

### 1. 抓取姿态预测（GraspNet）

- **功能**：基于RGB-D图像预测6D抓取姿态
- **输入**：点云数据（20000个点）
- **输出**：抓取姿态组（位置、旋转、得分）
- **关键文件**：`graspnet-baseline/models/graspnet.py`

### 2. 点云处理

- **功能**：将RGB-D图像转换为点云数据
- **处理步骤**：
  1. 根据相机内参将深度图转换为3D点云
  2. 过滤远距离点（深度<2.0米）
  3. 随机采样固定数量的点
- **关键函数**：`get_and_process_data()`

### 3. 碰撞检测

- **功能**：检测抓取姿态是否会与物体碰撞
- **方法**：基于体素的无模型碰撞检测
- **参数**：
  - 体素大小：0.01米
  - 碰撞阈值：0.01米
  - 接近距离：0.05米
- **关键函数**：`collision_detection()`

### 4. 轨迹规划

- **关节空间规划**：用于大幅度姿态变换
- **笛卡尔空间规划**：用于精确的位置控制
- **速度规划**：五次多项式速度曲线，保证平滑运动
- **关键模块**：`manipulator_grasp/arm/motion_planning/`

### 5. 机械臂控制

- **机器人型号**：UR5（6自由度）
- **控制频率**：500Hz（时间步长0.002秒）
- **控制方式**：位置控制 + 夹爪力控制
- **关键模块**：`manipulator_grasp/env/ur5_grasp_env.py`

## 程序执行流程

### 完整任务流程

```
1. 初始化阶段
   ├── 加载GraspNet神经网络模型
   ├── 创建UR5仿真环境
   └── 运行1000步稳定环境

2. 感知阶段
   ├── 渲染相机图像（RGB+深度）
   ├── 生成点云数据
   ├── 使用GraspNet预测抓取姿态
   ├── 碰撞检测过滤
   ├── 非极大值抑制（NMS）
   └── 选择得分最高的抓取

3. 坐标系变换
   ├── 定义相机坐标系
   ├── 计算抓取点在世界坐标系中的位置
   └── 转换为机器人基座坐标系

4. 移动到抓取位置
   ├── 轨迹0：移动到预备姿态（关节空间，2秒）
   ├── 轨迹1：移动到抓取点前方10cm（笛卡尔空间，2秒）
   └── 轨迹2：接近抓取点（笛卡尔空间，2秒）

5. 执行抓取
   └── 闭合夹爪（1500步，约3秒）

6. 搬运物体
   ├── 轨迹3：抬起物体10cm（2秒）
   ├── 轨迹4：移动到目标区域上方（2秒）
   ├── 轨迹5：移动到放置位置并调整姿态（2秒）
   └── 轨迹6：下降到放置位置（2秒）

7. 释放物体
   └── 打开夹爪（1500步，约3秒）

8. 返回初始位置
   ├── 轨迹7：抬起末端执行器20cm（2秒）
   └── 轨迹8：返回初始关节角（关节空间，2秒）

9. 完成任务
   └── 观察最终状态（2000步）
```

## 依赖环境

### Python版本
- Python 3.8+

### 主要依赖库

```bash
# 深度学习框架
torch>=1.8.0
torchvision>=0.9.0

# 点云处理和可视化
open3d>=0.13.0

# 科学计算
numpy>=1.19.0
scipy>=1.5.0

# 空间数学
spatialmath-python>=0.11.0

# 机器人学
roboticstoolbox-python>=0.11.0

# 图像处理
Pillow>=8.0.0

# GraspNet API
graspnetAPI
```

### 系统依赖（针对WSL2/Linux）

```bash
# 显示环境配置（用于Open3D可视化）
export WAYLAND_DISPLAY=""
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
```

## 安装教程

### 1. 克隆项目

```bash
git clone 
cd manipulator_old
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install open3d numpy scipy Pillow spatialmath-python roboticstoolbox-python

# 安装GraspNet相关模块
cd graspnet-baseline
pip install -r requirements.txt
cd ..
```

### 3. 下载预训练模型

将预训练的GraspNet模型权重放置到：
```
logs/log_rs/checkpoint-rs.tar
```

### 4. 配置显示环境（WSL2）

在 `~/.bashrc` 中添加：
```bash
export WAYLAND_DISPLAY=""
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
```

然后执行：
```bash
source ~/.bashrc
```

## 使用说明

### 运行主程序

```bash
python main.py
```

### 程序输出

1. **可视化窗口**：显示点云和预测的抓取姿态（Open3D窗口）
2. **仿真窗口**：显示UR5机械臂执行抓取任务的3D仿真
3. **终端输出**：显示执行进度和状态信息

### 调整参数

#### 修改抓取可视化

在 `main.py` 中修改：
```python
gg = generate_grasps(net, imgs, visual=True)  # True启用可视化，False禁用
```

#### 修改采样点数

在 `get_and_process_data()` 函数中修改：
```python
num_point = 20000  # 可以调整为其他值
```

#### 修改轨迹执行时间

在各个轨迹规划部分修改 `time0`, `time1`, ... 等参数

#### 修改夹爪控制

```python
# 抓取速度（增加步数=更慢）
for i in range(1500):  # 修改循环次数
    action[-1] += 0.2  # 修改增量
```

## 关键技术说明

### 1. GraspNet神经网络

GraspNet是一个端到端的抓取姿态检测网络：

- **输入**：点云数据（N×3）
- **特征提取**：PointNet++骨干网络
- **抓取生成**：在点云表面采样候选抓取点
- **抓取评估**：为每个候选抓取预测得分、宽度、深度等参数
- **输出**：抓取姿态参数（位置、旋转矩阵、宽度、得分）

### 2. 坐标系变换

系统涉及多个坐标系：

- **世界坐标系（W）**：仿真环境的全局坐标系
- **机器人基座坐标系（B）**：UR5机器人的基座
- **相机坐标系（C）**：相机的坐标系
- **物体坐标系（O）**：抓取点的局部坐标系

变换链：`T_wo = T_wc × T_co`

### 3. 轨迹规划

#### 关节空间规划
- 适用于大幅度姿态变换
- 直接在关节角空间插值
- 不保证末端执行器直线运动

#### 笛卡尔空间规划
- 适用于精确位置控制
- 末端执行器沿直线运动
- 需要逆运动学求解

#### 速度规划
使用五次多项式速度曲线：
- 起点和终点速度、加速度为零
- 保证运动平滑，无突变
- 适合机器人控制

### 4. 碰撞检测

基于体素的无模型碰撞检测：

1. 将点云体素化
2. 在抓取姿态周围生成夹爪模型
3. 检查夹爪与体素的重叠
4. 过滤有碰撞的抓取

## 常见问题

### Q1: Open3D可视化窗口无法显示

**解决方案**：
```bash
# 检查X11服务器是否运行
echo $DISPLAY

# 确保环境变量已设置
export DISPLAY=:0

# 在Windows上安装VcXsrv或Xming
```

### Q2: CUDA out of memory

**解决方案**：
```python
# 在get_and_process_data()中减少点数
num_point = 10000  # 从20000减少到10000

# 或使用CPU
device = torch.device('cpu')
```

### Q3: 找不到模型权重文件

**解决方案**：
```bash
# 确保模型文件存在
ls logs/log_rs/checkpoint-rs.tar

# 或修改main.py中的路径
checkpoint_path = '/path/to/your/checkpoint.tar'
```

### Q4: 轨迹规划失败

**可能原因**：
- 目标位置超出机器人工作空间
- 逆运动学无解
- 关节角度限制

**解决方案**：
- 检查目标位置是否合理
- 调整轨迹参数
- 查看终端错误信息

## 性能指标

- **抓取预测时间**：~0.1-0.5秒（GPU）
- **点云处理时间**：~0.05秒
- **碰撞检测时间**：~0.01秒
- **轨迹执行时间**：总计约28秒
  - 移动到抓取位置：6秒
  - 抓取：3秒
  - 搬运：8秒
  - 释放：3秒
  - 返回：4秒
- **控制频率**：500Hz

## 扩展功能建议

### 1. 多物体抓取
- 修改 `generate_grasps()` 返回多个抓取
- 添加物体分割算法
- 规划抓取顺序

### 2. 实时抓取
- 移除环境稳定步骤
- 添加动态物体检测
- 实现闭环控制

### 3. 真实机器人部署
- 替换仿真环境为真实机器人接口
- 添加传感器数据处理
- 实现安全监控

### 4. 强化学习
- 使用抓取成功率作为奖励
- 训练端到端抓取策略
- 优化轨迹规划

## 参考文献

1. **GraspNet-1Billion**: [论文链接](https://graspnet.net/)
   - Fang et al., "GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping", CVPR 2020

2. **PointNet++**: 
   - Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017

3. **UR5机器人**:
   - Universal Robots官方文档

## 许可证

本项目基于原始GraspNet和UR5相关项目的许可证。具体请查看 `LICENSE` 文件。

## 贡献指南

欢迎提交问题和改进建议！






