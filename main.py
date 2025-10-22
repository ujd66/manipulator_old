"""
UR5机械臂抓取系统主程序
该程序实现了基于GraspNet深度学习模型的物体抓取系统，使用UR5机械臂在模拟环境中执行抓取任务

主要功能：
1. 使用GraspNet神经网络预测抓取姿态
2. 从相机图像生成点云数据
3. 进行碰撞检测和抓取筛选
4. 规划机械臂轨迹
5. 在模拟环境中执行完整的抓取和放置任务
"""

import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

from graspnetAPI import GraspGroup

# 设置项目根目录并添加必要的路径到系统路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

# 导入GraspNet相关模块
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# 导入机械臂运动规划和环境模块
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv


def get_net():
    """
    初始化并加载GraspNet神经网络模型
    
    GraspNet是一个用于预测物体抓取姿态的深度学习模型
    该函数创建网络、加载预训练权重并设置为评估模式
    
    Returns:
        net: 加载好权重的GraspNet模型，处于评估模式
    """
    # 创建GraspNet网络
    # input_feature_dim=0: 不使用额外的输入特征
    # num_view=300: 视角数量
    # num_angle=12: 抓取角度的离散化数量
    # num_depth=4: 抓取深度的离散化数量
    # cylinder_radius=0.05: 圆柱半径，用于抓取采样
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    
    # 选择计算设备（优先使用GPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 加载预训练的模型权重
    checkpoint_path = './logs/log_rs/checkpoint-rs.tar'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式（关闭dropout等训练时的操作）
    net.eval()
    return net


def get_and_process_data(imgs):
    """
    从相机图像数据中提取和处理点云
    
    该函数将RGB-D图像转换为点云，并进行采样和预处理，
    为GraspNet网络的输入做准备
    
    Args:
        imgs: 包含'img'(RGB图像)和'depth'(深度图像)的字典
    
    Returns:
        end_points: 包含处理后的点云数据的字典，用于网络输入
        cloud: Open3D点云对象，用于可视化
    """
    num_point = 20000  # 采样点数

    # 归一化颜色图像到[0,1]范围
    color = imgs['img'] / 255.0
    depth = imgs['depth']

    # 设置相机参数
    height = 256
    width = 256
    fovy = np.pi / 4  # 视场角（45度）
    
    # 计算相机内参矩阵
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    # 创建相机信息对象
    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    # 从深度图像创建点云
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 使用深度掩码过滤远距离点（保留深度<2.0米的点）
    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # 采样固定数量的点
    if len(cloud_masked) >= num_point:
        # 如果点数足够，随机采样不重复
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        # 如果点数不足，先取全部，然后重复采样补足
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # 创建Open3D点云对象用于可视化
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    # 准备网络输入数据
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    """
    使用GraspNet网络预测抓取姿态
    
    Args:
        net: 训练好的GraspNet模型
        end_points: 包含点云数据的字典
    
    Returns:
        gg: GraspGroup对象，包含预测的所有抓取姿态
    """
    with torch.no_grad():  # 推理时不需要计算梯度
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)  # 解码网络输出为抓取姿态
    
    # 将预测结果转换为numpy数组并创建GraspGroup对象
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    """
    检测抓取姿态与点云的碰撞
    
    使用无模型碰撞检测器过滤掉会与物体发生碰撞的抓取姿态
    
    Args:
        gg: GraspGroup对象，包含待检测的抓取姿态
        cloud: 点云数据（numpy数组）
    
    Returns:
        gg: 过滤后的GraspGroup对象，不包含碰撞的抓取
    """
    voxel_size = 0.01  # 体素大小（米）
    collision_thresh = 0.01  # 碰撞阈值（米）

    # 创建无模型碰撞检测器
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    
    # 检测碰撞，返回碰撞掩码
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    
    # 过滤掉有碰撞的抓取
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    """
    可视化抓取姿态和点云
    
    使用Open3D可视化工具显示点云和抓取器的3D模型
    
    Args:
        gg: GraspGroup对象，包含要可视化的抓取姿态
        cloud: Open3D点云对象
    """
    # 将抓取姿态转换为Open3D几何对象列表（夹爪模型）
    grippers = gg.to_open3d_geometry_list()
    # 可视化点云和夹爪
    o3d.visualization.draw_geometries([cloud, *grippers])


def generate_grasps(net, imgs, visual=False):
    """
    生成并筛选抓取姿态的完整流程
    
    该函数整合了点云处理、抓取预测、碰撞检测、非极大值抑制等步骤
    
    Args:
        net: 训练好的GraspNet模型
        imgs: 包含RGB和深度图像的字典
        visual: 是否可视化结果
    
    Returns:
        gg: 最终选择的抓取姿态（只包含得分最高的一个）
    """
    # 处理图像数据，生成点云
    end_points, cloud = get_and_process_data(imgs)
    
    # 使用网络预测抓取姿态
    gg = get_grasps(net, end_points)
    
    # 碰撞检测，过滤不可行的抓取
    gg = collision_detection(gg, np.array(cloud.points))
    
    # 非极大值抑制（NMS），去除重复的抓取
    gg.nms()
    
    # 按得分排序
    gg.sort_by_score()
    
    # 只保留得分最高的抓取
    gg = gg[:1]
    
    # 如果需要，可视化结果
    if visual:
        vis_grasps(gg, cloud)
    
    return gg


if __name__ == '__main__':
    """
    主程序：执行完整的机械臂抓取任务
    
    任务流程：
    1. 初始化网络和环境
    2. 从相机获取图像并预测抓取姿态
    3. 规划机械臂轨迹移动到抓取位置
    4. 执行抓取
    5. 将物体移动到目标位置
    6. 释放物体
    7. 返回初始位置
    """
    
    # ==================== 第一阶段：初始化和抓取预测 ====================
    
    # 加载GraspNet神经网络
    net = get_net()

    # 创建UR5机械臂仿真环境
    env = UR5GraspEnv()
    env.reset()
    
    # 运行1000步让环境稳定（物体静止）
    for i in range(1000):
        env.step()
    
    # 渲染相机图像（RGB+深度）
    imgs = env.render()

    # 生成抓取姿态并可视化
    gg = generate_grasps(net, imgs, True)

    # ==================== 第二阶段：坐标系变换 ====================
    
    # 获取机器人对象和基座坐标系
    robot = env.robot
    T_wb = robot.base  # 世界坐标系到机器人基座的变换
    
    # 定义相机坐标系在世界坐标系中的位置和姿态
    # 相机坐标系的x轴方向（法向量）
    n_wc = np.array([0.0, -1.0, 0.0])
    # 相机坐标系的y轴方向
    o_wc = np.array([-1.0, 0.0, -0.5])
    # 相机坐标系的原点位置
    t_wc = np.array([1.0, 0.6, 2.0])
    # 构造世界坐标系到相机坐标系的变换矩阵
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    
    # 相机坐标系到物体抓取点的变换（从抓取姿态获取）
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    # 计算世界坐标系到抓取点的变换
    T_wo = T_wc * T_co

    # ==================== 第三阶段：规划到抓取位置的轨迹 ====================
    
    # 轨迹0：从当前关节角移动到预备姿态（关节空间规划）
    time0 = 2  # 执行时间2秒
    q0 = robot.get_joint()  # 当前关节角
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])  # 目标关节角（预备姿态）

    # 创建关节空间轨迹规划器
    parameter0 = JointParameter(q0, q1)  # 关节空间参数
    velocity_parameter0 = QuinticVelocityParameter(time0)  # 五次多项式速度规划
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner0 = TrajectoryPlanner(trajectory_parameter0)

    # 轨迹1：从预备姿态移动到抓取点前方（笛卡尔空间规划）
    time1 = 2
    robot.set_joint(q1)  # 临时设置为目标关节角以获取笛卡尔位姿
    T1 = robot.get_cartesian()  # 预备姿态的笛卡尔位姿
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)  # 抓取点前方10cm处（接近点）
    
    # 创建笛卡尔空间轨迹规划器（直线运动）
    position_parameter1 = LinePositionParameter(T1.t, T2.t)  # 位置直线插值
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))  # 姿态插值
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time1)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner1 = TrajectoryPlanner(trajectory_parameter1)

    # 轨迹2：从接近点移动到抓取点（笛卡尔空间规划）
    time2 = 2
    T3 = T_wo  # 最终抓取位置
    
    # 创建接近抓取点的轨迹
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time2)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner2 = TrajectoryPlanner(trajectory_parameter2)

    # ==================== 第四阶段：执行到抓取位置的运动 ====================
    
    # 组合所有轨迹段
    time_array = [0, time0, time1, time2]
    planner_array = [planner0, planner1, planner2]
    total_time = np.sum(time_array)

    # 生成时间序列（采样频率500Hz，步长0.002秒）
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0, total_time, time_step_num)

    # 计算每段轨迹的累计时间
    time_cumsum = np.cumsum(time_array)
    action = np.zeros(7)  # 动作向量：6个关节角 + 1个夹爪控制
    
    # 执行轨迹跟踪
    for i, timei in enumerate(times):
        for j in range(len(time_cumsum)):
            if timei < time_cumsum[j]:
                # 根据当前时间插值计算目标位姿
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                
                if isinstance(planner_interpolate, np.ndarray):
                    # 关节空间轨迹：直接使用关节角
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    # 笛卡尔空间轨迹：移动到目标位姿并获取关节角
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                
                # 更新动作并执行环境步进
                action[:6] = joint
                env.step(action)
                break

    # ==================== 第五阶段：闭合夹爪抓取物体 ====================
    
    # 逐渐增加夹爪控制信号（闭合夹爪）
    for i in range(1500):
        action[-1] += 0.2  # 增加夹爪力量
        action[-1] = np.min([action[-1], 255])  # 限制最大值
        env.step(action)

    # ==================== 第六阶段：规划到放置位置的轨迹 ====================
    
    # 轨迹3：抬起物体（向上移动10cm）
    time3 = 2
    T4 = sm.SE3.Trans(0.0, 0.0, 0.1) * T3  # 在当前位置上方10cm
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time3)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner3 = TrajectoryPlanner(trajectory_parameter3)

    # 轨迹4：移动到目标区域上方
    time4 = 2
    T5 = sm.SE3.Trans(1.4, 0.2, T4.t[2]) * sm.SE3(sm.SO3(T4.R))  # 移动到目标位置(1.4, 0.2)
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time4)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner4 = TrajectoryPlanner(trajectory_parameter4)

    # 轨迹5：移动到最终放置位置上方并调整姿态
    time5 = 2
    # 移动到(0.2, 0.2)并绕z轴旋转-90度
    T6 = sm.SE3.Trans(0.2, 0.2, T5.t[2]) * sm.SE3(sm.SO3.Rz(-np.pi / 2) * sm.SO3(T5.R))
    position_parameter5 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter5 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter5 = CartesianParameter(position_parameter5, attitude_parameter5)
    velocity_parameter5 = QuinticVelocityParameter(time5)
    trajectory_parameter5 = TrajectoryParameter(cartesian_parameter5, velocity_parameter5)
    planner5 = TrajectoryPlanner(trajectory_parameter5)

    # 轨迹6：下降到放置位置（向下移动10cm）
    time6 = 2
    T7 = sm.SE3.Trans(0.0, 0.0, -0.1) * T6
    position_parameter6 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # ==================== 第七阶段：执行到放置位置的运动 ====================
    
    # 组合搬运轨迹段
    time_array = [0.0, time3, time4, time5, time6]
    planner_array = [planner3, planner4, planner5, planner6]
    total_time = np.sum(time_array)

    # 生成时间序列
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    # 执行搬运轨迹
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                # 插值计算目标位姿
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                
                action[:6] = joint
                env.step(action)
                break

    # ==================== 第八阶段：打开夹爪释放物体 ====================
    
    # 逐渐减小夹爪控制信号（打开夹爪）
    for i in range(1500):
        action[-1] -= 0.2  # 减小夹爪力量
        action[-1] = np.max([action[-1], 0])  # 限制最小值
        env.step(action)

    # ==================== 第九阶段：抬起末端执行器 ====================
    
    # 轨迹7：向上移动20cm离开放置点
    time7 = 2
    T8 = sm.SE3.Trans(0.0, 0.0, 0.2) * T7
    position_parameter7 = LinePositionParameter(T7.t, T8.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T7.R), sm.SO3(T8.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)

    # 执行抬起轨迹
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # ==================== 第十阶段：返回初始位置 ====================
    
    # 轨迹8：从当前位置返回初始关节角（关节空间规划）
    time8 = 2.0
    q8 = robot.get_joint()  # 当前关节角
    q9 = q0  # 初始关节角

    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)

    # 执行返回轨迹
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # ==================== 第十一阶段：完成任务 ====================
    
    # 运行2000步以观察最终状态
    for i in range(2000):
        env.step()

    # 关闭环境
    env.close()
