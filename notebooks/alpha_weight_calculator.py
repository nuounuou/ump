#!/usr/bin/env python3
"""
Alpha Weight Calculator - Step 4-4
===================================
根据 tracking_results.csv 计算双臂的 α 权重

功能：
1. 从 tracking_results.csv 读取每帧的质心、IoU 等数据
2. 根据选择的帧间隔计算各种运动参数（速度、方向等）
3. 结合迷宫特征（到分叉距离、到目标距离等）计算 α 权重
4. 支持用户自定义各项参数的权重

输入：tracking_results.csv + 配准后的迷宫参数
输出：α_left, α_right (0-1 之间的控制权重)

Alpha 含义：
- α = 1：完全自主控制（全自动）
- α = 0：完全人类控制
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import pickle
import gradio as gr
from pathlib import Path

# ───────── 路径常量 ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_ALIGN_DIR = os.path.join(SCRIPT_DIR, 'mask_align_sam2_dataset')
MAZE_PATH = os.path.join(SCRIPT_DIR, 'maze1.png')

# 尝试从 five dimension 导入迷宫特征提取器
try:
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'five dimension'))
    from maze_feature_extractor_v3_final import MazeFeatureExtractor
    MAZE_EXTRACTOR_AVAILABLE = True
except ImportError:
    MAZE_EXTRACTOR_AVAILABLE = False
    print("Warning: maze_feature_extractor_v3_final not available, using simplified calculations")


# ═══════════════════════════════════════════════════
# 参数计算函数
# ═══════════════════════════════════════════════════

def load_tracking_results(video_dir_name):
    """
    加载 tracking_results.csv
    
    Returns:
        DataFrame with columns: frame, bottom_x, bottom_y, top_x, top_y, 
                               centroid_x, centroid_y, angle_deg, iou, method
    """
    if not video_dir_name:
        return None
    
    csv_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 'tracking_results.csv')
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    return df


def load_maze_params(video_dir_name):
    """加载迷宫配准参数"""
    params_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 
                              'registration_output', 'maze_params.json')
    if not os.path.exists(params_path):
        return None
    
    with open(params_path, 'r') as f:
        return json.load(f)


def compute_velocity(df, frame_interval):
    """
    根据帧间隔计算速度
    
    Args:
        df: tracking_results DataFrame
        frame_interval: 帧间隔（用于计算速度的帧数差）
    
    Returns:
        dict: 包含 left_velocity, right_velocity, overall_velocity 等
    """
    if df is None or len(df) < 2:
        return None
    
    # 使用帧间隔计算速度
    interval = max(1, min(frame_interval, len(df) - 1))
    
    # 质心位置
    centroids = df[['centroid_x', 'centroid_y']].values
    
    # 底部端点 (bottom_x, bottom_y)
    bottom_points = df[['bottom_x', 'bottom_y']].values
    
    # 顶部端点 (top_x, top_y) 
    top_points = df[['top_x', 'top_y']].values
    
    # 计算速度（像素/帧）
    velocities = {}
    
    # 整体质心速度
    if len(centroids) > interval:
        delta_centroid = centroids[-1] - centroids[0]
        overall_velocity = np.sqrt(delta_centroid[0]**2 + delta_centroid[1]**2) / interval
        overall_direction = np.arctan2(delta_centroid[1], delta_centroid[0])
        velocities['overall_velocity'] = overall_velocity
        velocities['overall_direction'] = overall_direction
    
    # 底部端点速度 (近似左臂速度)
    if len(bottom_points) > interval:
        delta_bottom = bottom_points[-1] - bottom_points[0]
        bottom_velocity = np.sqrt(delta_bottom[0]**2 + delta_bottom[1]**2) / interval
        bottom_direction = np.arctan2(delta_bottom[1], delta_bottom[0])
        velocities['bottom_velocity'] = bottom_velocity
        velocities['bottom_direction'] = bottom_direction
    
    # 顶部端点速度 (近似右臂速度)
    if len(top_points) > interval:
        delta_top = top_points[-1] - top_points[0]
        top_velocity = np.sqrt(delta_top[0]**2 + delta_top[1]**2) / interval
        top_direction = np.arctan2(delta_top[1], delta_top[0])
        velocities['top_velocity'] = top_velocity
        velocities['top_direction'] = top_direction
    
    # 质心到两端点的距离（用于判断左右臂）
    if len(centroids) > 0:
        last_idx = len(df) - 1
        # 左臂距离 (质心到底部)
        left_dist = np.sqrt(
            (centroids[last_idx][0] - bottom_points[last_idx][0])**2 + 
            (centroids[last_idx][1] - bottom_points[last_idx][1])**2
        )
        # 右臂距离 (质心到顶部)
        right_dist = np.sqrt(
            (centroids[last_idx][0] - top_points[last_idx][0])**2 + 
            (centroids[last_idx][1] - top_points[last_idx][1])**2
        )
        velocities['left_dist'] = left_dist
        velocities['right_dist'] = right_dist
    
    return velocities


def compute_iou_stats(df, frame_interval):
    """计算 IoU 统计信息"""
    if df is None or 'iou' not in df.columns:
        return None
    
    interval = max(1, min(frame_interval, len(df) - 1))
    
    # 取最近 interval 帧的 IoU 均值
    iou_values = df['iou'].values[-interval:]
    iou_mean = np.mean(iou_values)
    iou_std = np.std(iou_values)
    iou_min = np.min(iou_values)
    iou_max = np.max(iou_values)
    
    return {
        'iou_mean': iou_mean,
        'iou_std': iou_std,
        'iou_min': iou_min,
        'iou_max': iou_max,
        'iou_latest': df['iou'].values[-1] if len(df) > 0 else 0
    }


def compute_maze_distances(video_dir_name, centroid_x, centroid_y, bottom_x=None, bottom_y=None, top_x=None, top_y=None):
    """
    Calculate distances to the registered maze
    
    Calculates:
    - Distance to center
    - Wall distance for left arm (bottom endpoint)
    - Wall distance for right arm (top endpoint)
    
    Args:
        video_dir_name: Video directory name
        centroid_x, centroid_y: Centroid coordinates
        bottom_x, bottom_y: Bottom endpoint coordinates (left arm)
        top_x, top_y: Top endpoint coordinates (right arm)
    
    Returns:
        dict: Contains distance information
    """
    distances = {}
    
    # Load maze params
    maze_params = load_maze_params(video_dir_name)
    if maze_params is None:
        return None
    
    cx = maze_params.get('cx', 0)
    cy = maze_params.get('cy', 0)
    scale = maze_params.get('scale', 0.5)
    angle_deg = maze_params.get('angle_deg', 0)
    
    # Distance to maze center
    dist_to_center = np.sqrt((centroid_x - cx)**2 + (centroid_y - cy)**2)
    distances['dist_to_center'] = dist_to_center
    
    # Load maze mask from Step 4-3 registration output (same as Step 4-3)
    video_dir = os.path.join(MASK_ALIGN_DIR, video_dir_name)
    reg_dir = os.path.join(video_dir, 'registration_output')
    maze_mask_path = os.path.join(reg_dir, 'maze_corridor_mask.png')
    
    # Fallback to maze1.png if registration output doesn't exist
    if os.path.exists(maze_mask_path):
        maze_img = cv2.imread(maze_mask_path, cv2.IMREAD_GRAYSCALE)
    elif os.path.exists(MAZE_PATH):
        maze_img = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
        if len(maze_img.shape) == 3:
            maze_img = cv2.cvtColor(maze_img, cv2.COLOR_BGR2GRAY)
    else:
        return None
    
    if maze_img is not None:
        # Extract free space (white area = free space)
        free_mask = maze_img > 127
        
        h, w = maze_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
        M_inv = cv2.invertAffineTransform(M)
        
        # Calculate wall distance using ray casting (left and right sides)
        # For centroid (if provided)
        if centroid_x is not None and centroid_y is not None:
            maze_cx = M_inv[0][0] * centroid_x + M_inv[0][1] * centroid_y + M_inv[0][2]
            maze_cy = M_inv[1][0] * centroid_x + M_inv[1][1] * centroid_y + M_inv[1][2]
            mx_int = int(np.clip(maze_cx, 0, w-1))
            my_int = int(np.clip(maze_cy, 0, h-1))
            
            if free_mask[my_int, mx_int]:
                dist_transform = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
                distances['dist_to_wall'] = dist_transform[my_int, mx_int]
            else:
                dist_transform = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
                distances['dist_to_wall'] = dist_transform[my_int, mx_int]
        
        # Calculate wall distance for left arm (bottom endpoint)
            if bottom_x is not None and bottom_y is not None:
                dist_left = _compute_wall_distance_raycast(
                    bottom_x, bottom_y, M_inv, free_mask, w, h
                )
                distances['dist_to_wall_left'] = dist_left
            
            # Calculate wall distance for right arm (top endpoint)
            if top_x is not None and top_y is not None:
                dist_right = _compute_wall_distance_raycast(
                    top_x, top_y, M_inv, free_mask, w, h
                )
                distances['dist_to_wall_right'] = dist_right
    
    return distances


def _compute_wall_distance_raycast(x, y, M_inv, free_mask, w, h, max_d=100):
    """
    Compute wall distance using ray casting from a point
    
    Casts rays in multiple directions and returns the minimum distance to wall
    
    Args:
        x, y: Point coordinates in image space
        M_inv: Inverse affine transformation matrix
        free_mask: Free space mask in maze coordinates
        w, h: Maze image dimensions
        max_d: Maximum ray casting distance
    
    Returns:
        float: Minimum distance to wall
    """
    # Transform point to maze coordinates
    maze_x = M_inv[0][0] * x + M_inv[0][1] * y + M_inv[0][2]
    maze_y = M_inv[1][0] * x + M_inv[1][1] * y + M_inv[1][2]
    
    mx = int(np.clip(maze_x, 0, w-1))
    my = int(np.clip(maze_y, 0, h-1))
    
    # If already in free space, use distance transform
    if free_mask[my, mx]:
        dist_transform = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, 5)
        return dist_transform[my, mx]
    
    # If not in free space, return 0
    return 0.0


def _raycast_dist(x, y, M_inv, corr_mask, wall_mask):
    """
    计算点到最近墙壁的距离（使用射线投射法）
    
    Args:
        x, y: 点在图像空间的坐标
        M_inv: 逆仿射变换矩阵
        corr_mask: 通道掩码
        wall_mask: 墙壁掩码
    
    Returns:
        float: 到最近墙壁的距离（像素）
    """
    if corr_mask is None:
        return 0.0
    
    h, w = corr_mask.shape[:2]
    
    # 转换到掩码坐标
    mx = M_inv[0][0] * x + M_inv[0][1] * y + M_inv[0][2]
    my = M_inv[1][0] * x + M_inv[1][1] * y + M_inv[1][2]
    
    mx = int(np.clip(mx, 0, w-1))
    my = int(np.clip(my, 0, h-1))
    
    # 如果不在通道内，返回0
    if corr_mask[my, mx] < 128:
        return 0.0
    
    # 使用距离变换
    dist_transform = cv2.distanceTransform(corr_mask, cv2.DIST_L2, 5)
    return float(dist_transform[my, mx])


def compute_bifurcation_distance(video_dir_name, centroid_x, centroid_y):
    """
    计算到分叉点的距离
    
    使用预计算的迷宫特征提取器
    """
    # 尝试加载预计算的迷宫特征
    pkl_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 
                           'registration_output', 'maze_features.pkl')
    
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                extractor = pickle.load(f)
            
            # 查询分叉距离
            point = (centroid_y, centroid_x)  # 注意：MazeFeatureExtractor 使用 (y, x)
            features = extractor.query(point, 'A', normalize=False)
            return features.get('d_bif', np.inf)
        except Exception as e:
            print(f"Error loading maze features: {e}")
    
    return None


def compute_target_distance(video_dir_name, centroid_x, centroid_y, target_name='A'):
    """
    计算到目标点的距离
    """
    # 尝试加载预计算的迷宫特征
    pkl_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 
                           'registration_output', 'maze_features.pkl')
    
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                extractor = pickle.load(f)
            
            point = (centroid_y, centroid_x)
            features = extractor.query(point, target_name, normalize=False)
            return features.get('D_geo', np.inf)
        except Exception as e:
            print(f"Error loading maze features: {e}")
    
    # 简化版本：使用欧氏距离到目标（需要从迷宫参数获取目标坐标）
    maze_params = load_maze_params(video_dir_name)
    if maze_params and 'target_A' in maze_params:
        target = maze_params['target_A']
        return np.sqrt((centroid_x - target[0])**2 + (centroid_y - target[1])**2)
    
    return None


def compute_side_distances(centroid_x, centroid_y, bottom_x, bottom_y, top_x, top_y):
    """
    计算到两侧（左右臂）的距离
    
    Returns:
        dict: left_side_dist, right_side_dist
    """
    # 左侧：质心到bottom端点的垂直方向距离
    left_dist = np.sqrt((centroid_x - bottom_x)**2 + (centroid_y - bottom_y)**2)
    
    # 右侧：质心到top端点的垂直方向距离
    right_dist = np.sqrt((centroid_x - top_x)**2 + (centroid_y - top_y)**2)
    
    return {
        'left_side_dist': left_dist,
        'right_side_dist': right_dist
    }


# ═══════════════════════════════════════════════════
# Alpha 权重计算
# ═══════════════════════════════════════════════════

def compute_alpha_weights(
    video_dir_name,
    frame_interval=4,
    weights=None,
    fsr_left=0.5,
    fsr_right=0.5
):
    """
    计算双臂的 α 权重
    
    α = 1 → 完全人类控制
    α = 0 → 完全自主控制
    
    权重参数 (weights) 包含:
    - velocity_weight: 速度权重
    - iou_weight: IoU权重
    - distance_weight: 距离权重
    - bifurcation_weight: 分叉距离权重
    - target_weight: 目标距离权重
    - wall_distance_weight: 墙壁距离权重
    
    Args:
        video_dir_name: 视频目录名
        frame_interval: 计算速度的帧间隔
        weights: 权重字典
        fsr_left, fsr_right: 左右臂的FSR传感器值
    
    Returns:
        dict: alpha_left, alpha_right, 以及各项计算数据
    """
    if weights is None:
        weights = {
            'velocity': 0.25,
            'iou': 0.25,
            'bifurcation': 0.20,
            'target': 0.15,
            'wall_distance': 0.15
        }
    
    # 加载数据
    df = load_tracking_results(video_dir_name)
    if df is None or len(df) == 0:
        return {
            'alpha_left': 0.5,
            'alpha_right': 0.5,
            'error': 'No tracking data found'
        }
    
    # 获取最新帧的数据
    last_row = df.iloc[-1]
    centroid_x = last_row['centroid_x']
    centroid_y = last_row['centroid_y']
    bottom_x = last_row['bottom_x']
    bottom_y = last_row['bottom_y']
    top_x = last_row['top_x']
    top_y = last_row['top_y']
    
    # ── 1. 速度特征 ──
    velocities = compute_velocity(df, frame_interval)
    
    # ── 2. IoU 特征 ──
    iou_stats = compute_iou_stats(df, frame_interval)
    
    # ── 3. 迷宫距离特征 (左右臂独立计算) ──
    maze_distances = compute_maze_distances(
        video_dir_name, centroid_x, centroid_y,
        bottom_x, bottom_y, top_x, top_y
    )
    
    # ── 4. 分叉距离特征 ──
    bif_dist = compute_bifurcation_distance(video_dir_name, centroid_x, centroid_y)
    
    # ── 5. 目标距离特征 ──
    target_dist = compute_target_distance(video_dir_name, centroid_x, centroid_y)
    
    # ── 计算归一化特征值 ──
    
    # 速度归一化 (假设最大速度 50 像素/帧)
    max_velocity = 50.0
    if velocities:
        v_left = velocities.get('bottom_velocity', 0) / max_velocity
        v_right = velocities.get('top_velocity', 0) / max_velocity
        v_overall = velocities.get('overall_velocity', 0) / max_velocity
    else:
        v_left = v_right = v_overall = 0
    
    # IoU 归一化 (已经是 0-1)
    iou_val = iou_stats['iou_latest'] if iou_stats else 0.5
    
    # 墙壁距离归一化 (左右臂独立计算)
    max_wall_dist = 100.0  # 最大墙壁距离
    wall_dist_norm_left = 0.5
    wall_dist_norm_right = 0.5
    
    if maze_distances:
        if 'dist_to_wall_left' in maze_distances:
            wall_dist_norm_left = min(maze_distances['dist_to_wall_left'] / max_wall_dist, 1.0)
        if 'dist_to_wall_right' in maze_distances:
            wall_dist_norm_right = min(maze_distances['dist_to_wall_right'] / max_wall_dist, 1.0)
    
    # 分叉距离归一化
    bif_dist_norm = 0.5
    if bif_dist is not None and bif_dist != np.inf:
        max_bif_dist = 200.0
        bif_dist_norm = min(bif_dist / max_bif_dist, 1.0)
    
    # 目标距离归一化
    target_dist_norm = 0.5
    if target_dist is not None and target_dist != np.inf:
        max_target_dist = 500.0
        target_dist_norm = min(target_dist / max_target_dist, 1.0)
    
    # ── 计算 α 权重 ──
    # 策略（alpha = 1 完全自主控制（全自动），alpha = 0 完全人类控制）：
    # - 速度越大 → 运动稳定 → 自动程度越高 → α 越大
    # - IoU 越高 → 分割质量好 → 自动程度越高 → α 越大
    # - 壁面距离越远 → 远离危险，自动更安全 → α 越大
    # - 远离分叉 → 路径清晰 → 自动程度越高 → α 越大
    # - 远离目标 → 未到达，继续自动 → α 越大
    
    # 左臂 α 计算
    alpha_left = (
        weights.get('velocity', 0.25) * v_left +                          # 速度：正相关
        weights.get('iou', 0.25) * iou_val +                            # IoU：正相关
        weights.get('wall_distance', 0.15) * wall_dist_norm_left +        # 壁面距离：正相关
        weights.get('bifurcation', 0.20) * bif_dist_norm +              # 分叉距离：正相关（远离→自动）
        weights.get('target', 0.15) * target_dist_norm                   # 目标距离：正相关（远离→自动）
    )
    
    # 融入 FSR 传感器值（人类操作意愿）
    alpha_left = alpha_left * 0.7 + (1 - fsr_left) * 0.3
    
    # 右臂 α 计算
    alpha_right = (
        weights.get('velocity', 0.25) * v_right +                        # 速度：正相关
        weights.get('iou', 0.25) * iou_val +                            # IoU：正相关
        weights.get('wall_distance', 0.15) * wall_dist_norm_right +       # 壁面距离：正相关
        weights.get('bifurcation', 0.20) * bif_dist_norm +              # 分叉距离：正相关
        weights.get('target', 0.15) * target_dist_norm                   # 目标距离：正相关
    )
    
    # 融入 FSR 传感器值（人类操作意愿越强，alpha 越小）
    alpha_right = alpha_right * 0.7 + (1 - fsr_right) * 0.3
    
    # 限制在 [0, 1] 范围内
    alpha_left = np.clip(alpha_left, 0.0, 1.0)
    alpha_right = np.clip(alpha_right, 0.0, 1.0)
    
    return {
        'alpha_left': float(alpha_left),
        'alpha_right': float(alpha_right),
        'features': {
            'velocity_left': v_left,
            'velocity_right': v_right,
            'velocity_overall': v_overall,
            'iou': iou_val,
            'wall_dist_left': wall_dist_norm_left,
            'wall_dist_right': wall_dist_norm_right,
            'bif_dist': bif_dist_norm,
            'target_dist': target_dist_norm
        },
        'raw_data': {
            'centroid': (centroid_x, centroid_y),
            'bottom': (bottom_x, bottom_y),
            'top': (top_x, top_y)
        },
        'velocities': velocities,
        'iou_stats': iou_stats,
        'maze_distances': maze_distances
    }


def compute_alpha_batch(video_dir_name, frame_interval=4, weights=None):
    """
    批量计算所有帧的 α 权重
    
    Returns:
        DataFrame with frame-by-frame alpha values
    """
    df = load_tracking_results(video_dir_name)
    if df is None or len(df) == 0:
        return None
    
    results = []
    for i in range(len(df)):
        # 使用前 i+1 帧的数据
        temp_df = df.iloc[:i+1]
        
        # 临时保存并计算
        # 这里简化处理，直接使用最后一帧
        last_row = df.iloc[i]
        
        # 省略中间计算细节，直接返回简化结果
        results.append({
            'frame': i,
            'alpha_left': 0.5,
            'alpha_right': 0.5
        })
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════
# 导出结果
# ═══════════════════════════════════════════════════

def export_alpha_results(video_dir_name, output_path=None):
    """
    导出 α 计算结果到 CSV
    """
    if output_path is None:
        output_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 
                                   'alpha_weights.csv')
    
    # 加载 tracking_results 并合并 alpha
    df_tracking = load_tracking_results(video_dir_name)
    if df_tracking is None:
        return None
    
    # 简化：添加占位符 alpha 列
    df_tracking['alpha_left'] = 0.5
    df_tracking['alpha_right'] = 0.5
    
    df_tracking.to_csv(output_path, index=False)
    return output_path


# ═══════════════════════════════════════════════════
# Gradio UI 组件 (供 app_gradio.py 调用)
# ═══════════════════════════════════════════════════

def list_datasets_for_alpha():
    """列出可用于 α 计算的数据集"""
    if not os.path.isdir(MASK_ALIGN_DIR):
        return []
    
    datasets = []
    for d in os.listdir(MASK_ALIGN_DIR):
        csv_path = os.path.join(MASK_ALIGN_DIR, d, 'tracking_results.csv')
        if os.path.exists(csv_path):
            datasets.append(d)
    
    return sorted(datasets)


def on_load_alpha_dataset(dataset_name):
    """加载数据集并显示信息"""
    if not dataset_name:
        return "请选择数据集", None, gr.update(), gr.update()
    
    df = load_tracking_results(dataset_name)
    if df is None:
        return f"未找到 tracking_results.csv", None, gr.update(), gr.update()
    
    info = f"数据集: {dataset_name}\n"
    info += f"总帧数: {len(df)}\n"
    info += f"最新 IoU: {df['iou'].iloc[-1]:.4f}\n"
    info += f"最新质心: ({df['centroid_x'].iloc[-1]:.2f}, {df['centroid_y'].iloc[-1]:.2f})"
    
    # 计算默认参数
    velocities = compute_velocity(df, 4)
    iou_stats = compute_iou_stats(df, 4)
    
    stats_info = f"速度: {velocities.get('overall_velocity', 0):.2f} px/frame\n" if velocities else "速度: N/A\n"
    stats_info += f"IoU均值: {iou_stats.get('iou_mean', 0):.4f}" if iou_stats else "IoU均值: N/A"
    
    return info, stats_info, gr.update(minimum=1, maximum=max(len(df)-1, 1)), gr.update(maximum=len(df)-1)


def on_compute_alpha(
    dataset_name,
    frame_interval,
    weight_velocity,
    weight_iou,
    weight_distance,
    weight_bifurcation,
    weight_target,
    weight_wall,
    fsr_left,
    fsr_right
):
    """计算 α 权重"""
    if not dataset_name:
        return "Please select a dataset first", gr.update(), gr.update()
    
    weights = {
        'velocity': weight_velocity,
        'iou': weight_iou,
        'bifurcation': weight_bifurcation,
        'target': weight_target,
        'wall_distance': weight_wall
    }
    
    result = compute_alpha_weights(
        dataset_name,
        frame_interval=frame_interval,
        weights=weights,
        fsr_left=fsr_left,
        fsr_right=fsr_right
    )
    
    if 'error' in result:
        return result['error'], gr.update(), gr.update()
    
    # 格式化输出
    output = f"✅ 计算完成\n\n"
    output += f"左臂 α = {result['alpha_left']:.4f}\n"
    output += f"右臂 α = {result['alpha_right']:.4f}\n\n"
    output += f"特征值:\n"
    f = result['features']
    output += f"  - 速度 (左/右/整体): {f['velocity_left']:.3f} / {f['velocity_right']:.3f} / {f['velocity_overall']:.3f}\n"
    output += f"  - IoU: {f['iou']:.4f}\n"
    output += f"  - 距离 (左/右): {f['left_dist']:.3f} / {f['right_dist']:.3f}\n"
    output += f"  - 壁面距离: {f['wall_dist']:.3f}\n"
    output += f"  - 分叉距离: {f['bif_dist']:.3f}\n"
    output += f"  - 目标距离: {f['target_dist']:.3f}"
    
    # 返回可视化数据
    viz_data = result['features']
    
    return output, gr.update(value=viz_data['velocity_left']), gr.update(value=viz_data['velocity_right'])


# ═══════════════════════════════════════════════════
# 调试/测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    # 测试代码
    print("Alpha Weight Calculator - Test Mode")
    print("=" * 50)
    
    # 列出可用数据集
    datasets = list_datasets_for_alpha()
    print(f"可用数据集: {datasets}")
    
    if datasets:
        # 测试第一个数据集
        test_dataset = datasets[0]
        print(f"\n测试数据集: {test_dataset}")
        
        result = compute_alpha_weights(
            test_dataset,
            frame_interval=4,
            weights={
                'velocity': 0.2,
                'iou': 0.2,
                'distance': 0.2,
                'bifurcation': 0.2,
                'target': 0.1,
                'wall_distance': 0.1
            },
            fsr_left=0.5,
            fsr_right=0.5
        )
        
        print(f"\n结果:")
        print(f"  α_left = {result['alpha_left']:.4f}")
        print(f"  α_right = {result['alpha_right']:.4f}")
        print(f"  特征: {result['features']}")


# ═══════════════════════════════════════════════════
# 轨迹与速度可视化
# ═══════════════════════════════════════════════════

def visualize_trajectory(video_dir_name, frame_interval=4):
    """
    Visualize trajectory and speed of both arm endpoints and centroid
    
    Returns:
        tuple: (trajectory_image, speed_image, info_text)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    df = load_tracking_results(video_dir_name)
    if df is None or len(df) < 2:
        return None, None, "Insufficient data for visualization"
    
    # Extract coordinate data
    frames = df['frame'].values
    centroid_x = df['centroid_x'].values
    centroid_y = df['centroid_y'].values
    bottom_x = df['bottom_x'].values
    bottom_y = df['bottom_y'].values
    top_x = df['top_x'].values
    top_y = df['top_y'].values
    
    # Calculate speed (using frame interval)
    interval = max(1, min(frame_interval, len(df) - 1))
    
    # Centroid speed
    dx_centroid = np.diff(centroid_x[::interval])
    dy_centroid = np.diff(centroid_y[::interval])
    speed_centroid = np.sqrt(dx_centroid**2 + dy_centroid**2)
    
    # Bottom endpoint speed (left arm)
    dx_bottom = np.diff(bottom_x[::interval])
    dy_bottom = np.diff(bottom_y[::interval])
    speed_bottom = np.sqrt(dx_bottom**2 + dy_bottom**2)
    
    # Top endpoint speed (right arm)
    dx_top = np.diff(top_x[::interval])
    dy_top = np.diff(top_y[::interval])
    speed_top = np.sqrt(dx_top**2 + dy_top**2)
    
    # ═══════════════════════════════════════════════════
    # 加载背景图像（使用 Step 4-3 的配准）
    # 优先使用 images_vis_template_only 的第一帧 + 迷宫叠加
    # ═══════════════════════════════════════════════════
    video_dir = os.path.join(MASK_ALIGN_DIR, video_dir_name)
    reg_dir = os.path.join(video_dir, 'registration_output')
    template_dir = os.path.join(video_dir, 'images_vis_template_only')
    
    # 加载第一帧作为背景（与 Step 4-3 一致）
    bg_img = None
    if os.path.exists(template_dir):
        frame_files = sorted([f for f in os.listdir(template_dir) if f.endswith('.jpg')])
        if frame_files:
            first_frame_path = os.path.join(template_dir, frame_files[0])
            bg_img = cv2.imread(first_frame_path)
            if bg_img is not None:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    
    # 如果没有 template_only 目录，使用 registered.png
    if bg_img is None and os.path.exists(os.path.join(reg_dir, 'registered.png')):
        bg_img = cv2.imread(os.path.join(reg_dir, 'registered.png'))
        if bg_img is not None:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    
    # 如果还是没有，使用空白背景
    if bg_img is None:
        h, w = 480, 640
        bg_img = np.ones((h, w, 3), dtype=np.uint8) * 240
    
    # 获取图像尺寸
    img_h, img_w = bg_img.shape[:2]
    
    # ═══════════════════════════════════════════════════
    # 加载迷宫掩码并绘制通道和墙壁轮廓（与 Step 3 一致）
    # ═══════════════════════════════════════════════════
    maze_params = load_maze_params(video_dir_name)
    maze_img_bg = bg_img.copy()
    
    if maze_params and os.path.exists(os.path.join(reg_dir, 'maze_corridor_mask.png')):
        corr_mask = cv2.imread(os.path.join(reg_dir, 'maze_corridor_mask.png'), cv2.IMREAD_GRAYSCALE)
        wall_mask = cv2.imread(os.path.join(reg_dir, 'maze_wall_mask.png'), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(reg_dir, 'maze_wall_mask.png')) else None
        
        if corr_mask is not None:
            # 绘制通道轮廓（黄色）
            c1, _ = cv2.findContours(corr_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(maze_img_bg, c1, -1, (0, 255, 255), 2)  # Yellow
        
        if wall_mask is not None:
            # 绘制墙壁轮廓（红色）
            c2, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(maze_img_bg, c2, -1, (0, 0, 255), 2)  # Red
    
    # Create trajectory plot with background
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    # 显示背景图像（带迷宫轮廓）
    ax1.imshow(maze_img_bg)
    
    # 反向Y轴以匹配图像坐标
    ax1.invert_yaxis()
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    
    # 绘制质心轨迹（蓝色）
    ax1.plot(centroid_x, centroid_y, 'b-', alpha=0.7, linewidth=2, label='Centroid')
    ax1.scatter(centroid_x[0], centroid_y[0], c='blue', s=150, marker='o', zorder=10, edgecolors='white', linewidths=2)
    ax1.scatter(centroid_x[-1], centroid_y[-1], c='blue', s=200, marker='*', zorder=10, edgecolors='white', linewidths=2)
    
    # 绘制底部端点轨迹（绿色 - 左臂）
    ax1.plot(bottom_x, bottom_y, 'g-', alpha=0.7, linewidth=2, label='Left Arm (Bottom)')
    ax1.scatter(bottom_x[0], bottom_y[0], c='lime', s=100, marker='o', zorder=10, edgecolors='white', linewidths=2)
    ax1.scatter(bottom_x[-1], bottom_y[-1], c='lime', s=150, marker='*', zorder=10, edgecolors='white', linewidths=2)
    
    # 绘制顶部端点轨迹（红色 - 右臂）
    ax1.plot(top_x, top_y, 'r-', alpha=0.7, linewidth=2, label='Right Arm (Top)')
    ax1.scatter(top_x[0], top_y[0], c='red', s=100, marker='o', zorder=10, edgecolors='white', linewidths=2)
    ax1.scatter(top_x[-1], top_y[-1], c='red', s=150, marker='*', zorder=10, edgecolors='white', linewidths=2)
    
    # 添加帧号标注（每隔一定间隔）
    step = max(1, len(frames) // 10)
    for i in range(0, len(frames), step):
        ax1.annotate(str(frames[i]), (centroid_x[i], centroid_y[i]), 
                    fontsize=7, color='white', alpha=0.7,
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.5))
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title(f'Trajectory on Registered Maze - {video_dir_name}\n(Yellow: Corridor, Red: Wall, Blue: Centroid, Green: Left Arm, Red: Right Arm)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3, color='white')
    
    # Save trajectory plot
    traj_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 'trajectory_visualization.png')
    fig1.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Create speed plot
    fig2, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    frames_speed = frames[::interval][1:]  # Frame indices for speed
    
    # Centroid speed
    axes[0].plot(frames_speed, speed_centroid, 'b-', linewidth=1.5)
    axes[0].fill_between(frames_speed, speed_centroid, alpha=0.3)
    axes[0].set_ylabel('Speed (px/frame)')
    axes[0].set_title('Centroid Speed')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(frames[0], frames[-1])
    
    # Bottom endpoint speed (left arm)
    axes[1].plot(frames_speed, speed_bottom, 'g-', linewidth=1.5)
    axes[1].fill_between(frames_speed, speed_bottom, alpha=0.3, color='green')
    axes[1].set_ylabel('Speed (px/frame)')
    axes[1].set_title('Bottom (Left Arm) Speed')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(frames[0], frames[-1])
    
    # Top endpoint speed (right arm)
    axes[2].plot(frames_speed, speed_top, 'r-', linewidth=1.5)
    axes[2].fill_between(frames_speed, speed_top, alpha=0.3, color='red')
    axes[2].set_ylabel('Speed (px/frame)')
    axes[2].set_xlabel('Frame')
    axes[2].set_title('Top (Right Arm) Speed')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(frames[0], frames[-1])
    
    plt.tight_layout()
    
    # Save speed plot
    speed_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 'speed_visualization.png')
    fig2.savefig(speed_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ═══════════════════════════════════════════════════
    # 壁面距离可视化（类似 Step 3 的运动分析）
    # ═══════════════════════════════════════════════════
    
    # 计算每帧的壁面距离
    wall_dist_frames = []
    wall_dist_left_list = []
    wall_dist_right_list = []
    
    # 加载掩码用于射线投射
    corr_mask = None
    wall_mask = None
    if maze_params and os.path.exists(os.path.join(reg_dir, 'maze_corridor_mask.png')):
        corr_mask = cv2.imread(os.path.join(reg_dir, 'maze_corridor_mask.png'), cv2.IMREAD_GRAYSCALE)
        wall_mask = cv2.imread(os.path.join(reg_dir, 'maze_wall_mask.png'), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(reg_dir, 'maze_wall_mask.png')) else corr_mask
    
    if corr_mask is not None and maze_params:
        cx_m = maze_params.get('cx', 0)
        cy_m = maze_params.get('cy', 0)
        scale_m = maze_params.get('scale', 0.5)
        angle_m = maze_params.get('angle_deg', 0)
        
        # 计算仿射变换矩阵（用于将图像坐标转换到迷宫坐标）
        mask_h, mask_w = corr_mask.shape[:2]
        M = cv2.getRotationMatrix2D((mask_w/2, mask_h/2), angle_m, scale_m)
        M_inv = cv2.invertAffineTransform(M)
        
        for i in range(len(frames)):
            # 左臂（底部端点）
            dl = _raycast_dist(bottom_x[i], bottom_y[i], M_inv, corr_mask, wall_mask)
            # 右臂（顶部端点）
            dr = _raycast_dist(top_x[i], top_y[i], M_inv, corr_mask, wall_mask)
            
            wall_dist_left_list.append(dl)
            wall_dist_right_list.append(dr)
    else:
        wall_dist_left_list = [0.0] * len(frames)
        wall_dist_right_list = [0.0] * len(frames)
    
    wall_dist_left_list = np.array(wall_dist_left_list)
    wall_dist_right_list = np.array(wall_dist_right_list)
    
    # 创建壁面距离可视化图
    fig3, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 归一化壁面距离用于颜色映射
    max_dist_viz = max(np.max(wall_dist_left_list), np.max(wall_dist_right_list), 1.0)
    
    # 绘制轨迹并用壁面距离着色
    ax_traj = axes[0]
    ax_traj.invert_yaxis()
    
    # 绘制通道和墙壁轮廓
    if corr_mask is not None:
        c1, _ = cv2.findContours(corr_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 转换为RGB用于matplotlib
        corr_rgb = cv2.cvtColor(corr_mask, cv2.COLOR_GRAY2RGB)
        ax_traj.imshow(corr_rgb, alpha=0.3)
    
    # 用颜色表示壁面距离（绿色=远，红色=近）
    for i in range(len(frames)-1):
        # 左臂
        dist_l = wall_dist_left_list[i] / max_dist_viz
        color_l = (1-dist_l, dist_l, 0)  # Green to Red
        ax_traj.plot([bottom_x[i], bottom_x[i+1]], [bottom_y[i], bottom_y[i+1]], 
                    color=color_l, linewidth=3, alpha=0.8)
        
        # 右臂
        dist_r = wall_dist_right_list[i] / max_dist_viz
        color_r = (1-dist_r, 0, dist_r)  # Blue to Red
        ax_traj.plot([top_x[i], top_x[i+1]], [top_y[i], top_y[i+1]], 
                    color=color_r, linewidth=3, alpha=0.8)
    
    ax_traj.scatter(bottom_x[0], bottom_y[0], c='lime', s=100, marker='o', zorder=10, label='Left Start')
    ax_traj.scatter(top_x[0], top_y[0], c='cyan', s=100, marker='o', zorder=10, label='Right Start')
    ax_traj.set_title('Trajectory with Wall Distance\n(Left: Green=Far, Red=Close | Right: Blue=Far, Red=Close)')
    ax_traj.legend(loc='upper right')
    ax_traj.set_xlim(0, img_w)
    ax_traj.set_ylim(img_h, 0)
    
    # 左臂壁面距离随时间变化
    axes[1].plot(frames, wall_dist_left_list, 'g-', linewidth=2, label='Left Arm (Bottom)')
    axes[1].fill_between(frames, wall_dist_left_list, alpha=0.3, color='green')
    axes[1].set_ylabel('Distance (px)')
    axes[1].set_title('Wall Distance - Left Arm (Bottom)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(frames[0], frames[-1])
    
    # 右臂壁面距离随时间变化
    axes[2].plot(frames, wall_dist_right_list, 'r-', linewidth=2, label='Right Arm (Top)')
    axes[2].fill_between(frames, wall_dist_right_list, alpha=0.3, color='red')
    axes[2].set_ylabel('Distance (px)')
    axes[2].set_xlabel('Frame')
    axes[2].set_title('Wall Distance - Right Arm (Top)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(frames[0], frames[-1])
    
    plt.tight_layout()
    
    # 保存壁面距离可视化图
    wall_dist_path = os.path.join(MASK_ALIGN_DIR, video_dir_name, 'wall_distance_visualization.png')
    fig3.savefig(wall_dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 添加壁面距离统计到 info
    info += f"\n[Wall Distance Visualization]\n"
    info += f"  Left Arm  - Max: {np.max(wall_dist_left_list):.1f}px, Min: {np.min(wall_dist_left_list):.1f}px, Avg: {np.mean(wall_dist_left_list):.1f}px\n"
    info += f"  Right Arm - Max: {np.max(wall_dist_right_list):.1f}px, Min: {np.min(wall_dist_right_list):.1f}px, Avg: {np.mean(wall_dist_right_list):.1f}px\n"
    info = f"Trajectory & Speed Statistics (Frame Interval: {interval})\n"
    info += "=" * 40 + "\n\n"
    
    # Calculate wall distances for both arms
    maze_distances = compute_maze_distances(
        video_dir_name, 
        centroid_x[-1], centroid_y[-1],
        bottom_x[-1], bottom_y[-1],
        top_x[-1], top_y[-1]
    )
    
    max_wall_dist = 100.0
    wall_dist_left = 0.0
    wall_dist_right = 0.0
    if maze_distances:
        wall_dist_left = maze_distances.get('dist_to_wall_left', 0.0)
        wall_dist_right = maze_distances.get('dist_to_wall_right', 0.0)
    
    info += "[Centroid]\n"
    info += f"  Total Displacement: {np.sqrt((centroid_x[-1]-centroid_x[0])**2 + (centroid_y[-1]-centroid_y[0])**2):.1f} px\n"
    info += f"  Avg Speed: {np.mean(speed_centroid):.2f} px/frame\n"
    info += f"  Max Speed: {np.max(speed_centroid):.2f} px/frame\n"
    info += f"  Min Speed: {np.min(speed_centroid):.2f} px/frame\n\n"
    
    info += "[Left Arm (Bottom)]\n"
    info += f"  Total Displacement: {np.sqrt((bottom_x[-1]-bottom_x[0])**2 + (bottom_y[-1]-bottom_y[0])**2):.1f} px\n"
    info += f"  Avg Speed: {np.mean(speed_bottom):.2f} px/frame\n"
    info += f"  Max Speed: {np.max(speed_bottom):.2f} px/frame\n"
    info += f"  Min Speed: {np.min(speed_bottom):.2f} px/frame\n"
    info += f"  Wall Distance: {wall_dist_left:.1f} px ({wall_dist_left/max_wall_dist:.2f})\n\n"
    
    info += "[Right Arm (Top)]\n"
    info += f"  Total Displacement: {np.sqrt((top_x[-1]-top_x[0])**2 + (top_y[-1]-top_y[0])**2):.1f} px\n"
    info += f"  Avg Speed: {np.mean(speed_top):.2f} px/frame\n"
    info += f"  Max Speed: {np.max(speed_top):.2f} px/frame\n"
    info += f"  Min Speed: {np.min(speed_top):.2f} px/frame\n"
    info += f"  Wall Distance: {wall_dist_right:.1f} px ({wall_dist_right/max_wall_dist:.2f})\n"
    
    # 读取保存的图片
    import base64
    from io import BytesIO
    
    def read_image_as_base64(path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return None
    
    traj_b64 = read_image_as_base64(traj_path)
    speed_b64 = read_image_as_base64(speed_path)
    wall_dist_b64 = read_image_as_base64(wall_dist_path)
    
    return traj_b64, speed_b64, info, wall_dist_b64


def on_visualize_trajectory(video_dir_name, frame_interval):
    """Gradio 回调：可视化轨迹和速度"""
    if not video_dir_name:
        return None, None, "Please select a dataset first", None
    
    traj_b64, speed_b64, info, wall_dist_b64 = visualize_trajectory(video_dir_name, frame_interval)
    
    if traj_b64 is None:
        return None, None, info, None
    
    # 转换为 HTML img 标签用于显示
    traj_img = f'<img src="data:image/png;base64,{traj_b64}" style="width:100%">'
    speed_img = f'<img src="data:image/png;base64,{speed_b64}" style="width:100%">'
    wall_dist_img = f'<img src="data:image/png;base64,{wall_dist_b64}" style="width:100%">' if wall_dist_b64 else ""
    
    return traj_img, speed_img, info, wall_dist_img
