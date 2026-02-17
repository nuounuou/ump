#!/usr/bin/env python3
"""
迷宫配准 + 参数提取 — Step 4-3 / 4-4 后端逻辑

被 app_gradio.py 导入使用，所有 Gradio 回调函数在此定义。
后续新增的处理逻辑也写在这里，保持 app_gradio.py 精简。
"""

import os
import json
import glob
import numpy as np
import cv2
import gradio as gr

# ───────── 路径常量 ─────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAZE_PATH = os.path.join(SCRIPT_DIR, 'maze1.png')
MASK_ALIGN_DIR = os.path.join(SCRIPT_DIR, 'mask_align_sam2_dataset')


# ───────── 工具函数 ─────────

def list_mask_align_datasets():
    """列出 mask_align_sam2_dataset/ 下含 images_vis_template_only 的子文件夹"""
    if not os.path.isdir(MASK_ALIGN_DIR):
        return []
    return sorted([
        d for d in os.listdir(MASK_ALIGN_DIR)
        if os.path.isdir(os.path.join(MASK_ALIGN_DIR, d, 'images_vis_template_only'))
    ])


def _params_path(dataset_name):
    """配准参数 JSON 路径"""
    return os.path.join(MASK_ALIGN_DIR, dataset_name,
                        'registration_output', 'maze_params.json')


def _build_affine(mw, mh, cx, cy, scale, angle_deg):
    """构建 2×3 仿射矩阵: 以迷宫中心为原点 → 缩放 → 旋转 → 平移到 (cx, cy)"""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    mx, my = mw / 2.0, mh / 2.0
    M = np.array([
        [scale * cos_t, -scale * sin_t, cx - scale * (cos_t * mx - sin_t * my)],
        [scale * sin_t,  scale * cos_t, cy - scale * (sin_t * mx + cos_t * my)],
    ], dtype=np.float64)
    return M


def _get_maze_channels():
    """加载迷宫图原始通道 → (bgr, alpha, gray) 或 None"""
    if not os.path.exists(MAZE_PATH):
        return None
    maze_raw = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if maze_raw is None:
        return None
    if len(maze_raw.shape) == 3 and maze_raw.shape[2] == 4:
        bgr, a = maze_raw[:, :, :3], maze_raw[:, :, 3]
    else:
        bgr = maze_raw if len(maze_raw.shape) == 3 else cv2.cvtColor(maze_raw, cv2.COLOR_GRAY2BGR)
        a = np.full(bgr.shape[:2], 255, np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, a, gray


def _load_maze_rgb():
    """加载迷宫图为 RGB numpy (用于预览)"""
    if not os.path.exists(MAZE_PATH):
        return None
    m = cv2.imread(MAZE_PATH, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if len(m.shape) == 3 and m.shape[2] == 4:
        a = m[:, :, 3:4].astype(np.float32) / 255
        bgr = m[:, :, :3].astype(np.float32)
        rgb = (bgr * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


def _overlay(cam_bgr, cx, cy, scale, angle_deg, alpha_pct):
    """在相机 BGR 图上叠加迷宫 (浅红色)，返回 RGB numpy"""
    md = _get_maze_channels()
    if md is None:
        return cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
    bgr, a, _ = md
    mh, mw = bgr.shape[:2]
    h, w = cam_bgr.shape[:2]
    M = _build_affine(mw, mh, cx, cy, scale, angle_deg)
    w_alpha = cv2.warpAffine(a, M, (w, h))
    mask_f = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_pct / 100.0)
    # 浅红色叠加 (BGR: 50,80,255)
    overlay = np.zeros_like(cam_bgr)
    overlay[:] = (50, 80, 255)
    result = np.clip(
        cam_bgr.astype(np.float32) * (1 - mask_f) + overlay.astype(np.float32) * mask_f,
        0, 255,
    ).astype(np.uint8)
    # 红十字标记中心
    cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def _first_cam(dataset_name):
    """获取第一帧 BGR + 总帧数"""
    vis_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'images_vis_template_only')
    imgs = sorted(glob.glob(os.path.join(vis_dir, '*.jpg')))
    if not imgs:
        return None, 0
    cam = cv2.imread(imgs[0])
    return cam, len(imgs)


# ═══════════════════════════════════════════════════
#  Step 4-3  迷宫配准 — Gradio 回调
# ═══════════════════════════════════════════════════

def on_load(dataset_name):
    """加载数据集 + 尝试读取已保存的参数 JSON"""
    if not dataset_name:
        return None, None, None, None, gr.update(), gr.update(), "请选择数据集"

    cam_bgr, n_imgs = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, None, None, None, gr.update(), gr.update(), \
            f"未找到图像: {os.path.join(MASK_ALIGN_DIR, dataset_name, 'images_vis_template_only')}"

    img_h, img_w = cam_bgr.shape[:2]
    cx, cy = img_w / 2.0, img_h / 2.0
    scale_val, angle_val = 0.5, 0.0

    # 尝试加载已保存参数
    pf = _params_path(dataset_name)
    loaded = False
    if os.path.exists(pf):
        try:
            with open(pf) as f:
                p = json.load(f)
            cx = p.get('cx', cx)
            cy = p.get('cy', cy)
            scale_val = p.get('scale', scale_val)
            angle_val = p.get('angle_deg', angle_val)
            loaded = True
        except Exception:
            pass

    overlay_rgb = _overlay(cam_bgr, cx, cy, scale_val, angle_val, 50)
    maze_rgb = _load_maze_rgb()

    info = f"已加载 {n_imgs} 帧, 尺寸 {img_w}×{img_h}"
    info += f"\n{'✅ 已加载参数: ' + pf if loaded else '(无已保存参数, 使用默认值)'}"
    if maze_rgb is not None:
        info += f"\n迷宫图: {MAZE_PATH}"
    else:
        info += f"\n⚠️ 迷宫图不存在: {MAZE_PATH}"

    return (
        maze_rgb, overlay_rgb,
        cx, cy,
        gr.update(value=scale_val),
        gr.update(value=angle_val),
        info,
    )


def on_click(cx_old, cy_old, dataset_name, scale, angle, alpha,
             evt: gr.SelectData):
    """点击叠加预览图 → 设置迷宫中心"""
    x, y = evt.index
    if not dataset_name:
        return None, cx_old, cy_old, "请先加载数据"
    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, cx_old, cy_old, "未找到图像"
    ov = _overlay(cam_bgr, x, y, scale, angle, alpha)
    return ov, float(x), float(y), \
        f"center=({x}, {y}), scale={scale:.3f}, angle={angle:.1f}°"


def on_slider(cx, cy, dataset_name, scale, angle, alpha):
    """滑条变化 → 刷新叠加预览"""
    if not dataset_name or cx is None or cy is None:
        return None, "请先加载数据并点击设置位置"
    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, "未找到图像"
    ov = _overlay(cam_bgr, cx, cy, scale, angle, alpha)
    return ov, f"center=({cx:.0f}, {cy:.0f}), scale={scale:.3f}, angle={angle:.1f}°"


def save_params(dataset_name, cx, cy, scale, angle):
    """保存配准参数 → JSON"""
    if not dataset_name:
        return "请先选择数据集"
    if cx is None or cy is None:
        return "请先点击设置位置"
    out_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'registration_output')
    os.makedirs(out_dir, exist_ok=True)
    params = dict(cx=float(cx), cy=float(cy),
                  scale=float(scale), angle_deg=float(angle),
                  maze_path=MAZE_PATH)
    pf = _params_path(dataset_name)
    with open(pf, 'w') as f:
        json.dump(params, f, indent=2)
    return f"✅ 参数已保存: {pf}\n{json.dumps(params, indent=2)}"


def apply_registration(dataset_name, cx, cy, scale, angle, alpha):
    """应用配准 → 生成墙壁/通道 mask + 可视化 + 保存参数"""
    if not dataset_name:
        return None, "请先选择数据集"
    if cx is None or cy is None:
        return None, "请先点击设置位置"
    md = _get_maze_channels()
    if md is None:
        return None, f"迷宫图不存在: {MAZE_PATH}"
    bgr, a, gray = md
    mh, mw = bgr.shape[:2]

    cam_bgr, _ = _first_cam(dataset_name)
    if cam_bgr is None:
        return None, "未找到图像"
    h, w = cam_bgr.shape[:2]

    M = _build_affine(mw, mh, cx, cy, scale, angle)

    # 墙壁 / 通道 mask
    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpAffine(wall_src, M, (w, h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpAffine(corr_src, M, (w, h)) > 128).astype(np.uint8) * 255

    # 保存
    out_dir = os.path.join(MASK_ALIGN_DIR, dataset_name, 'registration_output')
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "maze_wall_mask.png"), wall_mask)
    cv2.imwrite(os.path.join(out_dir, "maze_corridor_mask.png"), corr_mask)
    # 兼容 Step 3 分析代码: 保存为 3×3 齐次矩阵
    H = np.vstack([M, [0, 0, 1]])
    np.save(os.path.join(out_dir, "homography.npy"), H)

    # 叠加可视化
    overlay_rgb = _overlay(cam_bgr, cx, cy, scale, angle, alpha)
    cv2.imwrite(os.path.join(out_dir, "registered.png"),
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    # 自动保存参数 JSON
    params = dict(cx=float(cx), cy=float(cy),
                  scale=float(scale), angle_deg=float(angle),
                  maze_path=MAZE_PATH)
    with open(_params_path(dataset_name), 'w') as f:
        json.dump(params, f, indent=2)

    info = (
        f"✅ 配准完成!\n"
        f"输出: {out_dir}\n"
        f"  ├── maze_wall_mask.png\n"
        f"  ├── maze_corridor_mask.png\n"
        f"  ├── homography.npy (3×3)\n"
        f"  ├── registered.png\n"
        f"  └── maze_params.json"
    )
    return overlay_rgb, info


# ═══════════════════════════════════════════════════
#  Step 4-4  参数提取与处理 (后续在此扩展)
# ═══════════════════════════════════════════════════

# TODO: 在这里添加 Step 4-4 的处理函数
# 例如:
# def analyze_trajectory(dataset_name, step, progress=gr.Progress()):
#     ...
