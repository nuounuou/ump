#!/usr/bin/env python3
"""
SAM2 Video Segmentation → YOLO Dataset — Gradio Web UI

运行：
cd /home/nuounuou/sam2/notebooks && python app_gradio.py

然后你的电脑):
SSH 端口转发:ssh -L 7860:localhost:7860 nuounuou@172.26.211.82

如果出现端口占用,杀死进程: kill $(lsof -t -i:7860) 2>/dev/null
重新运行:cd /home/nuounuou/sam2/notebooks && python app_gradio.py
"""

import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import gradio as gr
import traceback
import csv
import glob
import json
from pathlib import Path

# ───────── 路径设置 ─────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam2.build_sam import build_sam2_video_predictor
import app_template_align as ta
import maze_processing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'video_to_img')
VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'videos')
YOLO_DIR = os.path.join(SCRIPT_DIR, 'yolo')
YOLO_DATASET_DIR = os.path.join(SCRIPT_DIR, 'yolo_dataset')
SAM2_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'sam2.1_hiera_tiny.pt')
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_t.yaml'

# ───────── SAM2 模型全局缓存 ─────────
_predictor = None
_device = None


def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device('mps')
        else:
            _device = torch.device('cpu')
    return _device


def get_predictor():
    global _predictor
    if _predictor is None:
        device = get_device()
        print(f'[SAM2] Loading model on {device} ...')
        _predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        print('[SAM2] Model loaded.')
    return _predictor


# ───────── 工具函数─────────

def mask_to_yolo_seg(mask, img_w, img_h, simplify_tolerance=2.0):
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None
    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    mask_h, mask_w = mask_2d.shape
    if mask_h != img_h or mask_w != img_w:
        mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest = max(contours, key=cv2.contourArea)
    if simplify_tolerance > 0:
        epsilon = simplify_tolerance * cv2.arcLength(largest, True) / 100.0
        largest = cv2.approxPolyDP(largest, epsilon, True)
    if len(largest) < 3:
        return None
    polygon = largest.reshape(-1, 2).astype(np.float32)
    polygon[:, 0] /= img_w
    polygon[:, 1] /= img_h
    return polygon.flatten().tolist()


def mask_to_yolo_bbox(mask, img_w, img_h):
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None
    mask_h, mask_w = mask_2d.shape
    coords = np.column_stack(np.where(mask_2d > 0))
    if len(coords) == 0:
        return None
    y_coords, x_coords = coords[:, 0], coords[:, 1]
    x_min_mask, x_max_mask = float(x_coords.min()), float(x_coords.max())
    y_min_mask, y_max_mask = float(y_coords.min()), float(y_coords.max())
    if mask_h != img_h or mask_w != img_w:
        scale_x = img_w / mask_w
        scale_y = img_h / mask_h
        x_min = x_min_mask * scale_x
        x_max = x_max_mask * scale_x
        y_min = y_min_mask * scale_y
        y_max = y_max_mask * scale_y
    else:
        x_min, x_max = x_min_mask, x_max_mask
        y_min, y_max = y_min_mask, y_max_mask
    return int(x_min), int(y_min), int(x_max), int(y_max)


# ───────── 视频切帧 ─────────

def list_video_files():
    """列出 videos/ 下所有视频文件"""
    if not os.path.isdir(VIDEOS_DIR):
        return []
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    return sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(exts)])


def on_extract_frames(video_file, frame_interval):
    """从视频中抽帧，保存到 video_to_img/"""
    if not video_file:
        return "请先选择视频文件", gr.update()
    video_path = os.path.join(VIDEOS_DIR, video_file)
    output_name = Path(video_file).stem
    output_path = os.path.join(BASE_DIR, output_name)
    os.makedirs(output_path, exist_ok=True)
    frame_interval = max(1, int(frame_interval))
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='not(mod(n,{frame_interval}))'",
        "-vsync", "0", "-q:v", "2", "-start_number", "0",
        os.path.join(output_path, "%05d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore')
        n = len([f for f in os.listdir(output_path) if f.lower().endswith('.jpg')])
        new_dirs = list_video_dirs()
        return (f"切帧完成: {output_name}, 共 {n} 帧 (每 {frame_interval} 帧取 1 帧)\n"
                f"保存至: {output_path}"),\
               gr.update(choices=new_dirs, value=output_name)
    except Exception as e:
        return f"切帧失败: {e}", gr.update()


# ───────── 帧/目录管理 ─────────

def list_video_dirs():
    """列出 video_to_img 下所有包含 jpg 的子文件夹"""
    dirs = []
    if not os.path.isdir(BASE_DIR):
        return dirs
    for d in sorted(os.listdir(BASE_DIR)):
        full = os.path.join(BASE_DIR, d)
        if os.path.isdir(full):
            jpgs = [f for f in os.listdir(full) if f.lower().endswith(('.jpg', '.jpeg'))]
            if jpgs:
                dirs.append(d)
    return dirs


def get_sorted_frame_names(video_dir_name):
    """获取某个视频目录下所有排序后的帧文件名"""
    d = os.path.join(BASE_DIR, video_dir_name)
    names = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg'))]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return names


def load_frame_image(video_dir_name, frame_idx):
    """加载指定帧图片，返回 PIL Image"""
    names = get_sorted_frame_names(video_dir_name)
    if not names:
        return None
    frame_idx = max(0, min(frame_idx, len(names) - 1))
    path = os.path.join(BASE_DIR, video_dir_name, names[frame_idx])
    return Image.open(path).convert('RGB')


# ───────── 图片上绘制标记点 ─────────

def draw_points_on_image(pil_img, points, labels):
    """在图片上绘制选中的点，正样本=绿色，负样本=红色"""
    img_draw = pil_img.copy()
    draw = ImageDraw.Draw(img_draw)
    r = 6
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        x, y = int(pt[0]), int(pt[1])
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        outline = (255, 255, 255)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)
        # 星号标记
        draw.line([(x - r - 2, y), (x + r + 2, y)], fill=outline, width=1)
        draw.line([(x, y - r - 2), (x, y + r + 2)], fill=outline, width=1)
        # 编号
        draw.text((x + r + 4, y - r), str(i + 1), fill='white')
    return img_draw


def draw_mask_overlay(pil_img, mask, alpha=0.45):
    """在图片上叠加 mask（红色半透明）"""
    img_np = np.array(pil_img).copy()
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return pil_img
    # 将 mask resize 到图片尺寸
    h, w = img_np.shape[:2]
    mh, mw = mask_2d.shape
    if mh != h or mw != w:
        mask_2d = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    # 红色叠加
    overlay = img_np.copy()
    overlay[mask_2d > 0] = [255, 50, 50]
    img_np = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    # 轮廓
    contours, _ = cv2.findContours(
        (mask_2d > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_np, contours, -1, (0, 255, 0), 2)
    return Image.fromarray(img_np)


# ───────── Gradio 回调函数 ─────────

def on_video_dir_change(video_dir_name):
    """切换视频目录时：重置一切，显示第 0 帧"""
    if not video_dir_name:
        return None, 0, gr.update(maximum=0), "请选择视频目录", [], [], None
    names = get_sorted_frame_names(video_dir_name)
    total = len(names)
    img = load_frame_image(video_dir_name, 0)
    info = f"📂 {video_dir_name} — 共 {total} 帧，图片尺寸: {img.size[0]}x{img.size[1]}"
    return img, 0, gr.update(maximum=max(total - 1, 0)), info, [], [], None


def on_frame_change(video_dir_name, frame_idx, points_state, labels_state):
    """切换帧时：重新绘制标记点"""
    if not video_dir_name:
        return None
    img = load_frame_image(video_dir_name, frame_idx)
    if img is None:
        return None
    if points_state:
        img = draw_points_on_image(img, points_state, labels_state)
    return img


def on_image_click(video_dir_name, frame_idx, point_type, points_state, labels_state, evt: gr.SelectData):
    """用户在图片上点击 → 添加一个标记点"""
    if not video_dir_name:
        return None, points_state, labels_state, "请先选择视频目录"

    # evt.index 在 Gradio Image 组件中是 [x, y]（像素坐标）
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "正样本 (前景)" else 0

    points_state.append([x, y])
    labels_state.append(label)

    # 重新绘制
    img = load_frame_image(video_dir_name, frame_idx)
    img = draw_points_on_image(img, points_state, labels_state)

    # 生成点列表文本
    lines = []
    for i, (pt, lbl) in enumerate(zip(points_state, labels_state)):
        tag = "正样本" if lbl == 1 else "负样本"
        lines.append(f"  {i + 1}. ({int(pt[0])}, {int(pt[1])})  {tag}")
    info = f"已选 {len(points_state)} 个点：\n" + "\n".join(lines)

    return img, points_state, labels_state, info


def on_clear_points(video_dir_name, frame_idx):
    """清除所有选点"""
    img = load_frame_image(video_dir_name, frame_idx) if video_dir_name else None
    return img, [], [], "已清除所有点"


def on_undo_point(video_dir_name, frame_idx, points_state, labels_state):
    """撤销最后一个点"""
    if points_state:
        points_state.pop()
        labels_state.pop()
    img = load_frame_image(video_dir_name, frame_idx) if video_dir_name else None
    if img is not None and points_state:
        img = draw_points_on_image(img, points_state, labels_state)

    if points_state:
        lines = []
        for i, (pt, lbl) in enumerate(zip(points_state, labels_state)):
            tag = "正样本" if lbl == 1 else "负样本"
            lines.append(f"  {i + 1}. ({int(pt[0])}, {int(pt[1])})  {tag}")
        info = f"已选 {len(points_state)} 个点：\n" + "\n".join(lines)
    else:
        info = "已清除所有点"
    return img, points_state, labels_state, info


def on_preview_mask(video_dir_name, frame_idx, points_state, labels_state):
    """预览单帧 mask"""
    if not video_dir_name:
        return None, "请先选择视频目录"
    if not points_state:
        return None, "请先在图片上点击选择至少一个标记点"

    try:
        predictor = get_predictor()
        video_path = os.path.join(BASE_DIR, video_dir_name)
        inference_state = predictor.init_state(video_path=video_path)

        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=1,
            points=points_np,
            labels=labels_np,
        )

        # 取第一个目标的 mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()

        img = load_frame_image(video_dir_name, frame_idx)
        result_img = draw_mask_overlay(img, mask)
        result_img = draw_points_on_image(result_img, points_state, labels_state)

        # 释放
        predictor.reset_state(inference_state)

        mask_pixels = int(np.sum(np.squeeze(mask) > 0))
        return result_img, f"Mask 预览成功！mask 像素数: {mask_pixels}"
    except Exception as e:
        traceback.print_exc()
        return None, f"预览失败: {str(e)}"


def on_export_yolo(video_dir_name, frame_idx, points_state, labels_state, class_id, progress=gr.Progress()):
    """运行 SAM2 全序列传播 → 导出 YOLO 数据集"""
    if not video_dir_name:
        return "请先选择视频目录"
    if not points_state:
        return "请先在图片上点击选择至少一个标记点"

    try:
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            print(msg)

        predictor = get_predictor()
        video_path = os.path.join(BASE_DIR, video_dir_name)

        log(f"视频目录: {video_path}")
        log(f"设备: {get_device()}")
        log(f"标记帧: {frame_idx}, 选点数: {len(points_state)}, CLASS_ID: {class_id}")

        # 初始化
        progress(0.0, desc="初始化 SAM2 ...")
        inference_state = predictor.init_state(video_path=video_path)

        points_np = np.array(points_state, dtype=np.float32)
        labels_np = np.array(labels_state, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=1,
            points=points_np,
            labels=labels_np,
        )
        log(f"已添加提示点，目标数: {len(out_obj_ids)}")

        # 正向传播
        progress(0.1, desc="正向传播中 ...")
        video_segments = {}
        for out_frame_idx, out_obj_ids_prop, out_mask_logits_prop in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits_prop[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids_prop)
            }
        log(f"传播完成: {len(video_segments)} 帧有 mask")

        # 释放
        predictor.reset_state(inference_state)

        # 导出 YOLO 数据集
        progress(0.4, desc="导出 YOLO 数据集 ...")
        output_dir = os.path.join(SCRIPT_DIR, 'yolo_dataset', video_dir_name)
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        vis_dir = os.path.join(output_dir, 'images_vis')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        frame_names = get_sorted_frame_names(video_dir_name)
        total_frames = len(frame_names)
        saved_count = 0
        skipped_count = 0
        class_id_int = int(class_id)

        for fi in range(total_frames):
            progress(0.4 + 0.55 * (fi / total_frames), desc=f"处理帧 {fi + 1}/{total_frames} ...")

            img_path = os.path.join(video_path, frame_names[fi])
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size

            label_lines = []
            bboxes_px = []

            if fi in video_segments:
                for obj_id, mask in video_segments[fi].items():
                    mask_2d = np.squeeze(mask)
                    if mask_2d.ndim != 2 or not np.any(mask_2d > 0):
                        continue
                    polygon = mask_to_yolo_seg(mask_2d, img_w, img_h, simplify_tolerance=2.0)
                    if polygon is None or len(polygon) < 6:
                        continue
                    polygon_array = np.array(polygon)
                    if np.any(polygon_array < 0) or np.any(polygon_array > 1):
                        continue
                    polygon_str = ' '.join([f'{coord:.6f}' for coord in polygon])
                    label_lines.append(f"{class_id_int} {polygon_str}\n")
                    bbox = mask_to_yolo_bbox(mask_2d, img_w, img_h)
                    if bbox is not None:
                        bboxes_px.append(bbox)

            # 保存图片
            img_name = f'{fi:05d}.jpg'
            shutil.copy(img_path, os.path.join(images_dir, img_name))

            # 保存标注
            label_name = f'{fi:05d}.txt'
            with open(os.path.join(labels_dir, label_name), 'w', encoding='utf-8') as f:
                if label_lines:
                    f.writelines(label_lines)

            if not label_lines:
                skipped_count += 1

            # 可视化
            fig = plt.figure(figsize=(6, 4), dpi=100)
            ax = plt.gca()
            ax.axis('off')
            ax.set_title(f'frame {fi}')
            ax.imshow(img)
            if fi in video_segments:
                for obj_id, mask in video_segments[fi].items():
                    mask_2d = np.squeeze(mask)
                    if mask_2d.ndim != 2 or not np.any(mask_2d > 0):
                        continue
                    # 半透明 mask
                    m = mask_2d.astype(np.float32)
                    h, w = m.shape
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
                    rgba[..., 0] = 1.0
                    rgba[..., 1] = 0.2
                    rgba[..., 2] = 0.2
                    rgba[..., 3] = m * 0.45
                    ax.imshow(rgba)
                    for bbox in bboxes_px:
                        x_min, y_min, x_max, y_max = bbox
                        rect = Rectangle(
                            (x_min, y_min), x_max - x_min + 1, y_max - y_min + 1,
                            fill=False, linewidth=2, edgecolor='green',
                        )
                        ax.add_patch(rect)
            plt.savefig(os.path.join(vis_dir, img_name), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            saved_count += 1

        progress(1.0, desc="完成！")

        log(f"\n{'=' * 50}")
        log(f"   导出完成！")
        log(f"   总帧数: {total_frames}")
        log(f"   有 mask 的帧: {saved_count - skipped_count}")
        log(f"   无 mask 的帧: {skipped_count}")
        log(f"   输出目录: {output_dir}")
        log(f"     ├── images/     (原图)")
        log(f"     ├── labels/     (YOLO 实例分割label)")
        log(f"     └── images_vis/ (可视化)")

        return "\n".join(log_lines)

    except Exception as e:
        traceback.print_exc()
        return f"导出失败: {str(e)}\n{traceback.format_exc()}"


# ───────── Step 2: YOLO 训练 ─────────

def list_yolo_datasets():
    """列出 yolo_dataset/ 下可用的数据集"""
    if not os.path.isdir(YOLO_DATASET_DIR):
        return []
    return sorted([
        d for d in os.listdir(YOLO_DATASET_DIR)
        if os.path.isdir(os.path.join(YOLO_DATASET_DIR, d, 'images'))
    ])


def prepare_and_train_yolo(dataset_name, model_name, epochs, batch_size, imgsz,
                           class_name, val_ratio, progress=gr.Progress()):
    """准备数据集 + 训练 YOLO"""
    if not dataset_name:
        return "先选择数据集", ""
    try:
        import random
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            print(msg)

        # ── 1. 准备数据集 (split train/val) ──
        progress(0.0, desc="准备数据集-训练/验证集/测试集")
        dataset_root = Path(YOLO_DATASET_DIR) / dataset_name
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"

        train_img = dataset_root / "train" / "images"
        train_lbl = dataset_root / "train" / "labels"
        val_img = dataset_root / "val" / "images"
        val_lbl = dataset_root / "val" / "labels"
        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()

        image_files = [f.stem for f in images_dir.glob("*.jpg")]
        log(f"找到 {len(image_files)} 张图片")

        # 检测类别
        classes = set()
        for lf in labels_dir.glob("*.txt"):
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        classes.add(int(line.strip().split()[0]))
        nc = max(len(classes), 1)
        log(f"检测到 {nc} 个类别: {sorted(classes)}")

        # 分割
        rng = random.Random(42)
        rng.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        log(f"训练集: {len(train_files)}, 验证集: {len(val_files)}")

        for stem in train_files:
            shutil.copy2(images_dir / f"{stem}.jpg", train_img / f"{stem}.jpg")
            src_lbl = labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, train_lbl / f"{stem}.txt")

        for stem in val_files:
            shutil.copy2(images_dir / f"{stem}.jpg", val_img / f"{stem}.jpg")
            src_lbl = labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, val_lbl / f"{stem}.txt")


        # ── 2. 更新 dataset.yaml ──
        progress(0.05, desc="更新dataset.yaml")
        rel_path = os.path.relpath(dataset_root, YOLO_DIR)
        names_block = "\n".join([f"  {c}: {class_name}" for c in sorted(classes)]) if classes else f"  0: {class_name}"
        yaml_content = (
            f"path: {rel_path}\n"
            f"train: train/images\n"
            f"val: val/images\n\n"
            f"nc: {nc}\n\n"
            f"names:\n{names_block}\n"
        )
        yaml_path = Path(YOLO_DIR) / "dataset.yaml"
        yaml_path.write_text(yaml_content, encoding='utf-8')
        log(f"已更新 {yaml_path}")
        log(f"dataset.yaml 内容:\n{yaml_content}")

        # ── 3. 训练 ──
        progress(0.1, desc="开始 YOLO 训练")
        from ultralytics import YOLO

        os.chdir(YOLO_DIR)
        model = YOLO(model_name)
        log(f"模型: {model_name}, epochs={int(epochs)}, batch={int(batch_size)}, imgsz={int(imgsz)}")

        results = model.train(
            data=str(yaml_path),
            task="segment",
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch_size),
            device=0 if torch.cuda.is_available() else "cpu",
            workers=4,
            project="runs/segment",
            name=f"seg_{dataset_name}",
            save=True,
            save_period=10,
            plots=True,
            val=True,
            patience=50,
        )

        progress(0.95, desc="保存模型")
        best_pt_src = Path(results.save_dir) / "weights" / "best.pt"
        best_pt_dst = Path(YOLO_DIR) / "best.pt"
        if best_pt_src.exists():
            shutil.copy2(best_pt_src, best_pt_dst)
            log(f"\nbest.pt 已复制到: {best_pt_dst}")
        else:
            best_pt_dst = best_pt_src
            log(f"\nbest.pt 路径: {best_pt_src}")

        progress(1.0, desc="训练完成")
        log(f"\n{'=' * 50}")
        log(f"训练完成!")
        log(f"best.pt: {best_pt_dst}")
        log(f"训练目录: {results.save_dir}")

        # scp 指令
        scp_cmd = f"scp nuounuou@172.26.211.82:{best_pt_dst} ./" 
        result_text = (
            f"best.pt 路径:\n{best_pt_dst}\n\n"
            f"拷贝到本地 (scp):\n{scp_cmd} (拷贝回去的路径自己改一下!bro)\n\n"
            f"训练目录:\n{results.save_dir}"
        )
        return "\n".join(log_lines), result_text

    except Exception as e:
        traceback.print_exc()
        return f"训练失败: {e}\n{traceback.format_exc()}", ""


# ───────── Step 3: Shared Control Dataset ─────────

SC_DIR = os.path.join(PROJECT_ROOT, 'shared_control_dataset')
SC_OUT_DIR = os.path.join(SC_DIR, 'registration_output')
RAY_MAX = 200


def _load_maze_rgb(maze_path):
    """加载迷宫图为 RGB numpy"""
    m = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.shape[2] == 4:
        a = m[:, :, 3:4].astype(np.float32) / 255
        bgr = m[:, :, :3].astype(np.float32)
        rgb = (bgr * a + 255 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


def _draw_pts(img, pts, colors=((255,0,0),(0,200,0),(0,0,255),(255,165,0))):
    vis = img.copy()
    h, w = vis.shape[:2]
    r = max(5, min(h, w) // 60)
    for i, (x, y) in enumerate(pts):
        c = colors[i % len(colors)]
        cv2.circle(vis, (int(x), int(y)), r, c, -1)
        cv2.circle(vis, (int(x), int(y)), r, (255,255,255), 2)
        cv2.putText(vis, str(i+1), (int(x)+r+2, int(y)+r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return vis


def sc_on_dataset_change(dataset_name):
    """选择数据集 → 加载第一帧 + 复制数据 + 加载迷宫图"""
    if not dataset_name:
        return None, None, None, "请选择数据集"
    src = os.path.join(YOLO_DATASET_DIR, dataset_name)
    # 复制 images_vis & labels 到 shared_control_dataset
    for sub in ("images_vis", "labels"):
        dst = os.path.join(SC_DIR, sub)
        src_sub = os.path.join(src, sub)
        if os.path.isdir(src_sub):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src_sub, dst)
    # 加载第一帧
    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    if not imgs:
        return None, None, None, "未找到图像"
    cam = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
    img_h, img_w = cam.shape[:2]
    n_labels = len(glob.glob(os.path.join(SC_DIR, "labels", "*.txt")))
    # 加载迷宫图
    maze_path = os.path.join(SC_DIR, "maze1.png")
    maze_img = _load_maze_rgb(maze_path) if os.path.exists(maze_path) else None
    info = f"已加载 {len(imgs)} 帧, {n_labels} 标签, 尺寸 {img_w}x{img_h}"
    if maze_img is not None:
        info += f"\n迷宫图已加载: {maze_path}"
    else:
        info += f"\n迷宫图不存在: {maze_path}"
    return maze_img, cam, None, info


def sc_load_maze():
    """加载固定路径的迷宫图"""
    maze_path = os.path.join(SC_DIR, "maze1.png")
    if not os.path.exists(maze_path):
        return None, f"迷宫图不存在: {maze_path}"
    rgb = _load_maze_rgb(maze_path)
    return rgb, f"迷宫图已加载: {maze_path}"


def sc_click_maze(pts, evt: gr.SelectData):
    x, y = evt.index
    if len(pts) >= 4:
        pts = []
    pts.append([x, y])
    maze_path = os.path.join(SC_DIR, "maze1.png")
    rgb = _load_maze_rgb(maze_path)
    info = f"迷宫点 ({len(pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in pts)
    return _draw_pts(rgb, pts) if rgb is not None else None, pts, info


def sc_click_cam(pts, evt: gr.SelectData):
    x, y = evt.index
    if len(pts) >= 4:
        pts = []
    pts.append([x, y])
    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    cam = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB) if imgs else None
    info = f"相机点 ({len(pts)}/4): " + ", ".join(f"({p[0]},{p[1]})" for p in pts)
    return _draw_pts(cam, pts) if cam is not None else None, pts, info


def sc_register(maze_pts, cam_pts, alpha_val):
    """配准迷宫"""
    if len(maze_pts) != 4 or len(cam_pts) != 4:
        return None, "请在两张图上各点击 4 个对应点"
    os.makedirs(SC_OUT_DIR, exist_ok=True)
    maze_path = os.path.join(SC_DIR, "maze1.png")
    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    if not imgs:
        return None, "未找到图像"

    src, dst = np.float32(maze_pts), np.float32(cam_pts)
    H, _ = cv2.findHomography(src, dst)

    maze_raw = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    cam_bgr = cv2.imread(imgs[0])
    h, w = cam_bgr.shape[:2]
    bgr = maze_raw[:, :, :3]
    a = maze_raw[:, :, 3] if maze_raw.shape[2] == 4 else np.full(maze_raw.shape[:2], 255, np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # warp
    w_alpha = cv2.warpPerspective(a, H, (w, h))
    mask = (w_alpha.astype(np.float32) / 255.0)[..., None] * (alpha_val / 100.0)
    overlay = np.zeros_like(cam_bgr); overlay[:] = (0, 128, 128)
    result = np.clip(cam_bgr * (1 - mask) + overlay * mask, 0, 255).astype(np.uint8)

    # 保存墙壁/通道 mask + 单应性
    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpPerspective(wall_src, H, (w, h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpPerspective(corr_src, H, (w, h)) > 128).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(SC_OUT_DIR, "maze_wall_mask.png"), wall_mask)
    cv2.imwrite(os.path.join(SC_OUT_DIR, "maze_corridor_mask.png"), corr_mask)
    cv2.imwrite(os.path.join(SC_OUT_DIR, "registered.png"), result)
    np.save(os.path.join(SC_OUT_DIR, "homography.npy"), H)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb, f"配准完成!\nH =\n{H}"


def _raycast(cx, cy, dx, dy, mask, hit_val, max_d=RAY_MAX):
    h, w = mask.shape
    for s in range(1, max_d):
        x, y = int(round(cx + dx*s)), int(round(cy + dy*s))
        if x < 0 or x >= w or y < 0 or y >= h:
            return float(s)
        if mask[y, x] == hit_val:
            return float(s)
    return float(max_d)


def _heading_dirs(vx, vy):
    sp = np.sqrt(vx**2 + vy**2)
    if sp < 0.01: vx, vy, sp = 0.0, 1.0, 1.0
    hx, hy = vx/sp, vy/sp
    return (hx, hy), (-hy, hx), (hy, -hx)


def sc_analyze(step_val, progress=gr.Progress()):
    """运行机器人分析"""
    h_path = os.path.join(SC_OUT_DIR, "homography.npy")
    if not os.path.exists(h_path):
        return None, None, "请先完成配准"

    step = max(1, int(step_val))
    maze_path = os.path.join(SC_DIR, "maze1.png")
    maze = cv2.imread(maze_path, cv2.IMREAD_UNCHANGED)
    a, gray = maze[:,:,3], cv2.cvtColor(maze[:,:,:3], cv2.COLOR_BGR2GRAY)
    H = np.load(h_path)

    imgs = sorted(glob.glob(os.path.join(SC_DIR, "images_vis", "*.jpg")))
    sample = cv2.imread(imgs[0])
    img_h, img_w = sample.shape[:2]

    wall_src = ((a > 50) & (gray < 100)).astype(np.uint8) * 255
    wall_mask = (cv2.warpPerspective(wall_src, H, (img_w, img_h)) > 128).astype(np.uint8) * 255
    corr_src = ((a > 50) & (gray >= 100)).astype(np.uint8) * 255
    corr_mask = (cv2.warpPerspective(corr_src, H, (img_w, img_h)) > 128).astype(np.uint8) * 255

    # 解析标签
    label_dir = os.path.join(SC_DIR, "labels")
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    all_pos = {}
    for lf in label_files:
        fid = int(os.path.splitext(os.path.basename(lf))[0])
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = list(map(float, parts[1:]))
                pts = np.array([[coords[i]*img_w, coords[i+1]*img_h]
                                for i in range(0, len(coords), 2)])
                all_pos[fid] = tuple(pts.mean(axis=0))
                break

    sampled = sorted(fid for fid in all_pos if fid % step == 0)
    if not sampled:
        return None, None, "无有效标签"

    results = []
    prev_hd = None
    for i, fid in enumerate(progress.tqdm(sampled, desc="分析中")):
        cx, cy = all_pos[fid]
        vx = vy = speed = 0.0
        if i > 0:
            px, py = all_pos[sampled[i-1]]
            vx, vy = (cx-px)/step, (cy-py)/step
            speed = np.sqrt(vx**2 + vy**2)

        heading = np.arctan2(vy, vx) if speed > 0.01 else (prev_hd or np.pi/2)
        curv = 0.0
        if prev_hd is not None and speed > 0.01:
            dth = (heading - prev_hd + np.pi) % (2*np.pi) - np.pi
            ds = speed * step
            curv = dth / ds if ds > 0.1 else 0.0
        prev_hd = heading

        _, (lx,ly), (rx,ry) = _heading_dirs(vx, vy)
        dl_w = _raycast(cx, cy, lx, ly, wall_mask, 255)
        dr_w = _raycast(cx, cy, rx, ry, wall_mask, 255)
        dl_c = _raycast(cx, cy, lx, ly, corr_mask, 0)
        dr_c = _raycast(cx, cy, rx, ry, corr_mask, 0)

        results.append({
            "frame": fid,
            "cx": round(float(cx),2), "cy": round(float(cy),2),
            "vx": round(float(vx),3), "vy": round(float(vy),3),
            "speed": round(float(speed),3),
            "heading_deg": round(float(np.degrees(heading)),1),
            "curvature": round(float(curv),4),
            "dist_l_wall": round(float(dl_w),1), "dist_r_wall": round(float(dr_w),1),
            "dist_l_corr": round(float(dl_c),1), "dist_r_corr": round(float(dr_c),1),
            "corridor_w": round(float(dl_c + dr_c),1),
        })

    # 保存 CSV
    os.makedirs(SC_OUT_DIR, exist_ok=True)
    csv_path = os.path.join(SC_OUT_DIR, "robot_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    # 可视化轨迹
    vis = sample.copy()
    c1, _ = cv2.findContours(corr_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c1, -1, (200,180,0), 1)
    c2, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, c2, -1, (0,0,200), 1)

    max_k = max((abs(r["curvature"]) for r in results[1:]), default=0.001) or 0.001
    for j in range(1, len(results)):
        p1 = (int(results[j-1]["cx"]), int(results[j-1]["cy"]))
        p2 = (int(results[j]["cx"]), int(results[j]["cy"]))
        kn = min(abs(results[j]["curvature"]) / max_k, 1.0)
        cv2.line(vis, p1, p2, (0, int(255*(1-kn)), int(255*kn)), 2)
    for j in range(0, len(results), max(1, len(results)//15)):
        r = results[j]
        pt = (int(r["cx"]), int(r["cy"]))
        (hx,hy),_,_ = _heading_dirs(r["vx"], r["vy"])
        cv2.arrowedLine(vis, pt, (int(r["cx"]+hx*15), int(r["cy"]+hy*15)), (255,255,255), 1, tipLength=0.3)
        cv2.circle(vis, pt, 3, (0,255,255), -1)

    vis_path = os.path.join(SC_OUT_DIR, "trajectory.png")
    cv2.imwrite(vis_path, vis)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # CSV 预览文本
    lines = [",".join(results[0].keys())]
    for r in results[:10]:
        lines.append(",".join(str(v) for v in r.values()))
    if len(results) > 10:
        lines.append(f"... 共 {len(results)} 行")
    preview = "\n".join(lines)

    return vis_rgb, preview, f"分析完成! {len(results)} 个采样点\nCSV: {csv_path}\n轨迹: {vis_path}"


def sc_reset():
    return None, None, None, [], [], "已重置"


# Step 4-3 / 4-4 后端逻辑已迁移至 maze_processing.py (import mp)


# ───────── 构建 Gradio 界面 ─────────

def build_app():
    video_dirs = list_video_dirs()

    with gr.Blocks(
        title="拒绝无效加班！！！",
    ) as app:
        gr.Markdown("# SAM2 - YOLO → SHARED CONTREOL 端到端 拒绝无效加班！！！")
        gr.Markdown("视频切帧 → 点击选目标 → sam分割传播 → 导出 YOLO 数据集 → 训练 YOLO 模型 → SHARED CONTREOL 数据集")

        # ── Step 0: 视频切帧 ──
        with gr.Accordion("Step 0: 视频切帧", open=False):
            with gr.Row():
                video_file_dropdown = gr.Dropdown(
                    choices=list_video_files(), label="选择视频文件",
                    info="notebooks/videos/ 下的视频",
                )
                frame_interval = gr.Number(value=3, label="帧间隔", info="每N帧取1帧", precision=0)
                extract_btn = gr.Button("切帧", variant="primary")
            extract_log = gr.Textbox(label="切帧结果", lines=2, interactive=False)

        # ── Step 1: SAM 选目标分割 & yolo_dataset 创建 ──
        with gr.Accordion("Step 1: SAM 选目标分割, yolo_dataset 创建", open=False):
            # ── State ──
            points_state = gr.State([])
            labels_state = gr.State([])

            with gr.Row():
                # ── 左栏：设置 ──
                with gr.Column(scale=1):
                    video_dir_dropdown = gr.Dropdown(
                        choices=video_dirs,
                        label="选择视频帧目录",
                        info="notebooks/video_to_img/ 下的子文件夹",
                    )
                    frame_slider = gr.Slider(
                        minimum=0, maximum=0, step=1, value=0,
                        label="帧索引",
                        info="选择要标注的帧",
                    )
                    point_type = gr.Radio(
                        choices=["正样本 (前景)", "负样本 (背景)"],
                        value="正样本 (前景)",
                        label="点击类型",
                        info="正样本=目标区域，负样本=排除区域",
                    )
                    class_id = gr.Number(value=0, label="CLASS_ID", info="YOLO 类别 ID", precision=0)
                    info_box = gr.Textbox(label="信息", lines=6, interactive=False)

                    with gr.Row():
                        clear_btn = gr.Button("清除所有点", variant="secondary", size="sm")
                        undo_btn = gr.Button("撤销上一个点", variant="secondary", size="sm")

                # ── 右栏：图片 ──
                with gr.Column(scale=2):
                    image_display = gr.Image(
                        label="点击图片选择目标点（绿色=正样本，红色=负样本）",
                        type="pil",
                        interactive=False,
                    )

            with gr.Row():
                export_btn = gr.Button("propagate & 导出 YOLO 实例分割数据集", variant="primary", size="lg")

            with gr.Row():
                preview_image = gr.Image(label="Mask 预览", type="pil", interactive=False)

            export_log = gr.Textbox(label="logs", lines=15, interactive=False)

        # ── Step 2: YOLO 训练 ──
        with gr.Accordion("Step 2: YOLO 训练", open=False):
            with gr.Row():
                yolo_dataset_dropdown = gr.Dropdown(
                    choices=list_yolo_datasets(), label="选择数据集",
                    info="notebooks/yolo_dataset/ 下的数据集",
                )
                yolo_model_name = gr.Textbox(value="yolo26s-seg.pt", label="模型")
                yolo_class_name = gr.Textbox(value="magnet", label="类别名称")
            with gr.Row():
                yolo_epochs = gr.Number(value=30, label="epochs", precision=0)
                yolo_batch = gr.Number(value=32, label="batch", precision=0)
                yolo_imgsz = gr.Number(value=640, label="imgsz", precision=0)
                yolo_val_ratio = gr.Number(value=0.2, label="val比例")
            train_btn = gr.Button("准备数据集 & 开始训练", variant="primary", size="lg")
            train_log = gr.Textbox(label="训练日志", lines=15, interactive=False)
            train_result = gr.Textbox(label="best.pt 路径 & scp 指令", lines=5, interactive=False)

        # ── Step 3: Shared Control Dataset ──
        with gr.Accordion("Step 3: Shared Control Dataset (迷宫配准 + 运动分析)", open=False):
            sc_maze_pts = gr.State([])
            sc_cam_pts = gr.State([])

            with gr.Row():
                sc_dataset_dd = gr.Dropdown(
                    choices=list_yolo_datasets(), label="选择 YOLO 数据集",
                    info="Step 1 导出的数据集 (images_vis + labels)",
                )
                sc_load_btn = gr.Button("加载数据", variant="primary")

            gr.Markdown(f"迷宫图固定路径: `{os.path.join(SC_DIR, 'maze1.png')}`")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 迷宫图 (点击 4 个特征点)")
                    sc_maze_img = gr.Image(label="迷宫", interactive=False)
                    sc_maze_info = gr.Textbox(label="迷宫特征点", value="(0/4)")
                with gr.Column():
                    gr.Markdown("#### 相机图 (点击对应 4 个点)")
                    sc_cam_img = gr.Image(label="相机", interactive=False)
                    sc_cam_info = gr.Textbox(label="相机特征点", value="(0/4)")

            with gr.Row():
                sc_alpha = gr.Slider(10, 100, value=70, step=5, label="叠加透明度 %")
                sc_step = gr.Number(value=4, label="采样间隔 (帧)", precision=0)
                sc_reg_btn = gr.Button("配准", variant="primary")
                sc_analyze_btn = gr.Button("运动分析", variant="primary")
                sc_reset_btn = gr.Button("重置")

            with gr.Row():
                sc_result_img = gr.Image(label="配准/轨迹结果")
            sc_csv_preview = gr.Textbox(label="CSV 预览", lines=8, interactive=False)
            sc_info = gr.Textbox(label="信息", lines=3, interactive=False)

            # Step 3 事件
            sc_load_btn.click(
                sc_on_dataset_change, [sc_dataset_dd],
                [sc_maze_img, sc_cam_img, sc_result_img, sc_info],
            )
            sc_maze_img.select(
                sc_click_maze, [sc_maze_pts],
                [sc_maze_img, sc_maze_pts, sc_maze_info],
            )
            sc_cam_img.select(
                sc_click_cam, [sc_cam_pts],
                [sc_cam_img, sc_cam_pts, sc_cam_info],
            )
            sc_reg_btn.click(
                sc_register, [sc_maze_pts, sc_cam_pts, sc_alpha],
                [sc_result_img, sc_info],
            )
            sc_analyze_btn.click(
                sc_analyze, [sc_step],
                [sc_result_img, sc_csv_preview, sc_info],
            )
            sc_reset_btn.click(
                sc_reset, [],
                [sc_maze_img, sc_cam_img, sc_result_img, sc_maze_pts, sc_cam_pts, sc_info],
            )

        # ── Step 4: 模板对齐补全 (来自 app_template_align) ──
        with gr.Accordion("Step 4: Shared Control Dataset (mask align + alpha标签生成)", open=False):
                                   
            gr.Markdown(
                "Step 4 会调用 `app_template_align.py` 的同一套回调，"
            )

            with gr.Accordion("Step 4-1: mask align", open=False):
                with gr.Row():
                    t4_video_dir_dd = gr.Dropdown(
                        choices=ta.list_video_dirs(),
                        label="选择视频帧目录",
                        info="video_to_img/ 下的子文件夹",
                    )
                    t4_template_path = gr.Textbox(
                        value=ta.DEFAULT_TEMPLATE, label="mask路径"
                    )
                    t4_load_btn = gr.Button("加载数据", variant="primary")

                t4_pos_x_state = gr.State(None)
                t4_pos_y_state = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=1):
                        t4_tpl_img = gr.Image(label="mask轮廓", interactive=False, height=220)
                        gr.Markdown("点击右图定位，滑条调缩放/旋转/膨胀")
                        t4_scale = gr.Slider(0.01, 0.5, 0.075, step=0.005, label="缩放")
                        t4_angle = gr.Slider(-180, 180, -92.5, step=0.5, label="旋转 (度)")
                        t4_dilate = gr.Slider(0, 10, 3, step=1, label="mask膨胀 (px)")
                        t4_alpha = gr.Slider(10, 80, 40, step=5, label="透明度 %")
                        with gr.Row():
                            t4_update_btn = gr.Button("刷新", variant="secondary")
                            t4_save_btn = gr.Button("保存对齐", variant="primary")
                        t4_info_box = gr.Textbox(label="状态", lines=3, interactive=False)
                    with gr.Column(scale=2):
                        t4_preview_img = gr.Image(
                            label="点击 | 绿轮廓 | mask | 红十字中心",
                            interactive=False,
                        )
                        t4_preview_info = gr.Textbox(label="预览", lines=1, interactive=False)
                t4_save_result = gr.Textbox(label="保存结果", lines=3, interactive=False)

            with gr.Accordion("Step 4-2: SAM传播 + mask align", open=False):
                t4_pts_state = gr.State([])
                t4_labels_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1):
                        t4_frame_slider = gr.Slider(0, 1000, 0, step=1, label="标注帧索引")
                        t4_point_type = gr.Radio(
                            ["正样本 (前景)", "负样本 (背景)"],
                            value="正样本 (前景)",
                            label="点击类型",
                        )
                        t4_class_id = gr.Number(value=0, label="CLASS_ID", precision=0)
                        t4_track_angle = gr.Checkbox(value=True, label="角度追踪")
                        t4_use_endpoints = gr.Checkbox(
                            value=True, label="端点圆匹配", info="默认开启"
                        )
                        with gr.Row():
                            t4_clear_btn = gr.Button("清除标注", variant="secondary")
                            t4_preview_btn = gr.Button("预览 SAM Mask", variant="secondary")
                        t4_run_btn = gr.Button(
                            "传播 + 补全 + 导出 YOLO", variant="primary", size="lg"
                        )
                        t4_pts_info = gr.Textbox(label="标注", lines=2, interactive=False)
                    with gr.Column(scale=2):
                        t4_annotate_img = gr.Image(
                            label="点击标注目标 (绿=前景, 红=背景)", interactive=False
                        )

                with gr.Row():
                    t4_sam_preview_img = gr.Image(label="SAM mask vs 模板", interactive=False)
                    t4_sam_preview_info = gr.Textbox(label="对比信息", lines=12, interactive=False)

                t4_log = gr.Textbox(label="运行日志", lines=12, interactive=False)

                with gr.Row():
                    t4_browse_slider = gr.Slider(0, 1000, 0, step=1, label="浏览帧索引")
                with gr.Row():
                    t4_browse_img = gr.Image(label="左: SAM分割 + mask align (images_vis)", interactive=False)
                    t4_browse_img_tpl_only = gr.Image(
                        label="右: 仅mask (images_vis_template_only)", interactive=False
                    )
                t4_browse_info = gr.Textbox(label="帧信息", lines=1, interactive=False)

            with gr.Accordion("Step 4-3: 迷宫配准 (maze registration)", open=False):
                with gr.Row():
                    t43_dataset_dd = gr.Dropdown(
                        choices=mp.list_mask_align_datasets(),
                        label="选择 mask align 数据集",
                        info="mask_align_sam2_dataset/ 下的文件夹",
                    )
                    t43_load_btn = gr.Button("加载数据", variant="primary")
                gr.Markdown(
                    f"迷宫图: `{mp.MAZE_PATH}` · "
                    "**点击右图设置迷宫中心，滑条调缩放/旋转，参数可保存/自动加载**"
                )

                t43_cx_state = gr.State(None)
                t43_cy_state = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=1):
                        t43_maze_img = gr.Image(
                            label="迷宫原图", interactive=False, height=200
                        )
                        t43_scale = gr.Slider(
                            0.05, 2.0, 0.5, step=0.01, label="缩放"
                        )
                        t43_angle = gr.Slider(
                            -180, 180, 0, step=1, label="旋转 (度)"
                        )
                        t43_alpha = gr.Slider(
                            10, 100, 35, step=5, label="叠加透明度 %"
                        )
                        t43_info = gr.Textbox(label="信息", lines=5, interactive=False)
                    with gr.Column(scale=2):
                        t43_overlay_img = gr.Image(
                            label="点击设置迷宫中心 | 预览叠加效果",
                            interactive=False,
                        )
        # ── 事件绑定 ──

        # 切帧
        def extract_and_refresh(video_file, interval):
            log_text, vd_update = on_extract_frames(video_file, interval)
            return log_text, vd_update, vd_update

        extract_btn.click(
            fn=extract_and_refresh,
            inputs=[video_file_dropdown, frame_interval],
            outputs=[extract_log, video_dir_dropdown, t4_video_dir_dd],
        )

        # 切换视频目录
        video_dir_dropdown.change(
            fn=on_video_dir_change,
            inputs=[video_dir_dropdown],
            outputs=[image_display, frame_slider, frame_slider, info_box, points_state, labels_state, preview_image],
        )

        # 切换帧
        frame_slider.release(
            fn=on_frame_change,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state],
            outputs=[image_display],
        )

        # 点击图片添加点
        image_display.select(
            fn=on_image_click,
            inputs=[video_dir_dropdown, frame_slider, point_type, points_state, labels_state],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        # 清除所有点
        clear_btn.click(
            fn=on_clear_points,
            inputs=[video_dir_dropdown, frame_slider],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        # 撤销上一个点
        undo_btn.click(
            fn=on_undo_point,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state],
            outputs=[image_display, points_state, labels_state, info_box],
        )

        # 导出 YOLO 数据集，完成后刷新 Step 2 数据集列表
        def export_and_refresh(*args):
            log = on_export_yolo(*args)
            return log, gr.update(choices=list_yolo_datasets())

        export_btn.click(
            fn=export_and_refresh,
            inputs=[video_dir_dropdown, frame_slider, points_state, labels_state, class_id],
            outputs=[export_log, yolo_dataset_dropdown],
        )

        # YOLO 训练
        train_btn.click(
            fn=prepare_and_train_yolo,
            inputs=[yolo_dataset_dropdown, yolo_model_name, yolo_epochs,
                    yolo_batch, yolo_imgsz, yolo_class_name, yolo_val_ratio],
            outputs=[train_log, train_result],
        )

        # ── Step 4: 模板对齐回调绑定 ──
        t4_load_btn.click(
            fn=ta.on_load_data,
            inputs=[t4_video_dir_dd, t4_template_path],
            outputs=[t4_tpl_img, t4_preview_img, t4_pos_x_state, t4_pos_y_state, t4_scale, t4_info_box],
        )
        t4_preview_img.select(
            fn=ta.on_click_frame,
            inputs=[t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path,
                    t4_scale, t4_angle, t4_dilate, t4_alpha],
            outputs=[t4_preview_img, t4_pos_x_state, t4_pos_y_state, t4_preview_info],
        )
        _t4_s_in = [t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path,
                    t4_scale, t4_angle, t4_dilate, t4_alpha]
        _t4_s_out = [t4_preview_img, t4_preview_info]
        for ctrl in [t4_scale, t4_angle, t4_dilate, t4_alpha]:
            ctrl.release(fn=ta.on_slider_change, inputs=_t4_s_in, outputs=_t4_s_out)
        t4_update_btn.click(fn=ta.on_slider_change, inputs=_t4_s_in, outputs=_t4_s_out)
        t4_save_btn.click(
            fn=ta.on_save_params,
            inputs=[t4_pos_x_state, t4_pos_y_state, t4_video_dir_dd, t4_template_path, t4_scale, t4_angle],
            outputs=[t4_save_result],
        )

        def t4_on_dir_or_frame(vdir, fi):
            img, info = ta.on_s2_load_frame(vdir, fi)
            frames = ta.get_sorted_frames(vdir) if vdir else []
            return (
                img, info,
                gr.update(maximum=max(len(frames) - 1, 0)),
                gr.update(maximum=max(len(frames) - 1, 0)),
                [], [],
            )

        t4_video_dir_dd.change(
            fn=t4_on_dir_or_frame,
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_info, t4_frame_slider, t4_browse_slider, t4_pts_state, t4_labels_state],
        )
        t4_frame_slider.release(
            fn=lambda vd, fi: ta.on_s2_load_frame(vd, fi),
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_info],
        )
        t4_annotate_img.select(
            fn=ta.on_s2_click,
            inputs=[t4_video_dir_dd, t4_frame_slider, t4_point_type, t4_pts_state, t4_labels_state],
            outputs=[t4_annotate_img, t4_pts_state, t4_labels_state, t4_pts_info],
        )
        t4_clear_btn.click(
            fn=ta.on_s2_clear,
            inputs=[t4_video_dir_dd, t4_frame_slider],
            outputs=[t4_annotate_img, t4_pts_state, t4_labels_state, t4_pts_info],
        )
        t4_preview_btn.click(
            fn=ta.on_s2_preview_mask,
            inputs=[t4_video_dir_dd, t4_template_path, t4_frame_slider,
                    t4_pts_state, t4_labels_state, t4_pos_x_state, t4_pos_y_state, t4_scale, t4_angle],
            outputs=[t4_sam_preview_img, t4_sam_preview_info],
        )
        t4_run_btn.click(
            fn=ta.on_s2_propagate_and_complete,
            inputs=[t4_video_dir_dd, t4_template_path, t4_frame_slider,
                    t4_pts_state, t4_labels_state, t4_class_id,
                    t4_pos_x_state, t4_pos_y_state, t4_scale, t4_angle, t4_dilate,
                    t4_track_angle, t4_use_endpoints],
            outputs=[t4_log, t4_browse_img],
        )
        def t4_on_browse_dual(vdir, fi):
            fi = int(fi)
            img_name = f"{fi:05d}.jpg"
            left = None
            right = None

            if vdir:
                out_dir = os.path.join(ta.SAM2_DATASET_DIR, vdir)
                left_path = os.path.join(out_dir, "images_vis", img_name)
                right_path = os.path.join(out_dir, "images_vis_template_only", img_name)
                if os.path.exists(left_path):
                    left = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
                if os.path.exists(right_path):
                    right = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

            # 如果磁盘结果还没有，就回退到内存可视化
            info = f"帧 {fi}"
            if left is None:
                left_mem, info_mem = ta.on_s2_browse(vdir, fi)
                left = left_mem
                info = info_mem
            else:
                if right is None:
                    info = f"帧 {fi} | 找不到 images_vis_template_only/{img_name}"
                else:
                    info = f"帧 {fi} | 左=images_vis | 右=images_vis_template_only"
            return left, right, info

        t4_browse_slider.release(
            fn=t4_on_browse_dual,
            inputs=[t4_video_dir_dd, t4_browse_slider],
            outputs=[t4_browse_img, t4_browse_img_tpl_only, t4_browse_info],
        )

        # ── Step 4-3: 迷宫配准回调绑定 (来自 maze_processing.py) ──
        t43_load_btn.click(
            mp.on_load, [t43_dataset_dd],
            [t43_maze_img, t43_overlay_img,
             t43_cx_state, t43_cy_state, t43_scale, t43_angle, t43_info],
        )
        t43_overlay_img.select(
            mp.on_click,
            [t43_cx_state, t43_cy_state, t43_dataset_dd,
             t43_scale, t43_angle, t43_alpha],
            [t43_overlay_img, t43_cx_state, t43_cy_state, t43_info],
        )
        _t43_s_in = [t43_cx_state, t43_cy_state, t43_dataset_dd,
                     t43_scale, t43_angle, t43_alpha]
        _t43_s_out = [t43_overlay_img, t43_info]
        for ctrl in [t43_scale, t43_angle, t43_alpha]:
            ctrl.release(fn=mp.on_slider, inputs=_t43_s_in, outputs=_t43_s_out)

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
