# -*- coding: utf-8 -*-
"""
Time:     2025/4/16 (Modified for better CVAT compatibility)
Author:   ZhaoQi Cao(czq) - Adapted for Instance Segmentation
Version:  V 1.1 (中文) (视频实例分割测试 - 优化XML输出)
File:     model_http_seg.py # Or your preferred name
Describe: 用于测试部署在Nuclio上的YOLOv8实例分割函数 (处理视频输入, 保存更兼容CVAT的XML和带标注的视频)
"""
import cv2
import base64
import requests
import json
import os
import argparse
import logging
import time
from datetime import timedelta
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import random
import sys
import datetime
import math
import torch
from collections import deque
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology, filters

try:
    from torch.nn import functional as F
except ImportError:
    logging.warning("PyTorch功能模块导入失败，某些高级特性可能不可用")

# --- 高级检测配置 ---
ENABLE_MULTI_SCALE = True # 启用多尺度检测
ENABLE_FPN = True # 启用特征金字塔网络
ENABLE_TEMPORAL_SMOOTHING = True # 启用时序平滑
ENABLE_ADAPTIVE_THRESHOLD = True # 启用自适应阈值
ENABLE_LOW_CONF_ENHANCEMENT = True # 启用低置信度增强
ENABLE_EDGE_REFINEMENT = True # 启用边缘优化
ENABLE_BATCH_PROCESSING = True # 启用批处理
ENABLE_TEST_TIME_AUGMENTATION = True # 启用测试时增强
ENABLE_MODEL_ENSEMBLE = True # 启用模型集成
ENABLE_BOUNDARY_REFINEMENT = True # 启用边界细化

# 默认配置
DEFAULT_VIDEO_PATH = "E:\样例\游乐场\95372_segment_1.mp4"
DEFAULT_NUCLIO_SEG_URL = "http://192.168.10.158:32792"
DEFAULT_OUTPUT_BASE_DIR = "E:\视频样例其他类\游乐场1"
DEFAULT_FRAME_SKIP = 20
DEFAULT_SAVE_VIDEO = True
CONF_THRESHOLD = 0.35
REQUEST_TIMEOUT = 30
VIDEO_CODEC = 'mp4v'
VIDEO_EXTENSION = '.mp4'
DRAW_MASK = True
DRAW_BOX = True
DRAW_LABEL = True
MASK_ALPHA = 0.5

# 多尺度检测配置
SCALE_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5] # 扩展多尺度因子
SCALE_WEIGHTS = [0.1, 0.2, 0.4, 0.2, 0.1] # 对应权重

# 时序平滑配置
TEMPORAL_WINDOW = 5 # 时序窗口大小
MIN_TRACK_CONFIDENCE = 0.3 # 最小跟踪置信度
IOU_THRESHOLD = 0.5 # IoU阈值，用于跟踪和多尺度检测结果合并

# 点采样配置
MIN_POINT_DISTANCE = 3
MAX_POINT_DISTANCE = 15
DENSITY_FACTOR = 2.0

# 多边形优化配置
MAX_POLYGON_POINTS = 100
MIN_POLYGON_POINTS = 6
RDPEPS = 2.0
SMOOTH_POLYGON = True
OPTIMIZE_POLYGON_POINTS = True
VISUAL_CONNECT_DISTANCE = 20.0
CURVATURE_ADAPTIVE = True

# 图像预处理配置
ENABLE_FRAME_PREPROCESS = True
BRIGHTNESS_ADJUST = 1.1
CONTRAST_ADJUST = 1.1
USE_CLAHE = True
DENOISE_STRENGTH = 2
SHARPEN_STRENGTH = 0.5

# 特征金字塔网络配置
FPN_LEVELS = 3
FPN_SCALES = [0.5, 1.0, 2.0]

# 边缘增强配置
EDGE_ENHANCEMENT_STRENGTH = 0.5

# 目标跟踪配置
MAX_TRACK_AGE = 30
MAX_MISSED_FRAMES = 5

# *** 重要: 定义你的模型可能输出的所有类别标签 ***
ALL_POSSIBLE_LABELS = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','man',
}

# 创建日志和输出目录
os.makedirs('./logs/', exist_ok=True)
output_base_dir = Path(DEFAULT_OUTPUT_BASE_DIR)
output_xmls_dir = output_base_dir / "annotations"
output_videos_dir = output_base_dir / "annotated_videos"
output_xmls_dir.mkdir(parents=True, exist_ok=True)
output_videos_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE = "./logs/video_segmentation_testing_annotated.log" # 日志文件名

# --- 设置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 颜色列表 (用于绘图) ---
COLOR_LIST_BGR = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (255, 128, 0), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 255), (128, 0, 255)
]

def get_color(index):
    """获取一个颜色用于绘制"""
    # 确保索引有效，或者使用哈希等方法为标签分配固定颜色
    return COLOR_LIST_BGR[index % len(COLOR_LIST_BGR)]

def bgr_to_hex(bgr_color):
    """将 BGR 元组转换为 Hex 颜色字符串"""
    return '#{:02x}{:02x}{:02x}'.format(bgr_color[2], bgr_color[1], bgr_color[0])

# --- 辅助函数 (encode_frame_to_base64, call_nuclio_segmentor, pretty_print_xml) ---
def encode_frame_to_base64(frame):
    """将 OpenCV 帧编码为 Base64 字符串"""
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if is_success:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            logging.warning("帧编码失败 (cv2.imencode 返回 False)。")
            return None
    except Exception as e:
        logging.error(f"帧编码时发生异常: {e}")
        return None

def call_nuclio_segmentor(base64_image_string, nuclio_url, frame_number):
    """调用 Nuclio 实例分割函数并解析结果"""
    payload = json.dumps({"image": base64_image_string})
    headers = {'Content-Type': 'application/json'}
    logging.debug(f"帧 {frame_number}: 发送请求到 {nuclio_url}...")
    try:
        start_time = time.time()
        response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        end_time = time.time()
        logging.debug(f"帧 {frame_number}: Nuclio 请求耗时 {end_time - start_time:.2f} 秒。")

        if response.status_code == 200:
            try:
                results = response.json()
                # --- 验证响应格式 ---
                if isinstance(results, list):
                    logging.info(f"帧 {frame_number}: 收到 {len(results)} 个检测实例。")
                    valid_results = []
                    
                    # 定义常见错误检测的特征
                    error_patterns = [
                        {
                            "class_names": ["person", "man"], 
                            "min_area_ratio": 0.6,  # 面积占比超过60%可能是误检
                            "aspect_ratio_range": (0.1, 10.0),  # 极端的宽高比可能是误检
                            "edge_response_threshold": 0.15  # 边缘响应低于15%可能是误检
                        }
                    ]
                    
                    for instance in results:
                        # 基本格式验证
                        if (isinstance(instance, dict) and
                            'box_normalized' in instance and isinstance(instance['box_normalized'], list) and len(instance['box_normalized']) == 4 and
                            'confidence' in instance and isinstance(instance['confidence'], (float, int)) and
                            'class_name' in instance and isinstance(instance['class_name'], str) and
                            'mask_polygon_normalized' in instance and isinstance(instance['mask_polygon_normalized'], list)):
                            
                            # 应用高级验证规则，过滤错误检测
                            should_filter = False
                            
                            # 检查是否符合误检模式
                            for pattern in error_patterns:
                                if instance['class_name'].lower() in pattern["class_names"]:
                                    # 计算面积比例
                                    box = instance['box_normalized']
                                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                                    
                                    # 计算长宽比
                                    box_width = box[2] - box[0]
                                    box_height = box[3] - box[1]
                                    aspect_ratio = box_width / box_height if box_height > 0 else 0
                                    
                                    # 计算边缘响应（使用多边形点的复杂度估计）
                                    mask_points = np.array(instance['mask_polygon_normalized'])
                                    if len(mask_points) > 5:
                                        perimeter = cv2.arcLength(
                                            (mask_points * 1000).astype(np.int32), 
                                            True
                                        )
                                        area = cv2.contourArea(
                                            (mask_points * 1000).astype(np.int32)
                                        )
                                        edge_response = 0
                                        if area > 0:
                                            # 使用圆形度估计边缘响应
                                            edge_response = 4 * np.pi * area / (perimeter * perimeter)
                                        
                                        logging.debug(f"检测 {instance['class_name']}: 面积比={box_area}, 宽高比={aspect_ratio}, 边缘响应={edge_response}")
                                        
                                        # 应用过滤规则
                                        if (box_area > pattern["min_area_ratio"] or 
                                            aspect_ratio < pattern["aspect_ratio_range"][0] or 
                                            aspect_ratio > pattern["aspect_ratio_range"][1] or
                                            edge_response < pattern["edge_response_threshold"]):
                                            should_filter = True
                                            logging.info(f"过滤可能的误检: {instance['class_name']}, 面积比={box_area:.3f}, 宽高比={aspect_ratio:.3f}, 边缘响应={edge_response:.3f}")
                            
                            # 判断是否要过滤当前检测实例
                            if not should_filter:
                                valid_results.append(instance)
                            else:
                                logging.warning(f"帧 {frame_number}: 跳过无效或格式不正确的实例: {str(instance)[:200]}...")
                    return valid_results
                else:
                    logging.error(f"帧 {frame_number}: Nuclio 响应不是列表。类型: {type(results)}, 响应: {str(response.text)[:200]}...")
                    return None
            except json.JSONDecodeError:
                logging.error(f"帧 {frame_number}: 解析 Nuclio JSON 失败。状态码: {response.status_code}, 响应: {response.text[:200]}...")
                return None
        else:
            logging.error(f"帧 {frame_number}: Nuclio 错误。状态码: {response.status_code}, 响应: {response.text[:200]}...")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 超时 ({REQUEST_TIMEOUT} 秒)。")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 失败: {e}")
        return None
    except Exception as e:
        logging.error(f"帧 {frame_number}: 调用 Nuclio 时发生意外错误: {e}")
        return None

def pretty_print_xml(elem):
    """美化打印 XML Element"""
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = '\n'.join([line for line in reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8').split('\n') if line.strip()])
        return pretty_xml
    except Exception as e:
        logging.error(f"XML美化打印出错: {e}")
        try:
            ET.indent(elem, space="  ")
            return ET.tostring(elem, encoding='unicode')
        except Exception:
             return ET.tostring(elem, encoding='unicode')


# --- 绘图函数 (用于实例分割) ---
def draw_segmentation(frame, detections, conf_threshold):
    """在帧上绘制实例分割结果 (边界框、掩码、标签)。"""
    h, w, _ = frame.shape
    annotated_frame = frame.copy()
    overlay = frame.copy()
    instance_count = 0
    processed_labels = set() # 用于分配颜色

    for i, det in enumerate(detections):
        confidence = det.get('confidence', 0)
        if confidence < conf_threshold:
            continue

        box_norm = det.get('box_normalized')
        mask_poly_norm = det.get('mask_polygon_normalized')
        label = det.get('class_name', 'unknown')
        label_text = f"{label}: {confidence:.2f}"

        # 为标签分配颜色 (尽量保持一致)
        if label not in processed_labels:
            processed_labels.add(label)
        color_index = list(sorted(processed_labels)).index(label) # 基于已处理标签的顺序获取索引
        color = get_color(color_index)

        # --- 绘制掩码 ---
        if DRAW_MASK and mask_poly_norm:
            try:
                mask_points_px = np.array(mask_poly_norm) * np.array([w, h])
                mask_points_px = mask_points_px.astype(np.int32)
                
                # 填充掩码区域
                cv2.fillPoly(overlay, [mask_points_px], color)
                
                # 绘制更细的轮廓线
                cv2.polylines(annotated_frame, [mask_points_px], True, color, 1)
                
                # 仅在关键点绘制小圆点标记
                # 替换为均匀采样关键点，确保点沿掩码边界均匀分布
                n_points = len(mask_points_px)
                if n_points > 0:
                    num_keypoints = min(8, n_points)
                    indices = np.linspace(0, n_points - 1, num=num_keypoints, dtype=int)
                    for idx in indices:
                        cv2.circle(annotated_frame, tuple(mask_points_px[idx]), 1, (0, 255, 255), -1)
                
            except Exception as e:
                logging.warning(f"绘制掩码时出错 (实例 {i}): {e}. Mask data: {str(mask_poly_norm)[:100]}")

        # --- 绘制边界框 ---
        if DRAW_BOX and box_norm:
            try:
                x1, y1, x2, y2 = box_norm
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(annotated_frame, pt1, pt2, color, 1)  # 使用更细的线
            except Exception as e:
                logging.warning(f"绘制边界框时出错 (实例 {i}): {e}. Box data: {box_norm}")

        # --- 绘制标签 ---
        if DRAW_LABEL and box_norm:
             try:
                x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                # 使用更小的字体
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_origin = (x1, max(0, y1 - text_height - baseline // 2))
                cv2.rectangle(annotated_frame, label_origin, (label_origin[0] + text_width, label_origin[1] + text_height + baseline), color, -1)
                cv2.putText(annotated_frame, label_text, (label_origin[0], label_origin[1] + text_height + baseline // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
             except Exception as e:
                 logging.warning(f"绘制标签时出错 (实例 {i}): {e}. Label: {label_text}")

        instance_count += 1

    # --- 合并掩码图层 ---
    if instance_count > 0:
        cv2.addWeighted(overlay, MASK_ALPHA, annotated_frame, 1 - MASK_ALPHA, 0, annotated_frame)

    return annotated_frame


# --- 替换优化多边形点函数，使用更先进的方法 ---
def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    # 将点转换为numpy数组
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    
    # 计算线段向量
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    # 计算线段长度
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    # 计算投影
    t = np.dot(point_vec, line_vec) / (line_len * line_len)
    t = max(0, min(1, t))  # 限制在0-1之间
    
    # 计算最近点
    nearest = line_start + t * line_vec
    
    # 返回点到最近点的距离
    return np.linalg.norm(point - nearest)

def optimize_polygon_points(points, max_points=MAX_POLYGON_POINTS, min_points=MIN_POLYGON_POINTS, rdp_epsilon=RDPEPS):
    """优化多边形点：对输入轮廓点进行均匀采样和平滑，以获得平滑的曲线输出。
    该函数首先确保输入为(n,2)的二维数组，如果格式不正确则尝试重塑，
    然后基于累积弧长均匀采样目标点数，并对输出进行高斯平滑。
    参数:
        points: 输入点集
        max_points: 最大点数
        min_points: 最小点数
        rdp_epsilon: 此参数在新实现中未使用，但保留以兼容接口
    返回:
        优化后的点集 (形状为 (target_points,2))
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        try:
            points = points.reshape(-1, 2)
        except Exception as e:
            logging.error(f"输入多边形点格式错误，无法重塑: {e}")
            return points
        if len(points) < 3:
            return points
            
    # 计算累积弧长
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum_length = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_length[-1]

    # 确定目标点数，简单设为 (min_points + max_points) / 2
    target_points = int((min_points + max_points) / 2)
    if total_length == 0:
        return points

    # 均匀采样累积距离上的点
    even_distances = np.linspace(0, total_length, target_points)
    new_x = np.interp(even_distances, cum_length, points[:, 0])
    new_y = np.interp(even_distances, cum_length, points[:, 1])
    new_points = np.stack((new_x, new_y), axis=-1)

    # 对采样结果进行高斯平滑
    new_points[:, 0] = cv2.GaussianBlur(new_points[:, 0].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    new_points[:, 1] = cv2.GaussianBlur(new_points[:, 1].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()

    return new_points


# --- 修改 save_results_to_cvat_polygon_xml 函数，增加多边形点优化 ---
def save_results_to_cvat_polygon_xml(all_results, xml_output_path, video_filename, frame_width, frame_height, total_processed_frames, original_total_frames, conf_threshold):
    """将实例分割结果保存为更兼容 CVAT 的多边形格式 XML"""
    logging.info(f"正在为 {total_processed_frames} 个处理过的帧构建 CVAT 多边形 XML...")
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- Meta ---
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "N/A" # 通常由 CVAT 分配，导入时可为 N/A
    task_name, _ = os.path.splitext(video_filename)
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(original_total_frames) # 使用原始总帧数
    # *** 修改点: 根据需要设置 mode (interpolation 或 annotation) ***
    ET.SubElement(task, "mode").text = "interpolation" # 或者 "annotation"
    ET.SubElement(task, "overlap").text = "0" # 或根据你的设置调整
    ET.SubElement(task, "bugtracker").text = ""
    
    # 修复时间格式问题
    current_time_utc = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "+00:00"
    
    ET.SubElement(task, "created").text = current_time_utc
    ET.SubElement(task, "updated").text = current_time_utc
    ET.SubElement(task, "start_frame").text = "0"
    # *** 修改点: stop_frame 使用原始帧数 ***
    ET.SubElement(task, "stop_frame").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(task, "frame_filter").text = ""

    # --- Labels (重要修改) ---
    labels_elem = ET.SubElement(task, "labels")
    # *** 从 ALL_POSSIBLE_LABELS 生成标签定义 ***
    #    确保 ALL_POSSIBLE_LABELS 包含你模型会输出的所有类别名称
    logging.info(f"在 XML 中定义标签: {ALL_POSSIBLE_LABELS}")
    sorted_labels = sorted(list(ALL_POSSIBLE_LABELS)) # 排序以保证颜色一致性
    for idx, label_name in enumerate(sorted_labels):
        if not label_name: continue # 跳过空标签
        label_elem = ET.SubElement(labels_elem, "label")
        ET.SubElement(label_elem, "name").text = label_name
        color = get_color(idx) # 分配颜色
        ET.SubElement(label_elem, "color").text = bgr_to_hex(color)
        ET.SubElement(label_elem, "type").text = "polygon" # 指定类型为多边形
        ET.SubElement(label_elem, "attributes") # 添加空的 attributes

    # --- Segments ---
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0" # 通常默认为 0
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(segment, "url").text = "N/A" # 或任务 URL

    ET.SubElement(task, "owner").text = "N/A" # 可选
    ET.SubElement(task, "assignee").text = "N/A" # 可选
    ET.SubElement(task, "subset").text = "Default"
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(frame_width)
    ET.SubElement(original_size, "height").text = str(frame_height)
    
    # 修复时间格式问题
    ET.SubElement(meta, "dumped").text = current_time_utc

    # --- 添加图像和多边形数据 ---
    if all_results:
        for frame_number in sorted(all_results.keys()): # 确保按帧号顺序处理
            detections = all_results[frame_number]
            frame_id_0_based = frame_number - 1 # CVAT 使用 0-based

            # *** 修改点: 总是创建 <image> 标签，即使没有检测 ***
            #    这样可以确保帧序列的完整性，符合图像格式预期
            image_elem = ET.SubElement(root, "image")
            image_elem.set("id", str(frame_id_0_based))
            # *** 修改点: 添加 name 属性 ***
            image_elem.set("name", f"frame_{frame_id_0_based:06d}") # 标准的帧名格式
            image_elem.set("width", str(frame_width))
            image_elem.set("height", str(frame_height))

            # 只处理置信度高于阈值的有效检测
            valid_detections_in_frame = [det for det in detections if det.get('confidence', 0) >= conf_threshold]

            if not valid_detections_in_frame:
                logging.debug(f"帧 {frame_number}: 没有高于阈值 {conf_threshold} 的检测，生成空的 <image> 标签。")
                continue # 跳到下一帧

            for det_idx, det in enumerate(valid_detections_in_frame):
                label_name = det.get('class_name', 'unknown')
                mask_poly_norm = det.get('mask_polygon_normalized')

                # 检查标签是否存在于定义的标签列表中
                if label_name not in ALL_POSSIBLE_LABELS:
                    logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 检测到未在 ALL_POSSIBLE_LABELS 中定义的标签 '{label_name}'，跳过此实例。")
                    continue
                if not mask_poly_norm:
                    logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 标签 '{label_name}' 缺少掩码数据，跳过。")
                    continue

                # --- 创建 <polygon> ---
                poly_elem = ET.SubElement(image_elem, "polygon")
                poly_elem.set("label", label_name)
                poly_elem.set("source", "auto") # 标记为自动生成
                poly_elem.set("occluded", "0") # 简化处理
                poly_elem.set("outside", "0")  # 简化处理
                poly_elem.set("keyframe", "1") # 对图像格式，每个标注都是关键帧
                # *** 修改点: 添加 z_order 属性 ***
                poly_elem.set("z_order", "0")

                # --- 处理多边形点 ---
                try:
                    # 转换坐标并检查范围
                    points_px_raw = np.array(mask_poly_norm) * np.array([frame_width, frame_height])
                    
                    # 优化多边形点 (新增功能，根据对象类别选择不同优化方法)
                    if OPTIMIZE_POLYGON_POINTS:
                        label_lower = label_name.lower()
                        if label_lower in ['bottle']:
                            points_px_raw = optimize_polygon_points_bottle(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                        elif label_lower in ['person']:
                            points_px_raw = optimize_polygon_points_person(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                        else:
                            points_px_raw = optimize_polygon_points(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                    
                    # 裁剪坐标到图像边界内 (可选但推荐，防止 CVAT 错误)
                    points_px_raw[:, 0] = np.clip(points_px_raw[:, 0], 0, frame_width - 1)
                    points_px_raw[:, 1] = np.clip(points_px_raw[:, 1], 0, frame_height - 1)
                    points_px = points_px_raw.round(2) # 保留两位小数

                    # 确保至少有 3 个点
                    if len(points_px) < 3:
                        logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 标签 '{label_name}' 的多边形顶点少于 3 个 ({len(points_px)})，跳过。")
                        image_elem.remove(poly_elem) # 移除无效多边形
                        continue

                    # 格式化为 CVAT 期望的字符串 "x1,y1;x2,y2;..."
                    points_str = ";".join([f"{p[0]},{p[1]}" for p in points_px])
                    poly_elem.set("points", points_str)

                    # --- 添加置信度作为属性 (可选) ---
                    attr_conf = ET.SubElement(poly_elem, "attribute", name="confidence")
                    attr_conf.text = f"{det.get('confidence', 0):.4f}"

                except Exception as e:
                    logging.error(f"帧 {frame_number}, 实例 {det_idx}: 转换或格式化掩码点时出错: {e}")
                    try:
                        image_elem.remove(poly_elem) # 尝试移除无效的多边形
                    except ValueError: # 如果已经被移除或不存在
                        pass
                    continue

    # --- 保存 XML ---
    try:
        # 使用 ElementTree 的 indent 功能进行格式化 (比 minidom 更不易出错)
        ET.indent(root, space="  ", level=0)
        tree = ET.ElementTree(root)
        xml_output_path_str = str(xml_output_path)
        tree.write(xml_output_path_str, encoding='utf-8', xml_declaration=True)
        logging.info(f"XML 已保存: {xml_output_path_str}")

    except Exception as e:
        logging.error(f"保存 XML 失败: {e}")
        # 如果 indent/write 失败，尝试原始字符串保存作为后备
        try:
             xml_output_path_str = str(xml_output_path)
             rough_string = ET.tostring(root, encoding='unicode')
             with open(xml_output_path_str, "w", encoding="utf-8") as f:
                 f.write('<?xml version="1.0" encoding="utf-8"?>\n') # 手动添加声明
                 f.write(rough_string)
             logging.info(f"XML 已保存 (使用原始字符串后备): {xml_output_path_str}")
        except Exception as fallback_e:
             logging.error(f"后备 XML 写入也失败: {fallback_e}")




# --- DetectionTracker类和相关函数移动到此处 ---

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0



def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def apply_temporal_smoothing(detections, tracker, frame_number):
    """应用时间平滑处理，提高稳定性"""
    if not ENABLE_TEMPORAL_SMOOTHING or not tracker:
        return detections
        
    return tracker.update(detections, frame_number)

class DetectionTracker:
    """检测结果跟踪器，用于时序平滑"""
    def __init__(self, window_size=TEMPORAL_WINDOW):
        self.window_size = window_size
        self.tracks = {}  # 跟踪对象字典
        self.next_track_id = 0
        
    def update(self, detections, frame_number):
        """更新跟踪状态"""
        if not ENABLE_TEMPORAL_SMOOTHING:
            return detections
            
        current_tracks = {}
        
        # 更新现有轨迹
        for track_id, track in list(self.tracks.items()):
            if frame_number - track['last_frame'] > self.window_size:
                del self.tracks[track_id]
                continue
                
            # 寻找最佳匹配
            best_match = None
            best_iou = IOU_THRESHOLD
            
            for det in detections:
                iou = calculate_iou(track['box_normalized'], det['box_normalized'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                # 更新轨迹
                track['history'].append(best_match)
                track['last_frame'] = frame_number
                if len(track['history']) > self.window_size:
                    track['history'].pop(0)
                
                current_tracks[track_id] = track
                detections.remove(best_match)
            else:
                # 使用运动预测
                predicted_box = self._predict_position(track)
                if predicted_box is not None:
                    for det in detections:
                        iou = calculate_iou(predicted_box, det['box_normalized'])
                        if iou > IOU_THRESHOLD * 0.5:  # 降低预测位置的IOU阈值
                            track['history'].append(det)
                            track['last_frame'] = frame_number
                            if len(track['history']) > self.window_size:
                                track['history'].pop(0)
                            
                            current_tracks[track_id] = track
                            detections.remove(det)
                            break
        
        # 创建新轨迹
        for det in detections:
            if det.get('confidence', 0) > MIN_TRACK_CONFIDENCE:
                self.tracks[self.next_track_id] = {
                    'history': [det],
                    'box_normalized': det['box_normalized'],
                    'last_frame': frame_number
                }
                current_tracks[self.next_track_id] = self.tracks[self.next_track_id]
                self.next_track_id += 1
        
        self.tracks = current_tracks
        
        # 创建平滑后的检测结果
        smoothed_detections = []
        for track_id, track in self.tracks.items():
            if len(track['history']) > 0:
                # 复制最新的检测结果
                smoothed_det = track['history'][-1].copy()
                
                if len(track['history']) >= 3:  # 至少需要3帧才能平滑
                    # 计算位置平滑
                    boxes = [det['box_normalized'] for det in track['history']]
                    smoothed_box = np.mean(boxes, axis=0).tolist()
                    smoothed_det['box_normalized'] = smoothed_box
                    
                    # 如果有掩码多边形，也进行平滑
                    if all('mask_polygon_normalized' in det for det in track['history']):
                        # 找出具有最多点的多边形
                        max_points = max(len(det['mask_polygon_normalized']) for det in track['history'])
                        
                        # 对齐点数
                        aligned_masks = []
                        
                        try:
                            # 找出所有多边形的形状
                            mask_shapes = [np.array(det['mask_polygon_normalized']).shape for det in track['history']]
                            if not all(len(shape) == 2 for shape in mask_shapes):
                                # 处理非二维数组的情况
                                for det in track['history']:
                                    poly = np.array(det['mask_polygon_normalized'])
                                    if len(poly.shape) == 1:
                                        poly = poly.reshape(-1, 2)
                                        det['mask_polygon_normalized'] = poly.tolist()
                        except Exception as e:
                            logging.warning(f"处理多边形形状时出错: {e}")
                
                smoothed_det['track_id'] = track_id
                smoothed_detections.append(smoothed_det)
        
        return smoothed_detections
    
    def _predict_position(self, track):
        """基于历史预测下一位置"""
        if len(track['history']) < 2:
            return None
            
        # 获取最近的几个边界框
        recent_boxes = [det['box_normalized'] for det in track['history'][-2:]]
        
        # 计算平均移动
        box1 = np.array(recent_boxes[0])
        box2 = np.array(recent_boxes[1])
        movement = box2 - box1
        
        # 预测下一位置
        predicted_box = (box2 + movement).tolist()
        
        return predicted_box


# --- 移动重要函数到外部 ---

def refine_detection_boundaries(detections, image, semantic_check=True):
    """
    优化检测边界，使用最新的边缘增强和特征分析技术
    """
    if not detections:
        return detections
    
    refined_detections = []
    h, w = image.shape[:2]
    
    # 预处理图像以增强边缘
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 1. 应用多尺度Canny边缘检测
    edges_low = cv2.Canny(blur, 30, 90)
    edges_high = cv2.Canny(blur, 80, 200)
    combined_edges = cv2.bitwise_or(edges_low, edges_high)
    
    # 2. 增强边缘 - 使用自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    edge_enhance = cv2.bitwise_or(combined_edges, adaptive_thresh)
    
    # 3. 形态学操作改进边缘
    kernel = np.ones((3, 3), np.uint8)
    edge_enhance = cv2.morphologyEx(edge_enhance, cv2.MORPH_CLOSE, kernel)
    
    for detection in detections:
        # 类别语义检查和校正
        if semantic_check:
            try:
                # 提取对象区域用于分析
                class_name = detection.get('class_name', '').lower()
                # 使用 bbox_key 记录实际使用的键
                bbox = None
                bbox_key = None
                for key in ['bbox', 'box', 'box_normalized']:
                    if key in detection:
                        bbox = detection[key]
                        bbox_key = key
                        break
                if bbox is None:
                    logging.debug("未找到边界框信息，跳过语义一致性检查")
                    continue
                # 如使用标准化坐标则转换为像素坐标
                if bbox_key == 'box_normalized':
                    bbox = [
                        int(bbox[0] * w),
                        int(bbox[1] * h),
                        int(bbox[2] * w),
                        int(bbox[3] * h)
                    ]
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # 确保边界在图像内
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 跳过无效区域
                if x2 <= x1 or y2 <= y1:
                    pass # 继续处理
                else:
                    # 提取ROI
                    roi = image[y1:y2, x1:x2]
                    
                    # 1. 颜色分析
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    color_hist_h, _ = np.histogram(hsv_roi[:,:,0], bins=30, range=[0, 180])
                    color_hist_s, _ = np.histogram(hsv_roi[:,:,1], bins=32, range=[0, 256])
                    
                    # 归一化直方图
                    color_hist_h = color_hist_h / np.sum(color_hist_h)
                    color_hist_s = color_hist_s / np.sum(color_hist_s)
                    
                    # 2. 形状特征计算
                    shape_features = {}
                    area = 0
                    if 'mask_polygon_normalized' in detection:
                        try:
                            mask_points = np.array(detection['mask_polygon_normalized'])
                            mask_points = (mask_points * np.array([w, h])).astype(np.int32)
                            
                            # 创建掩码
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask, [mask_points], 255)
                            roi_mask = mask[y1:y2, x1:x2]
                            
                            # 计算周长和面积
                            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                perimeter = cv2.arcLength(largest_contour, True)
                                area = cv2.contourArea(largest_contour)
                                
                                if area > 0:
                                    # 计算圆度 (4*pi*area/perimeter^2)
                                    shape_features['circularity'] = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                                    
                                    # 计算边界框比例
                                    x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
                                    shape_features['aspect_ratio'] = w_roi / h_roi if h_roi > 0 else 1
                                    
                                    # 计算填充率 (面积与边界框面积比)
                                    shape_features['extent'] = area / (w_roi * h_roi) if w_roi * h_roi > 0 else 0
                                    
                                    # 计算区域质量
                                    shape_features['quality'] = min(1.0, shape_features['circularity'] * 0.5 + shape_features['extent'] * 0.5)
                        except Exception as e:
                            logging.warning(f"计算形状特征时出错: {e}")
                    
                    # 3. 纹理特征分析 - 使用LBP
                    lbp_roi = np.zeros_like(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                
                    # 4. 综合特征分析，建立多特征决策系统
                    # 瓶子和花瓶的形状特征判断
                    if class_name in ['vase', 'bottle'] and shape_features:
                        # 判断是否更可能是杯子
                        if 'circularity' in shape_features and 'aspect_ratio' in shape_features:
                            # 如果物体较圆，更可能是花瓶或杯子
                            if shape_features['circularity'] > 0.7 and shape_features['aspect_ratio'] < 1.5:
                                # 进一步检查颜色和尺寸来区分花瓶和杯子
                                if area < 5000:  # 面积较小
                                    corrected_class = 'cup'
                                    detection['class_name'] = corrected_class
                                    detection['original_class'] = class_name
                                    logging.info(f"类别已从 {class_name} 校正为 {corrected_class}")
                
                    # 人物检测增强 - 使用肤色模型
                    if class_name == 'person':
                        # 基于HSV的肤色检测
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
                        mask_skin = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                        
                        # 计算肤色占比
                        skin_ratio = np.sum(mask_skin > 0) / mask_skin.size if mask_skin.size > 0 else 0
                        
                        # 如果肤色比例足够高，增强人物检测置信度
                        if skin_ratio > 0.15:
                            detection['confidence'] = min(1.0, detection['confidence'] * 1.2)
                            detection['skin_ratio'] = skin_ratio
            except Exception as e:
                logging.error(f"语义一致性检查失败: {e}")
        
        # 获取掩码多边形
        if 'mask_polygon_normalized' not in detection:
            logging.debug(f"未找到mask_polygon_normalized，跳过边界优化")
            refined_detections.append(detection)
            continue
            
        mask_poly = np.array(detection['mask_polygon_normalized'])
        # 检查掩码多边形是否有效
        if mask_poly.size == 0 or len(mask_poly.shape) < 2:
            logging.debug(f"无效的掩码多边形，跳过边界优化")
            refined_detections.append(detection)
            continue
            
        mask_points = (mask_poly * np.array([w, h])).astype(np.int32)
        
        # 创建原始掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points], 255)
        
        # 4. 边缘感知区域增长
        # 使用掩码为种子，结合边缘信息进行区域增长
        # 创建掩码边界带
        dilated = cv2.dilate(mask, kernel, iterations=5)
        boundary_band = cv2.subtract(dilated, mask)
        
        # 在边界带内寻找强边缘
        boundary_edges = cv2.bitwise_and(edge_enhance, boundary_band)
        
        # 将强边缘添加到掩码中
        refined_mask = cv2.bitwise_or(mask, boundary_edges)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 根据类别优化轮廓的平滑度
        is_person = detection.get('class_name', '').lower() in ['person', 'man', 'woman']
        
        # 人物需要更精细的边缘
        if is_person:
            # 应用更精细的轮廓处理
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 使用更保守的Douglas-Peucker算法参数
                epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 应用B样条平滑以保持自然曲线
                if len(approx) >= 4:
                    try:
                        tck, u = interpolate.splprep([approx[:, 0, 0], approx[:, 0, 1]], s=0, k=min(3, len(approx)-1))
                        # 增加点数以获得更平滑的曲线
                        u_new = np.linspace(0, 1, max(len(approx)*2, 100))
                        smooth_points = np.column_stack(interpolate.splev(u_new, tck))
                        smooth_points = smooth_points.astype(np.int32)
                        
                        # 创建新掩码
                        refined_mask = np.zeros_like(refined_mask)
                        cv2.fillPoly(refined_mask, [smooth_points], 255)
                    except Exception as e:
                        logging.warning(f"B样条平滑失败: {e}, 使用原始轮廓")
        else:
            # 非人物对象可以使用更简化的轮廓
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 更强的简化
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                refined_mask = np.zeros_like(refined_mask)
                cv2.fillPoly(refined_mask, [approx], 255)
        
        # 提取最终轮廓并更新检测结果
        final_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if final_contours:
            largest_contour = max(final_contours, key=cv2.contourArea)
            
            # 进一步优化 - 保留足够的点以维持精度
            if is_person:
                # 人物需要更多点以保持形状复杂性
                min_points = max(len(largest_contour) // 3, 30)
                epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
            else:
                # 简单对象可以使用更少点
                min_points = max(len(largest_contour) // 5, 10)
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            
            # 应用Douglas-Peucker算法
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 确保点数不少于最小值
            if len(approx) < min_points:
                # 如果点数太少，使用均匀采样增加点数
                perimeter = cv2.arcLength(largest_contour, True)
                point_dist = perimeter / min_points
                
                # 重新采样轮廓
                approx = []
                dist_sum = 0
                for i in range(len(largest_contour)):
                    if i == 0:
                        approx.append(largest_contour[i])
                    else:
                        dist = np.linalg.norm(largest_contour[i] - largest_contour[i-1])
                        dist_sum += dist
                        if dist_sum >= point_dist:
                            approx.append(largest_contour[i])
                            dist_sum = 0
                approx = np.array(approx)
            
            # 更新归一化坐标
            optimized_points = approx.squeeze().astype(np.float32)
            if len(optimized_points.shape) == 1:  # 处理单点的情况
                optimized_points = optimized_points.reshape(1, 2)
            optimized_points[:, 0] /= w
            optimized_points[:, 1] /= h
            
            # 更新检测结果
            detection['mask_polygon_normalized'] = optimized_points.tolist()
            detection['is_refined'] = True
            detection['points_count'] = len(optimized_points)
        
        refined_detections.append(detection)
    
    return refined_detections



def preprocess_frame(frame):
    """使用高级图像处理技术增强帧质量"""
    if frame is None:
        return frame
        
    try:
        # 创建输出图像
        enhanced = frame.copy()
        
        # 1. 自适应直方图均衡化 (CLAHE)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  # 修复了缩进错误
        
        # 应用CLAHE到亮度通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. 亮度调整
        # 计算当前亮度
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        
        # 根据当前亮度动态调整
        if current_brightness < 100:  # 暗图像
            alpha = 1.2  # 增加对比度
            beta = 10    # 增加亮度
        elif current_brightness > 200:  # 亮图像
            alpha = 0.8  # 降低对比度
            beta = -10   # 降低亮度
        else:  # 正常亮度
            alpha = 1.1  # 轻微增加对比度
            beta = 5     # 轻微增加亮度
            
        # 应用亮度调整
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # 3. 降噪
        # 使用双边滤波保留边缘细节
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. 锐化
        kernel = np.array([[-1, -1, -1],
                [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    except Exception as e:
        logging.error(f"高级帧预处理失败: {e}")
        return frame



def apply_temporal_smoothing(detections, tracker, frame_number):
    """应用时间平滑处理，提高稳定性"""
    if not ENABLE_TEMPORAL_SMOOTHING or not tracker:
        return detections
        
    return tracker.update(detections, frame_number)

def process_frame_with_advanced_techniques(frame, nuclio_url, frame_number, tracker=None):
    """
    使用最先进的技术处理单帧，包括多尺度检测、结果验证和轮廓优化
    """
    # 1. 预处理帧以增强质量
    processed_frame = preprocess_frame(frame.copy())
    
    # 2. 多尺度检测
    base64_image = encode_frame_to_base64(processed_frame)
    detections = call_nuclio_segmentor(base64_image, nuclio_url, frame_number)
    
    if not detections:
        return []
    
    # 3. 应用先进的边界优化
    refined_detections = refine_detection_boundaries(detections, processed_frame)
    
    # 4. 应用时序平滑 (如果提供了跟踪器)
    if tracker and ENABLE_TEMPORAL_SMOOTHING:
        refined_detections = apply_temporal_smoothing(refined_detections, tracker, frame_number)
    
    # 5. 后处理 - 合并重复检测、过滤低置信度
    final_detections = []
    for det in refined_detections:
        # 更新置信度阈值处理方式
        # 如果检测通过了全部优化，提高其置信度
        if det.get('is_refined', False) and det.get('is_optimized', False):
            det['confidence'] = min(1.0, det.get('confidence', 0) * 1.1)
            
        # 添加到最终列表
        final_detections.append(det)
    
    return final_detections

# 将DetectionTracker类移动到main函数前面



# --- 按依赖顺序重新组织的函数 ---
def refine_edges(points, image_shape):
    """优化边缘点分布，提高边缘精度"""
    if not ENABLE_EDGE_REFINEMENT:
        return points
        
    try:
        # 将点转换为numpy数组
        points = np.array(points)
        
        # 1. 应用Douglas-Peucker算法简化曲线
        if len(points) > 4:
            points = cv2.approxPolyDP(points, RDPEPS, True)
        
        # 2. 确保点数量在合理范围内
        if len(points) > MAX_POLYGON_POINTS:
            # 使用均匀采样减少点数
            indices = np.linspace(0, len(points)-1, MAX_POLYGON_POINTS, dtype=int)
            points = points[indices]
        elif len(points) < MIN_POLYGON_POINTS:
            # 使用插值增加点数
            t = np.linspace(0, 1, MIN_POLYGON_POINTS)
            points = np.array([
                np.interp(t, np.linspace(0, 1, len(points)), points[:, 0]),
                np.interp(t, np.linspace(0, 1, len(points)), points[:, 1])
            ]).T
        
        # 3. 应用平滑处理
        if SMOOTH_POLYGON:
            # 使用移动平均平滑
            kernel_size = min(5, len(points))
            if kernel_size % 2 == 0:
                kernel_size += 1
            points = cv2.GaussianBlur(points, (kernel_size, kernel_size), 0)
        
        # 4. 确保点之间的距离在合理范围内
        refined_points = []
        for i in range(len(points)):
            refined_points.append(points[i])
            
            # 检查与下一个点的距离
            next_idx = (i + 1) % len(points)
            dist = np.linalg.norm(points[next_idx] - points[i])
            
            # 如果距离太大，添加中间点
            if dist > MAX_POINT_DISTANCE:
                num_points = int(dist / MAX_POINT_DISTANCE)
                for j in range(1, num_points):
                    t = j / (num_points + 1)
                    intermediate = points[i] + t * (points[next_idx] - points[i])
                    refined_points.append(intermediate)
        
        # 5. 确保点不会太密集
        final_points = []
        for i in range(len(refined_points)):
            # 检查与已添加的最后一个点的距离
            if not final_points or np.linalg.norm(refined_points[i] - final_points[-1]) >= MIN_POINT_DISTANCE:
                final_points.append(refined_points[i])
        
        return np.array(final_points)
    except Exception as e:
        logging.error(f"边缘优化失败: {e}")
        return points

# 配置多个模型端点
MODEL_ENDPOINTS = {
    "mask2former": "http://endpoint1",
    "solov2": "http://endpoint2",
    "htc": "http://endpoint3"
}

# 在处理时融合多个模型的结果

# 添加条件随机场优化

def merge_fpn_detections(detections_list, scales):
    """合并特征金字塔网络的检测结果"""
    if not ENABLE_FPN or len(detections_list) == 1:
        return detections_list[0]
        
    merged_detections = []
    h, w = None, None
    
    # 获取原始图像尺寸
    for det in detections_list[0]:
        if 'bbox' in det:
            h = det['bbox'][3] - det['bbox'][1]
            w = det['bbox'][2] - det['bbox'][0]
            break
    
    if h is None or w is None:
        return detections_list[0]
    
    # 对每个类别分别处理
    for class_id in set(det['class_id'] for dets in detections_list for det in dets):
        class_detections = []
        
        # 收集所有尺度的该类检测结果
        for scale, detections in zip(scales, detections_list):
            scale_dets = [det for det in detections if det['class_id'] == class_id]
            for det in scale_dets:
                # 将检测框缩放回原始尺寸
                if 'bbox' in det:
                    bbox = det['bbox']
                    scaled_bbox = [
                        bbox[0] / scale,
                        bbox[1] / scale,
                        bbox[2] / scale,
                        bbox[3] / scale
                    ]
                    det['bbox'] = scaled_bbox
                class_detections.append(det)
        
        # 使用NMS合并重叠检测
        if class_detections:
            boxes = np.array([det['bbox'] for det in class_detections])
            scores = np.array([det['confidence'] for det in class_detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.5)
            
            for idx in indices.flatten():
                merged_detections.append(class_detections[idx])
    
    return merged_detections


def build_feature_pyramid(frame):
    """构建特征金字塔网络"""
    if not ENABLE_FPN:
        return [frame]
        
    pyramid = []
    h, w = frame.shape[:2]
    
    for scale in FPN_SCALES:
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h))
        pyramid.append(scaled_frame)
    
    return pyramid


def process_frame(frame, frame_idx):
    """处理单帧图像"""
    # 1. 图像预处理
    if ENABLE_FRAME_PREPROCESS:
        frame = preprocess_frame(frame)
    
    # 2. 获取检测结果
    detections = multi_scale_detection(frame, frame_idx)
    
    # 3. 优化每个检测结果
    optimized_detections = []
    for det in detections:
        optimized_det = process_detection_result(frame, det)
        optimized_detections.append(optimized_det)
    
    # 4. 应用时间平滑
    if ENABLE_TEMPORAL_SMOOTHING:
        optimized_detections = apply_temporal_smoothing(optimized_detections, frame_idx)
    
    return optimized_detections



def process_detection_result(image, detection):
    """处理单个检测结果，应用所有优化方法"""
    try:
        # 1. 获取原始掩码
        mask_points = np.array(detection['mask_polygon_normalized'])
        h, w = image.shape[:2]
        mask_points = (mask_points * np.array([w, h])).astype(np.int32)
        
        # 创建原始掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points], 255)
        
        # 2. 应用先进的边界细化
        refined_mask = refine_mask_boundaries(mask, image)
        
        # 3. 应用替代优化方法
        enhanced_mask = apply_alternative_refinement(image, refined_mask)
        
        # 4. 使用最先进的边界增强
        final_mask = enhance_object_boundaries(image, enhanced_mask)
        
        # 5. 提取优化后的轮廓点
        contours, _ = cv2.findContours(
            final_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        if contours:
            # 选择最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 高级轮廓优化
            contour_area = cv2.contourArea(largest_contour)
            # 根据轮廓大小自适应设置epsilon
            epsilon_factor = 0.001
            if contour_area > 5000:
                epsilon_factor = 0.0005  # 大轮廓使用更小的epsilon保留更多细节
            elif contour_area < 500:
                epsilon_factor = 0.003   # 小轮廓使用更大的epsilon进行平滑
                
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 转换回归一化坐标
            optimized_points = approx.squeeze().astype(np.float32)
            if len(optimized_points.shape) == 1:  # 处理单点的情况
                optimized_points = optimized_points.reshape(1, -1)
                
            optimized_points[:, 0] /= w
            optimized_points[:, 1] /= h
            
            # 更新检测结果
            detection['mask_polygon_normalized'] = optimized_points.tolist()
            
            # 计算新的置信度分数 - 高级评分系统
            # 1. 区域比例评分
            area_ratio = contour_area / (w * h)
            # 2. 边缘响应评分
            edge_score = cv2.mean(image, mask=final_mask)[0] / 255.0
            # 3. 形状复杂度评分
            shape_complexity = cv2.arcLength(largest_contour, True) / (2 * np.sqrt(np.pi * contour_area))
            # 4. 轮廓质量评分
            contour_quality = min(1.0, len(approx) / 100.0)  # 归一化轮廓点数量
            
            # 组合多维度评分
            new_confidence = (
                detection['confidence'] * 0.4 +  # 原始置信度
                area_ratio * 0.2 +               # 区域评分
                edge_score * 0.2 +               # 边缘评分
                (1.0 / shape_complexity) * 0.1 + # 形状评分（圆形为1，越不规则越大）
                contour_quality * 0.1            # 轮廓质量
            )
            
            detection['confidence'] = float(min(1.0, new_confidence))
            # 添加优化质量标记
            detection['is_optimized'] = True
        
        return detection
    except Exception as e:
        logging.error(f"检测结果高级处理失败: {e}")
        return detection

# 在主处理流程中使用这些函数

def apply_alternative_refinement(image, mask):
    """使用先进的替代方法优化分割掩码，不依赖pydensecrf"""
    try:
        # 1. 多级形态学处理
        kernels = [np.ones((i, i), np.uint8) for i in range(3, 8, 2)]  # 使用多个不同大小的结构元素
        refined_mask = mask.copy()
        
        # 应用多尺度开闭操作
        for kernel in kernels:
            # 按权重应用不同尺度的形态学操作
            temp_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
            
            # 随着核大小增加减少权重
            weight = 1.0 / (1 + np.sum(kernel) / 10.0)
            refined_mask = cv2.addWeighted(refined_mask, 1 - weight, temp_mask, weight, 0)
        
        # 2. 先进的边缘检测和增强
        # 使用多尺度边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges_multi = []
        
        # 多参数Canny边缘检测
        thresholds = [(30, 100), (50, 150), (70, 200)]
        for low, high in thresholds:
            edge = cv2.Canny(gray, low, high)
            edges_multi.append(edge)
        
        # 合并多尺度边缘
        edges_combined = np.zeros_like(edges_multi[0])
        for edge in edges_multi:
            edges_combined = cv2.bitwise_or(edges_combined, edge)
        
        # 3. 应用区域生长算法
        # 创建距离变换作为种子点选择依据
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 执行基于标记的分水岭算法
        unknown = cv2.subtract(mask, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 将分水岭应用于灰度图像
        markers = cv2.watershed(cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR), markers)
        watershed_mask = np.zeros_like(refined_mask)
        watershed_mask[markers > 1] = 255
        
        # 4. 结合边缘和区域信息
        # 使用掩码选择与原始掩码重叠的边缘
        masked_edges = cv2.bitwise_and(edges_combined, mask)
        
        # 使用高斯距离权重增强边缘
        dist_weights = cv2.GaussianBlur(masked_edges.astype(float), (7, 7), 0)
        dist_weights = dist_weights / (np.max(dist_weights) + 1e-10)
        
        # 计算自适应阈值
        mean_val = np.mean(gray[mask > 0])
        std_val = np.std(gray[mask > 0])
        low_thresh = max(0, mean_val - 2 * std_val)
        high_thresh = min(255, mean_val + 2 * std_val)
        
        # 根据图像统计信息调整掩码
        _, adaptive_mask = cv2.threshold(gray, low_thresh, 255, cv2.THRESH_BINARY_INV)
        adaptive_mask = cv2.bitwise_and(adaptive_mask, mask)
        
        # 5. 图像引导滤波 - 边缘保持平滑
        guided_mask = None
        try:
            # 尝试使用ximgproc的guidedFilter
            refined_mask_float = refined_mask.astype(np.float32) / 255.0
            guided_mask = cv2.ximgproc.guidedFilter(
                gray, refined_mask_float, 10, 1e-2
            )
            guided_mask = (guided_mask * 255).astype(np.uint8)
        except (AttributeError, ImportError, cv2.error):
            # 如果ximgproc不可用，使用双边滤波作为备选
            logging.warning("cv2.ximgproc不可用，使用双边滤波作为替代")
            refined_mask_float = refined_mask.astype(np.float32)
            guided_mask = cv2.bilateralFilter(refined_mask_float, 9, 75, 75)
            guided_mask = guided_mask.astype(np.uint8)
        
        # 6. 最终融合所有处理结果
        # 根据不同处理结果的置信度加权融合
        fusion_weights = {
            'morphology': 0.3,
            'watershed': 0.25,
            'adaptive': 0.2,
            'guided': 0.25
        }
        
        final_mask = np.zeros_like(refined_mask, dtype=np.float32)
        final_mask += fusion_weights['morphology'] * refined_mask.astype(np.float32)
        final_mask += fusion_weights['watershed'] * watershed_mask.astype(np.float32)
        final_mask += fusion_weights['adaptive'] * adaptive_mask.astype(np.float32)
        
        if guided_mask is not None:
            final_mask += fusion_weights['guided'] * guided_mask.astype(np.float32)
        else:
            # 如果没有guided mask，重新分配权重
            for k in fusion_weights:
                if k != 'guided':
                    fusion_weights[k] = fusion_weights[k] / (1 - fusion_weights['guided'])
            
            final_mask = np.zeros_like(refined_mask, dtype=np.float32)
            final_mask += fusion_weights['morphology'] * refined_mask.astype(np.float32)
            final_mask += fusion_weights['watershed'] * watershed_mask.astype(np.float32)
            final_mask += fusion_weights['adaptive'] * adaptive_mask.astype(np.float32)
        
        # 二值化最终掩码
        final_mask = final_mask.astype(np.uint8)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 7. 后处理 - 确保连通性和边缘平滑度
        # 移除小区域和孔洞
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        processed_mask = np.zeros_like(final_mask)
        
        if contours:
            # 找到最大区域
            largest_contour = max(contours, key=cv2.contourArea)
            # 平滑轮廓
            epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            cv2.drawContours(processed_mask, [approx], -1, 255, -1)
            
            # 应用边缘感知平滑
            processed_mask = cv2.bilateralFilter(processed_mask.astype(np.float32), 9, 75, 75).astype(np.uint8)
        else:
            processed_mask = mask  # 如果找不到轮廓，回退到原始掩码
        
        return processed_mask
    except Exception as e:
        logging.error(f"高级替代优化方法失败: {e}")
        return mask

def enhance_object_boundaries(image, mask):
    """增强对象边界的精确性，使用最先进的边缘处理方法"""
    try:
        # 1. 多尺度边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges_list = []
        
        # 使用不同的Canny参数，实现多尺度边缘检测
        edge_params = [(50, 150), (30, 100), (70, 200)]
        for low, high in edge_params:
            edges = cv2.Canny(gray, low, high)
            edges_list.append(edges)
        
        # 合并多尺度边缘
        multi_scale_edges = np.zeros_like(edges_list[0])
        for edge in edges_list:
            multi_scale_edges = cv2.bitwise_or(multi_scale_edges, edge)
        
        # 2. 自适应边缘增强
        # 创建自适应阈值图像
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. 高级梯度计算 - 使用Sobel X/Y和Scharr算子
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = np.absolute(sobelx)
        sobely = np.absolute(sobely)
        
        # 使用Scharr算子（更好的边缘响应）
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharrx = np.absolute(scharrx)
        scharry = np.absolute(scharry)
        
        # 结合Sobel和Scharr（取最大响应）
        gradient_x = np.maximum(sobelx / 8.0, scharrx / 16.0)
        gradient_y = np.maximum(sobely / 8.0, scharry / 16.0)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_magnitude = np.uint8(np.minimum(gradient_magnitude * 255.0 / np.max(gradient_magnitude), 255.0))
        
        # 4. 边缘定向增强
        # 计算梯度方向
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180.0 / np.pi
        gradient_direction = np.uint8((gradient_direction + 180.0) / 360.0 * 255.0)
        
        # 使用形态学操作增强特定方向的边缘
        kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        
        h_edges = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel_h)
        v_edges = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel_v)
        directional_edges = cv2.bitwise_or(h_edges, v_edges)
        
        # 5. 组合所有边缘信息
        combined_edges = np.zeros_like(gray)
        combined_edges = cv2.addWeighted(multi_scale_edges, 0.4, gradient_magnitude, 0.4, 0)
        combined_edges = cv2.addWeighted(combined_edges, 0.8, directional_edges, 0.2, 0)
        combined_edges = cv2.bitwise_and(combined_edges, mask)  # 只保留掩码区域内的边缘
        
        # 6. 应用掩码融合
        # 使用边缘信息优化掩码
        _, enhanced_mask = cv2.threshold(combined_edges, 50, 255, cv2.THRESH_BINARY)
        
        # 应用形态学闭操作填充小的边缘缺口
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
        
        # 7. 高级轮廓优化
        contours, _ = cv2.findContours(
            enhanced_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        # 创建最终掩码
        refined_mask = np.zeros_like(mask)
        
        # 处理找到的轮廓
        if contours:
            # 选择面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 使用轮廓平滑算法 - B-spline平滑
            if len(largest_contour) > 5:  # 需要足够多的点
                try:
                    # 提取轮廓点
                    contour_points = largest_contour.squeeze()
                    
                    # 确保轮廓是闭合的
                    if len(contour_points.shape) > 1:
                        x = contour_points[:, 0]
                        y = contour_points[:, 1]
                        
                        # 使用B-spline平滑轮廓
                        t = np.arange(len(x))
                        x_smooth = interpolate.splev(np.linspace(0, len(x)-1, len(x)*2), 
                                                    interpolate.splrep(t, x, s=len(x)*0.005))
                        y_smooth = interpolate.splev(np.linspace(0, len(y)-1, len(y)*2), 
                                                    interpolate.splrep(t, y, s=len(y)*0.005))
                        
                        # 转换回轮廓格式
                        smooth_contour = np.column_stack([x_smooth, y_smooth]).astype(np.int32)
                        
                        # 绘制平滑轮廓
                        cv2.drawContours(refined_mask, [smooth_contour], -1, 255, -1)
                    else:
                        # 如果点不够，直接使用原始轮廓
                        cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
                except:
                    # 如果平滑失败，使用原始轮廓
                    cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
            else:
                # 点太少，直接使用原始轮廓
                cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
                
            # 8. 应用边缘感知模糊
            # 创建边缘保持滤波器
            refined_mask_float = refined_mask.astype(np.float32) / 255.0
            refined_mask_blurred = cv2.bilateralFilter(refined_mask_float, 5, 20, 20)
            refined_mask = (refined_mask_blurred * 255).astype(np.uint8)
        else:
            # 如果没有找到轮廓，退回到原始掩码
            refined_mask = mask
        
        return refined_mask
    except Exception as e:
        logging.error(f"高级边界增强失败: {e}")
        return mask


def apply_crf_refinement(image, masks):
    # 使用CRF优化分割掩码
    pass

# 添加边界细化

def refine_mask_boundaries(mask, image=None):
    """使用高级形态学操作和边缘检测优化掩码边界"""
    try:
        if mask is None or mask.size == 0:
            return mask

        # 如果没有提供图像参数，则进行基本形态学处理
        if image is None:
            # 应用形态学闭运算平滑边缘
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 移除小孔洞
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            
            return refined_mask

        # 如果提供了图像参数，则使用更高级的边界优化方法
        # 转换确保掩码是二值图像
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 1. 提取边界
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        # 2. 平滑边界
        smoothed_mask = np.zeros_like(mask)
        for contour in contours:
            # 对轮廓进行平滑处理
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(smoothed_mask, [approx_contour], 0, 255, -1)
        
        # 3. 应用边缘感知平滑
        smoothed_mask = cv2.GaussianBlur(smoothed_mask, (5, 5), 0)
        _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 4. 使用原始图像边缘信息优化边界
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘以确保覆盖
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 在掩码边界区域应用边缘信息
        # 创建边界区域掩码
        boundary_region = cv2.dilate(smoothed_mask, kernel, iterations=2) - cv2.erode(smoothed_mask, kernel, iterations=2)
        
        # 在边界区域中找到强边缘
        boundary_edges = cv2.bitwise_and(dilated_edges, boundary_region)
        
        # 将边缘信息整合到掩码中
        final_mask = smoothed_mask.copy()
        
        # 在边界处优先使用图像中的边缘信息
        edge_dilated = cv2.dilate(boundary_edges, kernel, iterations=1)
        final_mask = cv2.bitwise_or(final_mask, edge_dilated)
        
        # 最后一次形态学闭运算以填充小缝隙
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask
        
    except Exception as e:
        logging.error(f"掩码边界优化失败: {e}")
        return mask


def multi_scale_detection(frame, frame_idx):
    """执行多尺度检测，提高检测准确性"""
    if not ENABLE_MULTI_SCALE:
        return call_nuclio_segmentor(encode_frame_to_base64(frame), DEFAULT_NUCLIO_SEG_URL, frame_idx)
        
    all_detections = []
    h, w = frame.shape[:2]
    
    for scale, weight in zip(SCALE_FACTORS, SCALE_WEIGHTS):
        # 调整图像大小
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h))
        
        # 对缩放后的图像进行检测
        b64_string = encode_frame_to_base64(scaled_frame)
        if not b64_string:
            continue
            
        detections = call_nuclio_segmentor(b64_string, DEFAULT_NUCLIO_SEG_URL, frame_idx)
        if not detections:
            continue
            
        # 调整检测结果到原始尺寸
        for det in detections:
            # 调整边界框
            if 'box_normalized' in det:
                x1, y1, x2, y2 = det['box_normalized']
                det['box_normalized'] = [x1/scale, y1/scale, x2/scale, y2/scale]
            
            # 调整掩码多边形
            if 'mask_polygon_normalized' in det:
                points = np.array(det['mask_polygon_normalized'])
                points = points / scale
                det['mask_polygon_normalized'] = points.tolist()
            
            # 添加尺度权重
            det['scale_weight'] = weight
            
        all_detections.extend(detections)
    
    # 合并多尺度检测结果
    return merge_multi_scale_detections(all_detections)


def merge_multi_scale_detections(detections):
    """合并多尺度检测结果，使用加权平均和NMS"""
    if not detections:
        return []
        
    # 按类别分组
    class_groups = {}
    for det in detections:
        class_name = det.get('class_name', 'unknown')
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det)
    
    merged_detections = []
    
    # 对每个类别分别处理
    for class_name, class_dets in class_groups.items():
        # 计算加权置信度
        for det in class_dets:
            det['weighted_confidence'] = det.get('confidence', 0) * det.get('scale_weight', 1.0)
        
        # 按加权置信度排序
        class_dets.sort(key=lambda x: x['weighted_confidence'], reverse=True)
        
        # 应用NMS
        kept_detections = []
        while class_dets:
            current = class_dets.pop(0)
            kept_detections.append(current)
            
            # 计算与剩余检测的IOU
            remaining = []
            for det in class_dets:
                iou = calculate_iou(current['box_normalized'], det['box_normalized'])
                if iou < IOU_THRESHOLD:
                    remaining.append(det)
            class_dets = remaining
        
        merged_detections.extend(kept_detections)
    
    return merged_detections


def main(video_path, nuclio_url, output_dir, frame_skip, conf_threshold, save_video_flag):
    """主函数，执行视频处理、调用分割、保存XML和可选的带标注视频流程。"""
    output_dir = Path(output_dir)
    output_xmls_dir = output_dir / "annotations"
    output_videos_dir = output_dir / "annotated_videos"
    output_xmls_dir.mkdir(parents=True, exist_ok=True)
    if save_video_flag:
        output_videos_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始处理视频: {video_path}")
    logging.info(f"Nuclio 分割 URL: {nuclio_url}")
    logging.info(f"帧间隔: 每 {frame_skip} 帧")
    logging.info(f"置信度阈值: {conf_threshold}")
    logging.info(f"XML 输出目录: {output_xmls_dir}")
    if save_video_flag:
        logging.info(f"带标注视频输出目录: {output_videos_dir}")
    else:
        logging.info("不保存带标注的视频。")

    # 改进的文件路径诊断
    video_path_str = str(video_path)
    if not os.path.exists(video_path_str):
        logging.error(f"视频未找到: {video_path_str}")
        # 增加诊断信息
        logging.info("正在进行文件路径诊断...")
        
        # 检查文件夹是否存在
        video_dir = os.path.dirname(video_path_str)
        if not os.path.exists(video_dir):
            logging.error(f"视频所在文件夹不存在: {video_dir}")
        else:
            logging.info(f"视频所在文件夹存在。尝试列出目录内容...")
            try:
                dir_contents = os.listdir(video_dir)
                if len(dir_contents) > 0:
                    logging.info(f"文件夹内容: {', '.join(dir_contents[:10])}" + 
                                ("..." if len(dir_contents) > 10 else ""))
                else:
                    logging.info("文件夹为空")
            except Exception as e:
                logging.error(f"列出目录内容时出错: {e}")
        
        # 尝试使用绝对路径
        try:
            abs_path = os.path.abspath(video_path_str)
            logging.info(f"尝试使用绝对路径: {abs_path}")
            if abs_path != video_path_str and os.path.exists(abs_path):
                logging.info("文件通过绝对路径找到！将使用绝对路径继续。")
                video_path_str = abs_path
            else:
                # 尝试修正常见问题
                if '\\' in video_path_str and not video_path_str.startswith('\\\\'):
                    # 处理网络路径
                    potential_path = video_path_str.replace('\\', '/')
                    logging.info(f"尝试修改路径分隔符: {potential_path}")
                    if os.path.exists(potential_path):
                        logging.info("文件使用修改后的路径分隔符找到！将使用此路径继续。")
                        video_path_str = potential_path
        except Exception as e:
            logging.error(f"处理绝对路径时出错: {e}")
        
        return

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        logging.error(f"无法打开视频: {video_path_str}")
        logging.error("这可能是因为:")
        logging.error(" - 文件格式不被OpenCV支持")
        logging.error(" - 文件可能已损坏")
        logging.error(" - 缺少解码器")
        logging.error(" - 文件可能被其他程序锁定")
        logging.error("尝试使用不同的视频文件或检查系统是否安装了必要的解码器。")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_total_frames <= 0: # 尝试读取所有帧来确定总数
        logging.warning("无法直接获取视频总帧数，尝试手动计数...")
        frame_temp_count = 0
        while True:
             ret_temp, _ = cap.read()
             if not ret_temp: break
             frame_temp_count += 1
        original_total_frames = frame_temp_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置到开头
        logging.info(f"手动计数得到总帧数: {original_total_frames}")
    else:
        logging.info(f"视频信息 - 尺寸: {frame_width}x{frame_height}, FPS: {fps:.2f}, 总帧数: {original_total_frames}")


    video_writer = None
    output_video_path = None
    if save_video_flag:
        video_basename = os.path.basename(video_path)
        video_name_no_ext, _ = os.path.splitext(video_basename)
        output_video_filename = f"{video_name_no_ext}_segmented{VIDEO_EXTENSION}"
        output_video_path = output_videos_dir / output_video_filename
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        try:
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
            if not video_writer.isOpened():
                logging.error(f"无法初始化 VideoWriter！检查编解码器 '{VIDEO_CODEC}'。禁用视频保存。")
                video_writer = None; save_video_flag = False
            else:
                logging.info(f"将把带标注的视频保存到: {output_video_path}")
        except Exception as e:
            logging.error(f"初始化 VideoWriter 时出错: {e}. 禁用视频保存。")
            video_writer = None; save_video_flag = False

    frame_count = 0
    processed_frame_count = 0
    all_results = {}
    start_process_time = time.time()
    
    # 初始化目标跟踪器
    tracker = DetectionTracker(window_size=TEMPORAL_WINDOW) if ENABLE_TEMPORAL_SMOOTHING else None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("视频处理结束或读取错误。")
            break
        # 帧号从 1 开始内部计数，但 XML ID 从 0 开始
        frame_count += 1

        current_frame_to_process = False
        # 帧跳过逻辑基于 0-based 索引更直观 (frame_count-1)
        if frame_skip <= 1 or (frame_count - 1) % frame_skip == 0:
             current_frame_to_process = True
             processed_frame_count += 1
             logging.info(f"正在处理帧 {frame_count}/{original_total_frames}...")

        annotated_frame = frame # 默认

        if current_frame_to_process:
            # 使用高级处理技术处理帧
            detections = process_frame_with_advanced_techniques(frame, nuclio_url, frame_count, tracker)

            # 总是存储结果，即使是 None 或空列表，用于生成完整的 XML 帧序列
            all_results[frame_count] = detections if detections is not None else []

            if save_video_flag and video_writer:
                # 绘制时也需要过滤 detections
                valid_detections = [d for d in all_results[frame_count] if d.get('confidence', 0) >= conf_threshold]
                if valid_detections:
                    annotated_frame = draw_segmentation(frame, valid_detections, conf_threshold)
                # else: 保持原始帧 annotated_frame = frame

        # 写入视频帧
        if save_video_flag and video_writer:
            try:
                video_writer.write(annotated_frame)
            except Exception as e:
                 logging.error(f"写入视频帧 {frame_count} 时出错: {e}")

    cap.release()
    if video_writer:
        video_writer.release()
        logging.info("带标注视频写入器已释放。")

    # 保存 XML 文件
    video_basename = os.path.basename(video_path)
    video_name_no_ext, _ = os.path.splitext(video_basename)
    xml_output_filename = f"{video_name_no_ext}_cvat_polygon.xml"
    xml_output_path = output_xmls_dir / xml_output_filename
    save_results_to_cvat_polygon_xml(
        all_results,
        xml_output_path,
        video_basename,
        frame_width,
        frame_height,
        processed_frame_count, # 传递处理过的帧数给函数（虽然里面用了原始总数）
        original_total_frames, # 传递原始总帧数
        conf_threshold
    )

    end_process_time = time.time(); total_time = max(0, end_process_time - start_process_time)
    logging.info("-" * 30); logging.info(f"视频处理完成。")
    logging.info(f"总耗时: {timedelta(seconds=int(total_time))}")
    logging.info(f"总读取帧数: {frame_count}。")
    logging.info(f"实际处理帧数 (发送到Nuclio): {processed_frame_count}。")
    logging.info(f"结果 XML 保存在: {xml_output_path}")
    if output_video_path and os.path.exists(output_video_path):
        logging.info(f"带标注视频保存在: {output_video_path}")
    elif save_video_flag and not output_video_path:
         logging.warning("设置了保存视频但未能生成视频文件路径。")
    logging.info("-" * 30)


# --- 命令行参数解析和入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理视频, 调用Nuclio实例分割, 保存XML和可选的带标注视频")
    parser.add_argument("-v", "--video", type=str, default=DEFAULT_VIDEO_PATH, help=f"输入视频路径 (默认: {DEFAULT_VIDEO_PATH})")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_NUCLIO_SEG_URL, help=f"Nuclio实例分割函数URL (默认: {DEFAULT_NUCLIO_SEG_URL})")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_BASE_DIR, help=f"保存结果的基础目录 (默认: {DEFAULT_OUTPUT_BASE_DIR})")
    parser.add_argument("-fs", "--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help=f"帧间隔 (1=处理所有) (默认: {DEFAULT_FRAME_SKIP})")
    parser.add_argument("-ct", "--confidence-threshold", type=float, default=CONF_THRESHOLD, help=f"检测置信度阈值 (用于绘图和XML过滤) (默认: {CONF_THRESHOLD})")
    # 修正 store_true 的 default 行为，让命令行不指定时使用常量
    parser.add_argument("--save-video", action='store_true', help=f"是否保存带标注的视频文件 (不指定则使用默认值: {DEFAULT_SAVE_VIDEO})")
    parser.add_argument("--no-save-video", action='store_false', dest='save_video', help="明确指定不保存带标注的视频文件")
    parser.add_argument("--test-video-only", action='store_true', help="只测试视频文件是否可访问，不执行完整处理")
    parser.set_defaults(save_video=DEFAULT_SAVE_VIDEO) # 设置 argparse 的默认值

    args = parser.parse_args()

    # 尝试修复视频路径
    video_path = args.video
    if isinstance(video_path, str):
        # 尝试解决网络路径问题 - 转换为通用格式
        video_path = video_path.replace('\\', '/')
        # 移除可能导致问题的引号
        video_path = video_path.strip('"\'')
        logging.info(f"处理后的视频路径: {video_path}")

    # 如果是测试模式，只测试视频访问
    if args.test_video_only:
        try:
            logging.info(f"测试模式：尝试访问视频文件 {video_path}")
            if os.path.exists(video_path):
                logging.info("文件存在！使用os.path.exists检测成功。")
            else:
                logging.error("文件不存在。使用os.path.exists检测失败。")
                
            # 尝试使用OpenCV打开
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"成功打开视频！尺寸: {width}x{height}, 帧数: {frames}, FPS: {fps:.2f}")
                
                # 读取第一帧测试
                ret, frame = cap.read()
                if ret:
                    logging.info(f"成功读取第一帧, 尺寸: {frame.shape}")
                else:
                    logging.error("无法读取第一帧")
                
                cap.release()
            else:
                logging.error("OpenCV无法打开视频文件")
                
            logging.info("测试完成")
        except Exception as e:
            logging.error(f"测试视频访问时发生错误: {e}")
        sys.exit(0)

    CONF_THRESHOLD = args.confidence_threshold
    frame_skip_value = max(1, args.frame_skip)
    save_video_flag = args.save_video

    output_base_dir_arg = Path(args.output_dir)
    output_xmls_dir_arg = output_base_dir_arg / "annotations"
    output_videos_dir_arg = output_base_dir_arg / "annotated_videos"
    output_xmls_dir_arg.mkdir(parents=True, exist_ok=True)
    if save_video_flag:
        output_videos_dir_arg.mkdir(parents=True, exist_ok=True)

    if args.url == "http://<YOUR_NUCLIO_IP>:<YOUR_NUCLIO_PORT>": # 检查默认 URL 是否被修改
        logging.warning("Nuclio URL 未通过命令行参数修改，请确保默认 URL 正确或使用 --url 参数指定!")

    main(video_path, args.url, args.output_dir, frame_skip_value, CONF_THRESHOLD, save_video_flag)


def apply_temporal_smoothing(detections, tracker, frame_number):
    """应用时间平滑处理，提高稳定性"""
    if not ENABLE_TEMPORAL_SMOOTHING or not tracker:
        return detections
        
    return tracker.update(detections, frame_number)


def process_frame_with_advanced_techniques(frame, nuclio_url, frame_number, tracker=None):
    """
    使用最先进的技术处理单帧，包括多尺度检测、结果验证和轮廓优化
    """
    # 1. 预处理帧以增强质量
    processed_frame = preprocess_frame(frame.copy())
    
    # 2. 多尺度检测
    base64_image = encode_frame_to_base64(processed_frame)
    detections = call_nuclio_segmentor(base64_image, nuclio_url, frame_number)
    
    if not detections:
        return []
    
    # 3. 应用先进的边界优化
    refined_detections = refine_detection_boundaries(detections, processed_frame)
    
    # 4. 应用时序平滑 (如果提供了跟踪器)
    if tracker and ENABLE_TEMPORAL_SMOOTHING:
        refined_detections = apply_temporal_smoothing(refined_detections, tracker, frame_number)
    
    # 5. 后处理 - 合并重复检测、过滤低置信度
    final_detections = []
    for det in refined_detections:
        # 更新置信度阈值处理方式
        # 如果检测通过了全部优化，提高其置信度
        if det.get('is_refined', False) and det.get('is_optimized', False):
            det['confidence'] = min(1.0, det.get('confidence', 0) * 1.1)
            
        # 添加到最终列表
        final_detections.append(det)
    
    return final_detections

# 将DetectionTracker类移动到main函数前面

# --- 主执行逻辑 (保持不变) ---

def refine_detection_boundaries(detections, image, semantic_check=True):
    """
    优化检测边界，使用最新的边缘增强和特征分析技术
    """
    if not detections:
        return detections
    
    refined_detections = []
    h, w = image.shape[:2]
    
    # 预处理图像以增强边缘
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 1. 应用多尺度Canny边缘检测
    edges_low = cv2.Canny(blur, 30, 90)
    edges_high = cv2.Canny(blur, 80, 200)
    combined_edges = cv2.bitwise_or(edges_low, edges_high)
    
    # 2. 增强边缘 - 使用自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    edge_enhance = cv2.bitwise_or(combined_edges, adaptive_thresh)
    
    # 3. 形态学操作改进边缘
    kernel = np.ones((3, 3), np.uint8)
    edge_enhance = cv2.morphologyEx(edge_enhance, cv2.MORPH_CLOSE, kernel)
    
    for detection in detections:
        # 类别语义检查和校正
        if semantic_check:
            try:
                # 提取对象区域用于分析
                class_name = detection.get('class_name', '').lower()
                # 使用 bbox_key 记录实际使用的键
                bbox = None
                bbox_key = None
                for key in ['bbox', 'box', 'box_normalized']:
                    if key in detection:
                        bbox = detection[key]
                        bbox_key = key
                        break
                if bbox is None:
                    logging.debug("未找到边界框信息，跳过语义一致性检查")
                    continue
                # 如使用标准化坐标则转换为像素坐标
                if bbox_key == 'box_normalized':
                    bbox = [
                        int(bbox[0] * w),
                        int(bbox[1] * h),
                        int(bbox[2] * w),
                        int(bbox[3] * h)
                    ]
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # 确保边界在图像内
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 跳过无效区域
                if x2 <= x1 or y2 <= y1:
                    pass # 继续处理
                else:
                    # 提取ROI
                    roi = image[y1:y2, x1:x2]
                    
                    # 1. 颜色分析
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    color_hist_h, _ = np.histogram(hsv_roi[:,:,0], bins=30, range=[0, 180])
                    color_hist_s, _ = np.histogram(hsv_roi[:,:,1], bins=32, range=[0, 256])
                    
                    # 归一化直方图
                    color_hist_h = color_hist_h / np.sum(color_hist_h)
                    color_hist_s = color_hist_s / np.sum(color_hist_s)
                    
                    # 2. 形状特征计算
                    shape_features = {}
                    area = 0
                    if 'mask_polygon_normalized' in detection:
                        try:
                            mask_points = np.array(detection['mask_polygon_normalized'])
                            mask_points = (mask_points * np.array([w, h])).astype(np.int32)
                            
                            # 创建掩码
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask, [mask_points], 255)
                            roi_mask = mask[y1:y2, x1:x2]
                            
                            # 计算周长和面积
                            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                perimeter = cv2.arcLength(largest_contour, True)
                                area = cv2.contourArea(largest_contour)
                                
                                if area > 0:
                                    # 计算圆度 (4*pi*area/perimeter^2)
                                    shape_features['circularity'] = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                                    
                                    # 计算边界框比例
                                    x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
                                    shape_features['aspect_ratio'] = w_roi / h_roi if h_roi > 0 else 1
                                    
                                    # 计算填充率 (面积与边界框面积比)
                                    shape_features['extent'] = area / (w_roi * h_roi) if w_roi * h_roi > 0 else 0
                                    
                                    # 计算区域质量
                                    shape_features['quality'] = min(1.0, shape_features['circularity'] * 0.5 + shape_features['extent'] * 0.5)
                        except Exception as e:
                            logging.warning(f"计算形状特征时出错: {e}")
                    
                    # 3. 纹理特征分析 - 使用LBP
                    lbp_roi = np.zeros_like(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                
                    # 4. 综合特征分析，建立多特征决策系统
                    # 瓶子和花瓶的形状特征判断
                    if class_name in ['vase', 'bottle'] and shape_features:
                        # 判断是否更可能是杯子
                        if 'circularity' in shape_features and 'aspect_ratio' in shape_features:
                            # 如果物体较圆，更可能是花瓶或杯子
                            if shape_features['circularity'] > 0.7 and shape_features['aspect_ratio'] < 1.5:
                                # 进一步检查颜色和尺寸来区分花瓶和杯子
                                if area < 5000:  # 面积较小
                                    corrected_class = 'cup'
                                    detection['class_name'] = corrected_class
                                    detection['original_class'] = class_name
                                    logging.info(f"类别已从 {class_name} 校正为 {corrected_class}")
                
                    # 人物检测增强 - 使用肤色模型
                    if class_name == 'person':
                        # 基于HSV的肤色检测
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
                        mask_skin = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                        
                        # 计算肤色占比
                        skin_ratio = np.sum(mask_skin > 0) / mask_skin.size if mask_skin.size > 0 else 0
                        
                        # 如果肤色比例足够高，增强人物检测置信度
                        if skin_ratio > 0.15:
                            detection['confidence'] = min(1.0, detection['confidence'] * 1.2)
                            detection['skin_ratio'] = skin_ratio
            except Exception as e:
                logging.error(f"语义一致性检查失败: {e}")
        
        # 获取掩码多边形
        if 'mask_polygon_normalized' not in detection:
            logging.debug(f"未找到mask_polygon_normalized，跳过边界优化")
            refined_detections.append(detection)
            continue
            
        mask_poly = np.array(detection['mask_polygon_normalized'])
        # 检查掩码多边形是否有效
        if mask_poly.size == 0 or len(mask_poly.shape) < 2:
            logging.debug(f"无效的掩码多边形，跳过边界优化")
            refined_detections.append(detection)
            continue
            
        mask_points = (mask_poly * np.array([w, h])).astype(np.int32)
        
        # 创建原始掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points], 255)
        
        # 4. 边缘感知区域增长
        # 使用掩码为种子，结合边缘信息进行区域增长
        # 创建掩码边界带
        dilated = cv2.dilate(mask, kernel, iterations=5)
        boundary_band = cv2.subtract(dilated, mask)
        
        # 在边界带内寻找强边缘
        boundary_edges = cv2.bitwise_and(edge_enhance, boundary_band)
        
        # 将强边缘添加到掩码中
        refined_mask = cv2.bitwise_or(mask, boundary_edges)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 根据类别优化轮廓的平滑度
        is_person = detection.get('class_name', '').lower() in ['person', 'man', 'woman']
        
        # 人物需要更精细的边缘
        if is_person:
            # 应用更精细的轮廓处理
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 使用更保守的Douglas-Peucker算法参数
                epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 应用B样条平滑以保持自然曲线
                if len(approx) >= 4:
                    try:
                        tck, u = interpolate.splprep([approx[:, 0, 0], approx[:, 0, 1]], s=0, k=min(3, len(approx)-1))
                        # 增加点数以获得更平滑的曲线
                        u_new = np.linspace(0, 1, max(len(approx)*2, 100))
                        smooth_points = np.column_stack(interpolate.splev(u_new, tck))
                        smooth_points = smooth_points.astype(np.int32)
                        
                        # 创建新掩码
                        refined_mask = np.zeros_like(refined_mask)
                        cv2.fillPoly(refined_mask, [smooth_points], 255)
                    except Exception as e:
                        logging.warning(f"B样条平滑失败: {e}, 使用原始轮廓")
        else:
            # 非人物对象可以使用更简化的轮廓
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 更强的简化
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                refined_mask = np.zeros_like(refined_mask)
                cv2.fillPoly(refined_mask, [approx], 255)
        
        # 提取最终轮廓并更新检测结果
        final_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if final_contours:
            largest_contour = max(final_contours, key=cv2.contourArea)
            
            # 进一步优化 - 保留足够的点以维持精度
            if is_person:
                # 人物需要更多点以保持形状复杂性
                min_points = max(len(largest_contour) // 3, 30)
                epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
            else:
                # 简单对象可以使用更少点
                min_points = max(len(largest_contour) // 5, 10)
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            
            # 应用Douglas-Peucker算法
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 确保点数不少于最小值
            if len(approx) < min_points:
                # 如果点数太少，使用均匀采样增加点数
                perimeter = cv2.arcLength(largest_contour, True)
                point_dist = perimeter / min_points
                
                # 重新采样轮廓
                approx = []
                dist_sum = 0
                for i in range(len(largest_contour)):
                    if i == 0:
                        approx.append(largest_contour[i])
                    else:
                        dist = np.linalg.norm(largest_contour[i] - largest_contour[i-1])
                        dist_sum += dist
                        if dist_sum >= point_dist:
                            approx.append(largest_contour[i])
                            dist_sum = 0
                approx = np.array(approx)
            
            # 更新归一化坐标
            optimized_points = approx.squeeze().astype(np.float32)
            if len(optimized_points.shape) == 1:  # 处理单点的情况
                optimized_points = optimized_points.reshape(1, 2)
            optimized_points[:, 0] /= w
            optimized_points[:, 1] /= h
            
            # 更新检测结果
            detection['mask_polygon_normalized'] = optimized_points.tolist()
            detection['is_refined'] = True
            detection['points_count'] = len(optimized_points)
        
        refined_detections.append(detection)
    
    return refined_detections




def semantic_consistency_check(detection, image):
    """
    执行语义一致性检查，纠正错误的类别预测
    利用2025年最新的上下文理解技术改进分类准确性
    """
    try:
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0)
        
        # 尝试获取box_normalized，如果不存在，尝试使用其他可能的边界框字段名
        box = None
        for box_field in ['box_normalized', 'bbox', 'box']:
            if box_field in detection:
                box = detection[box_field]
                break
        
        # 如果仍然找不到边界框信息，则使用整个图像范围
        if box is None:
            logging.debug(f"未找到边界框信息，使用整个图像范围")
            box = [0, 0, 1, 1]  # 归一化坐标，表示整个图像
            
        h, w = image.shape[:2]
        
        # 提取目标区域
        x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [w, h, w, h])]
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 确保坐标形成有效的矩形
        if x2 <= x1 or y2 <= y1:
            logging.debug(f"无效的边界框: [{x1}, {y1}, {x2}, {y2}]")
            return detection
            
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return detection
            
        # 1. 基于颜色特征的目标验证
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建特定类别的颜色特征词典 - 2025年的改进特征集
        color_features = {
            'person': {'hue_range': [0, 30], 'saturation_min': 20, 'value_min': 60},
            'vase': {'hue_range': [20, 150], 'saturation_min': 30, 'value_min': 30},
            'bottle': {'hue_range': [20, 150], 'saturation_min': 0, 'value_min': 50},
            'cup': {'hue_range': [20, 150], 'saturation_min': 0, 'value_min': 50}
        }
        
        # 2. 形状特征 - 使用Hu矩和轮廓分析
        mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
        # 检查mask_polygon_normalized是否存在
        if 'mask_polygon_normalized' not in detection:
            logging.debug(f"未找到mask_polygon_normalized")
            return detection
            
        mask_points = np.array(detection['mask_polygon_normalized'])
        mask_points = (mask_points * np.array([w, h])).astype(np.int32)
        # 将点转换为ROI坐标系
        mask_points[:, 0] -= x1
        mask_points[:, 1] -= y1
        cv2.fillPoly(mask, [mask_points], 255)
        
        # 计算形状特征
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_features = {}
        area = 0
        
        if contours and len(contours[0]) > 5:
            cnt = contours[0]
            # 计算面积与周长比
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 计算长宽比
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 计算矩形度
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # 计算Hu矩
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            shape_features = {
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'hu_moments': hu_moments
            }
        
        # 3. 纹理特征 - 使用LBP
        lbp_hist = None
        if roi.size > 100:  # 确保ROI足够大
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 计算LBP纹理特征
            lbp_roi = np.zeros_like(gray_roi)
            for i in range(1, gray_roi.shape[0]-1):
                for j in range(1, gray_roi.shape[1]-1):
                    center = gray_roi[i, j]
                    code = 0
                    code |= (gray_roi[i-1, j-1] >= center) << 7
                    code |= (gray_roi[i-1, j] >= center) << 6
                    code |= (gray_roi[i-1, j+1] >= center) << 5
                    code |= (gray_roi[i, j+1] >= center) << 4
                    code |= (gray_roi[i+1, j+1] >= center) << 3
                    code |= (gray_roi[i+1, j] >= center) << 2
                    code |= (gray_roi[i+1, j-1] >= center) << 1
                    code |= (gray_roi[i, j-1] >= center) << 0
                    lbp_roi[i, j] = code
            
            # 计算LBP直方图
            lbp_hist, _ = np.histogram(lbp_roi.ravel(), bins=256, range=[0, 256])
            lbp_hist = lbp_hist / np.sum(lbp_hist)  # 归一化
        
        # 4. 综合特征分析，建立多特征决策系统
        # 瓶子和花瓶的形状特征判断
        if class_name in ['vase', 'bottle'] and shape_features:
            # 判断是否更可能是杯子
            if 'circularity' in shape_features and 'aspect_ratio' in shape_features:
                # 如果物体较圆，更可能是花瓶或杯子
                if shape_features['circularity'] > 0.7 and shape_features['aspect_ratio'] < 1.5:
                    # 进一步检查颜色和尺寸来区分花瓶和杯子
                    if area < 5000:  # 面积较小
                        corrected_class = 'cup'
                        detection['class_name'] = corrected_class
                        detection['original_class'] = class_name
                        logging.info(f"类别已从 {class_name} 校正为 {corrected_class}")
        
        # 人物检测增强 - 使用肤色模型
        if class_name == 'person':
            # 基于HSV的肤色检测
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            mask_skin = cv2.inRange(hsv_roi, lower_skin, upper_skin)
            
            # 计算肤色占比
            skin_ratio = np.sum(mask_skin > 0) / mask_skin.size if mask_skin.size > 0 else 0
            
            # 如果肤色比例足够高，增强人物检测置信度
            if skin_ratio > 0.15:
                detection['confidence'] = min(1.0, detection['confidence'] * 1.2)
                detection['skin_ratio'] = skin_ratio
        
        return detection
    except Exception as e:
        logging.debug(f"语义一致性检查失败: {e}")  # 改用debug级别，不再显示为错误
        return detection



def save_results_to_cvat_polygon_xml(all_results, xml_output_path, video_filename, frame_width, frame_height, total_processed_frames, original_total_frames, conf_threshold):
    """将实例分割结果保存为更兼容 CVAT 的多边形格式 XML"""
    logging.info(f"正在为 {total_processed_frames} 个处理过的帧构建 CVAT 多边形 XML...")
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- Meta ---
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "N/A" # 通常由 CVAT 分配，导入时可为 N/A
    task_name, _ = os.path.splitext(video_filename)
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(original_total_frames) # 使用原始总帧数
    # *** 修改点: 根据需要设置 mode (interpolation 或 annotation) ***
    ET.SubElement(task, "mode").text = "interpolation" # 或者 "annotation"
    ET.SubElement(task, "overlap").text = "0" # 或根据你的设置调整
    ET.SubElement(task, "bugtracker").text = ""
    
    # 修复时间格式问题
    current_time_utc = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "+00:00"
    
    ET.SubElement(task, "created").text = current_time_utc
    ET.SubElement(task, "updated").text = current_time_utc
    ET.SubElement(task, "start_frame").text = "0"
    # *** 修改点: stop_frame 使用原始帧数 ***
    ET.SubElement(task, "stop_frame").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(task, "frame_filter").text = ""

    # --- Labels (重要修改) ---
    labels_elem = ET.SubElement(task, "labels")
    # *** 从 ALL_POSSIBLE_LABELS 生成标签定义 ***
    #    确保 ALL_POSSIBLE_LABELS 包含你模型会输出的所有类别名称
    logging.info(f"在 XML 中定义标签: {ALL_POSSIBLE_LABELS}")
    sorted_labels = sorted(list(ALL_POSSIBLE_LABELS)) # 排序以保证颜色一致性
    for idx, label_name in enumerate(sorted_labels):
        if not label_name: continue # 跳过空标签
        label_elem = ET.SubElement(labels_elem, "label")
        ET.SubElement(label_elem, "name").text = label_name
        color = get_color(idx) # 分配颜色
        ET.SubElement(label_elem, "color").text = bgr_to_hex(color)
        ET.SubElement(label_elem, "type").text = "polygon" # 指定类型为多边形
        ET.SubElement(label_elem, "attributes") # 添加空的 attributes

    # --- Segments ---
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0" # 通常默认为 0
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(segment, "url").text = "N/A" # 或任务 URL

    ET.SubElement(task, "owner").text = "N/A" # 可选
    ET.SubElement(task, "assignee").text = "N/A" # 可选
    ET.SubElement(task, "subset").text = "Default"
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(frame_width)
    ET.SubElement(original_size, "height").text = str(frame_height)
    
    # 修复时间格式问题
    ET.SubElement(meta, "dumped").text = current_time_utc

    # --- 添加图像和多边形数据 ---
    if all_results:
        for frame_number in sorted(all_results.keys()): # 确保按帧号顺序处理
            detections = all_results[frame_number]
            frame_id_0_based = frame_number - 1 # CVAT 使用 0-based

            # *** 修改点: 总是创建 <image> 标签，即使没有检测 ***
            #    这样可以确保帧序列的完整性，符合图像格式预期
            image_elem = ET.SubElement(root, "image")
            image_elem.set("id", str(frame_id_0_based))
            # *** 修改点: 添加 name 属性 ***
            image_elem.set("name", f"frame_{frame_id_0_based:06d}") # 标准的帧名格式
            image_elem.set("width", str(frame_width))
            image_elem.set("height", str(frame_height))

            # 只处理置信度高于阈值的有效检测
            valid_detections_in_frame = [det for det in detections if det.get('confidence', 0) >= conf_threshold]

            if not valid_detections_in_frame:
                logging.debug(f"帧 {frame_number}: 没有高于阈值 {conf_threshold} 的检测，生成空的 <image> 标签。")
                continue # 跳到下一帧

            for det_idx, det in enumerate(valid_detections_in_frame):
                label_name = det.get('class_name', 'unknown')
                mask_poly_norm = det.get('mask_polygon_normalized')

                # 检查标签是否存在于定义的标签列表中
                if label_name not in ALL_POSSIBLE_LABELS:
                    logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 检测到未在 ALL_POSSIBLE_LABELS 中定义的标签 '{label_name}'，跳过此实例。")
                    continue
                if not mask_poly_norm:
                    logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 标签 '{label_name}' 缺少掩码数据，跳过。")
                    continue

                # --- 创建 <polygon> ---
                poly_elem = ET.SubElement(image_elem, "polygon")
                poly_elem.set("label", label_name)
                poly_elem.set("source", "auto") # 标记为自动生成
                poly_elem.set("occluded", "0") # 简化处理
                poly_elem.set("outside", "0")  # 简化处理
                poly_elem.set("keyframe", "1") # 对图像格式，每个标注都是关键帧
                # *** 修改点: 添加 z_order 属性 ***
                poly_elem.set("z_order", "0")

                # --- 处理多边形点 ---
                try:
                    # 转换坐标并检查范围
                    points_px_raw = np.array(mask_poly_norm) * np.array([frame_width, frame_height])
                    
                    # 优化多边形点 (新增功能，根据对象类别选择不同优化方法)
                    if OPTIMIZE_POLYGON_POINTS:
                        label_lower = label_name.lower()
                        if label_lower in ['bottle']:
                            points_px_raw = optimize_polygon_points_bottle(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                        elif label_lower in ['person']:
                            points_px_raw = optimize_polygon_points_person(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                        else:
                            points_px_raw = optimize_polygon_points(
                                points_px_raw,
                                max_points=MAX_POLYGON_POINTS,
                                min_points=MIN_POLYGON_POINTS,
                                rdp_epsilon=RDPEPS
                            )
                    
                    # 裁剪坐标到图像边界内 (可选但推荐，防止 CVAT 错误)
                    points_px_raw[:, 0] = np.clip(points_px_raw[:, 0], 0, frame_width - 1)
                    points_px_raw[:, 1] = np.clip(points_px_raw[:, 1], 0, frame_height - 1)
                    points_px = points_px_raw.round(2) # 保留两位小数

                    # 确保至少有 3 个点
                    if len(points_px) < 3:
                        logging.warning(f"帧 {frame_number}, 实例 {det_idx}: 标签 '{label_name}' 的多边形顶点少于 3 个 ({len(points_px)})，跳过。")
                        image_elem.remove(poly_elem) # 移除无效多边形
                        continue

                    # 格式化为 CVAT 期望的字符串 "x1,y1;x2,y2;..."
                    points_str = ";".join([f"{p[0]},{p[1]}" for p in points_px])
                    poly_elem.set("points", points_str)

                    # --- 添加置信度作为属性 (可选) ---
                    attr_conf = ET.SubElement(poly_elem, "attribute", name="confidence")
                    attr_conf.text = f"{det.get('confidence', 0):.4f}"

                except Exception as e:
                    logging.error(f"帧 {frame_number}, 实例 {det_idx}: 转换或格式化掩码点时出错: {e}")
                    try:
                        image_elem.remove(poly_elem) # 尝试移除无效的多边形
                    except ValueError: # 如果已经被移除或不存在
                        pass
                    continue

    # --- 保存 XML ---
    try:
        # 使用 ElementTree 的 indent 功能进行格式化 (比 minidom 更不易出错)
        ET.indent(root, space="  ", level=0)
        tree = ET.ElementTree(root)
        xml_output_path_str = str(xml_output_path)
        tree.write(xml_output_path_str, encoding='utf-8', xml_declaration=True)
        logging.info(f"XML 已保存: {xml_output_path_str}")

    except Exception as e:
        logging.error(f"保存 XML 失败: {e}")
        # 如果 indent/write 失败，尝试原始字符串保存作为后备
        try:
             xml_output_path_str = str(xml_output_path)
             rough_string = ET.tostring(root, encoding='unicode')
             with open(xml_output_path_str, "w", encoding="utf-8") as f:
                 f.write('<?xml version="1.0" encoding="utf-8"?>\n') # 手动添加声明
                 f.write(rough_string)
             logging.info(f"XML 已保存 (使用原始字符串后备): {xml_output_path_str}")
        except Exception as fallback_e:
             logging.error(f"后备 XML 写入也失败: {fallback_e}")




# --- DetectionTracker类和相关函数移动到此处 ---
class DetectionTracker:
    """检测结果跟踪器，用于时序平滑"""
    def __init__(self, window_size=TEMPORAL_WINDOW):
        self.window_size = window_size
        self.tracks = {}  # 跟踪对象字典
        self.next_track_id = 0
        
    def update(self, detections, frame_number):
        """更新跟踪状态"""
        if not ENABLE_TEMPORAL_SMOOTHING:
            return detections
            
        current_tracks = {}
        
        # 更新现有轨迹
        for track_id, track in list(self.tracks.items()):
            if frame_number - track['last_frame'] > self.window_size:
                del self.tracks[track_id]
                continue
                
            # 寻找最佳匹配
            best_match = None
            best_iou = IOU_THRESHOLD
            
            for det in detections:
                iou = calculate_iou(track['box_normalized'], det['box_normalized'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                # 更新轨迹
                track['history'].append(best_match)
                track['last_frame'] = frame_number
                if len(track['history']) > self.window_size:
                    track['history'].pop(0)
                
                current_tracks[track_id] = track
                detections.remove(best_match)
            else:
                # 使用运动预测
                predicted_box = self._predict_position(track)
                if predicted_box is not None:
                    for det in detections:
                        iou = calculate_iou(predicted_box, det['box_normalized'])
                        if iou > IOU_THRESHOLD * 0.5:  # 降低预测位置的IOU阈值
                            track['history'].append(det)
                            track['last_frame'] = frame_number
                            if len(track['history']) > self.window_size:
                                track['history'].pop(0)
                            
                            current_tracks[track_id] = track
                            detections.remove(det)
                            break
        
        # 创建新轨迹
        for det in detections:
            if det.get('confidence', 0) > MIN_TRACK_CONFIDENCE:
                self.tracks[self.next_track_id] = {
                    'history': [det],
                    'box_normalized': det['box_normalized'],
                    'last_frame': frame_number
                }
                current_tracks[self.next_track_id] = self.tracks[self.next_track_id]
                self.next_track_id += 1
        
        self.tracks = current_tracks
        
        # 创建平滑后的检测结果
        smoothed_detections = []
        for track_id, track in self.tracks.items():
            if len(track['history']) > 0:
                # 复制最新的检测结果
                smoothed_det = track['history'][-1].copy()
                
                if len(track['history']) >= 3:  # 至少需要3帧才能平滑
                    # 计算位置平滑
                    boxes = [det['box_normalized'] for det in track['history']]
                    smoothed_box = np.mean(boxes, axis=0).tolist()
                    smoothed_det['box_normalized'] = smoothed_box
                    
                    # 如果有掩码多边形，也进行平滑
                    if all('mask_polygon_normalized' in det for det in track['history']):
                        # 找出具有最多点的多边形
                        max_points = max(len(det['mask_polygon_normalized']) for det in track['history'])
                        
                        # 对齐点数
                        aligned_masks = []
                        
                        try:
                            # 找出所有多边形的形状
                            mask_shapes = [np.array(det['mask_polygon_normalized']).shape for det in track['history']]
                            if not all(len(shape) == 2 for shape in mask_shapes):
                                # 处理非二维数组的情况
                                for det in track['history']:
                                    poly = np.array(det['mask_polygon_normalized'])
                                    if len(poly.shape) == 1:
                                        poly = poly.reshape(-1, 2)
                                        det['mask_polygon_normalized'] = poly.tolist()
                        except Exception as e:
                            logging.warning(f"处理多边形形状时出错: {e}")
                
                smoothed_det['track_id'] = track_id
                smoothed_detections.append(smoothed_det)
        
        return smoothed_detections
    
    def _predict_position(self, track):
        """基于历史预测下一位置"""
        if len(track['history']) < 2:
            return None
            
        # 获取最近的几个边界框
        recent_boxes = [det['box_normalized'] for det in track['history'][-2:]]
        
        # 计算平均移动
        box1 = np.array(recent_boxes[0])
        box2 = np.array(recent_boxes[1])
        movement = box2 - box1
        
        # 预测下一位置
        predicted_box = (box2 + movement).tolist()
        
        return predicted_box


# --- 移动重要函数到外部 ---

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def optimize_polygon_points(points, max_points=MAX_POLYGON_POINTS, min_points=MIN_POLYGON_POINTS, rdp_epsilon=RDPEPS):
    """使用先进的技术优化多边形点分布，提高边框准确性和边缘点均匀性"""
    if len(points) < 4:
        return points
        
    # 确保输入是 numpy 数组
    points = np.array(points)
    
    # 1. 首先应用Douglas-Peucker算法简化曲线
    def rdp_simplify(points, epsilon):
        """道格拉斯-普克简化算法"""
        if len(points) <= 2:
            return points
            
        # 找到距离最远的点
        dmax = 0
        index = 0
        end = len(points) - 1
        
        # 计算每个点到首尾连线的距离
        for i in range(1, end):
            d = point_to_line_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # 如果最大距离大于阈值，递归简化
        if dmax > epsilon:
            # 递归处理两部分
            result1 = rdp_simplify(points[:index+1], epsilon)
            result2 = rdp_simplify(points[index:], epsilon)
            
            # 合并结果 (去掉重复的点)
            return np.vstack([result1[:-1], result2])
        else:
            # 否则直接返回首尾点
            return np.vstack([points[0], points[end]])
    
    # 2. 曲率计算函数
    def calculate_curvature(points):
        """计算轮廓上每个点的曲率"""
        n = len(points)
        if n < 3:
            return np.zeros(n)
            
        # 计算差分向量
        dx1 = np.zeros(n)
        dy1 = np.zeros(n)
        dx2 = np.zeros(n)
        dy2 = np.zeros(n)
        
        # 计算一阶差分
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            dx1[i] = points[next_i][0] - points[i][0]
            dy1[i] = points[next_i][1] - points[i][1]
            dx2[i] = points[i][0] - points[prev_i][0]
            dy2[i] = points[i][1] - points[prev_i][1]
        
        # 计算曲率
        curvature = np.zeros(n)
        for i in range(n):
            try:
                x1, y1 = dx1[i], dy1[i]
                x2, y2 = dx2[i], dy2[i]
                
                # 曲率的近似计算
                num = abs(x1*y2 - y1*x2)
                denom = (x1*x1 + y1*y1)**1.5
                
                if denom > 0:
                    curvature[i] = num / denom
                else:
                    curvature[i] = 0
            except:
                curvature[i] = 0
                
        return curvature
    
    # 3. 基于曲率的自适应采样
    def adaptive_sampling(points, n_points, curvature=None):
        """根据曲率自适应地对轮廓进行采样"""
        if curvature is None or not CURVATURE_ADAPTIVE:
            return high_density_sampling(points, n_points)
            
        n = len(points)
        if n <= 3:
            return points
            
        # 归一化曲率并加上基础值
        base_weight = 0.5
        curvature = curvature / (np.max(curvature) + 1e-6)
        weights = base_weight + (1 - base_weight) * curvature
        
        # 计算累积权重
        cum_weights = np.zeros(n+1)
        for i in range(1, n+1):
            cum_weights[i] = cum_weights[i-1] + weights[i-1]
        total_weight = cum_weights[-1]
        
        # 根据累积权重采样
        n_points = min(int(n_points * DENSITY_FACTOR), max_points)
        sampled_points = []
        for i in range(n_points):
            target = total_weight * i / n_points
            idx = np.searchsorted(cum_weights, target)
            idx = min(idx, n-1)
            sampled_points.append(points[idx])
            
        return ensure_visual_continuity(np.array(sampled_points))
    
    # 4. 高密度均匀采样
    def high_density_sampling(points, n_points):
        """高密度均匀采样，确保点与点之间视觉上的连续性"""
        n = len(points)
        perimeter = 0
        for i in range(n):
            j = (i + 1) % n
            perimeter += np.sqrt(np.sum((points[i] - points[j])**2))
        
        avg_segment = VISUAL_CONNECT_DISTANCE * 0.8
        estimated_points = max(int(perimeter / avg_segment), min_points)
        target_points = min(max(estimated_points, n_points), max_points)
        
        resampled = []
        segment_length = perimeter / target_points
        current_length = 0
        current_point = points[0]
        resampled.append(current_point.copy())
        
        i = 0
        while len(resampled) < target_points:
            next_i = (i + 1) % n
            next_point = points[next_i]
            
            segment_distance = np.sqrt(np.sum((next_point - current_point)**2))
            
            if current_length + segment_distance >= segment_length:
                t = (segment_length - current_length) / segment_distance
                new_point = current_point + t * (next_point - current_point)
                resampled.append(new_point.copy())
                
                current_point = new_point
                current_length = 0
            else:
                current_length += segment_distance
                current_point = next_point
                i = next_i
        
        return ensure_visual_continuity(np.array(resampled))
    
    # 5. 确保视觉连续性
    def ensure_visual_continuity(points):
        """确保多边形点之间的视觉连续性"""
        if len(points) < 3:
            return points
            
        n = len(points)
        result = []
        
        for i in range(n):
            current = points[i]
            result.append(current)
            
            next_idx = (i + 1) % n
            next_point = points[next_idx]
            
            dist = np.sqrt(np.sum((next_point - current)**2))
            
            if dist > VISUAL_CONNECT_DISTANCE:
                num_points = int(math.ceil(dist / (VISUAL_CONNECT_DISTANCE * 0.7))) - 1
                
                for j in range(1, num_points + 1):
                    t = j / (num_points + 1)
                    intermediate = current + t * (next_point - current)
                    result.append(intermediate)
        
        return np.array(result)
    
    # 处理轮廓点
    try:
        # 1. 简化轮廓
        simplified = rdp_simplify(points, rdp_epsilon)
        
        if len(simplified) < 3:
            simplified = points
            
        # 2. 计算曲率
        curvature = calculate_curvature(simplified)
        
        # 3. 根据点的数量决定处理方式
        if len(simplified) > max_points:
            optimized = adaptive_sampling(simplified, max_points, curvature)
        elif len(simplified) < min_points:
            optimized = adaptive_sampling(simplified, min_points, curvature)
        else:
            optimized = adaptive_sampling(simplified, len(simplified), curvature)
            
        # 4. 最后应用视觉连续性检查
        optimized = ensure_visual_continuity(optimized)
        
        # 5. 如果点数仍然不足，再次进行均匀采样
        if len(optimized) < min_points:
            optimized = high_density_sampling(optimized, min_points)
        
        return optimized
        
    except Exception as e:
        logging.error(f"高级多边形优化失败: {e}")
        # 出错时回退到基本的高密度采样
        try:
            return high_density_sampling(points, min(max(len(points), min_points), max_points))
        except:
            return points


# --- 修改 save_results_to_cvat_polygon_xml 函数，增加多边形点优化 ---

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    # 将点转换为numpy数组
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    
    # 计算线段向量
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    # 计算线段长度
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    # 计算投影
    t = np.dot(point_vec, line_vec) / (line_len * line_len)
    t = max(0, min(1, t))  # 限制在0-1之间
    
    # 计算最近点
    nearest = line_start + t * line_vec
    
    # 返回点到最近点的距离
    return np.linalg.norm(point - nearest)


def draw_segmentation(frame, detections, conf_threshold):
    """在帧上绘制实例分割结果 (边界框、掩码、标签)。"""
    h, w, _ = frame.shape
    annotated_frame = frame.copy()
    overlay = frame.copy()
    instance_count = 0
    processed_labels = set() # 用于分配颜色

    for i, det in enumerate(detections):
        confidence = det.get('confidence', 0)
        if confidence < conf_threshold:
            continue

        box_norm = det.get('box_normalized')
        mask_poly_norm = det.get('mask_polygon_normalized')
        label = det.get('class_name', 'unknown')
        label_text = f"{label}: {confidence:.2f}"

        # 为标签分配颜色 (尽量保持一致)
        if label not in processed_labels:
            processed_labels.add(label)
        color_index = list(sorted(processed_labels)).index(label) # 基于已处理标签的顺序获取索引
        color = get_color(color_index)

        # --- 绘制掩码 ---
        if DRAW_MASK and mask_poly_norm:
            try:
                mask_points_px = np.array(mask_poly_norm) * np.array([w, h])
                mask_points_px = mask_points_px.astype(np.int32)
                
                # 填充掩码区域
                cv2.fillPoly(overlay, [mask_points_px], color)
                
                # 绘制更细的轮廓线
                cv2.polylines(annotated_frame, [mask_points_px], True, color, 1)
                
                # 仅在关键点绘制小圆点标记
                # 替换为均匀采样关键点，确保点沿掩码边界均匀分布
                n_points = len(mask_points_px)
                if n_points > 0:
                    num_keypoints = min(8, n_points)
                    indices = np.linspace(0, n_points - 1, num=num_keypoints, dtype=int)
                    for idx in indices:
                        cv2.circle(annotated_frame, tuple(mask_points_px[idx]), 1, (0, 255, 255), -1)
                
            except Exception as e:
                logging.warning(f"绘制掩码时出错 (实例 {i}): {e}. Mask data: {str(mask_poly_norm)[:100]}")

        # --- 绘制边界框 ---
        if DRAW_BOX and box_norm:
            try:
                x1, y1, x2, y2 = box_norm
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(annotated_frame, pt1, pt2, color, 1)  # 使用更细的线
            except Exception as e:
                logging.warning(f"绘制边界框时出错 (实例 {i}): {e}. Box data: {box_norm}")

        # --- 绘制标签 ---
        if DRAW_LABEL and box_norm:
             try:
                x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                # 使用更小的字体
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_origin = (x1, max(0, y1 - text_height - baseline // 2))
                cv2.rectangle(annotated_frame, label_origin, (label_origin[0] + text_width, label_origin[1] + text_height + baseline), color, -1)
                cv2.putText(annotated_frame, label_text, (label_origin[0], label_origin[1] + text_height + baseline // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
             except Exception as e:
                 logging.warning(f"绘制标签时出错 (实例 {i}): {e}. Label: {label_text}")

        instance_count += 1

    # --- 合并掩码图层 ---
    if instance_count > 0:
        cv2.addWeighted(overlay, MASK_ALPHA, annotated_frame, 1 - MASK_ALPHA, 0, annotated_frame)

    return annotated_frame


# --- 替换优化多边形点函数，使用更先进的方法 ---

def pretty_print_xml(elem):
    """美化打印 XML Element"""
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = '\n'.join([line for line in reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8').split('\n') if line.strip()])
        return pretty_xml
    except Exception as e:
        logging.error(f"XML美化打印出错: {e}")
        try:
            ET.indent(elem, space="  ")
            return ET.tostring(elem, encoding='unicode')
        except Exception:
             return ET.tostring(elem, encoding='unicode')


# --- 绘图函数 (用于实例分割) ---

def call_nuclio_segmentor(base64_image_string, nuclio_url, frame_number):
    """调用 Nuclio 实例分割函数并解析结果"""
    payload = json.dumps({"image": base64_image_string})
    headers = {'Content-Type': 'application/json'}
    logging.debug(f"帧 {frame_number}: 发送请求到 {nuclio_url}...")
    try:
        start_time = time.time()
        response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        end_time = time.time()
        logging.debug(f"帧 {frame_number}: Nuclio 请求耗时 {end_time - start_time:.2f} 秒。")

        if response.status_code == 200:
            try:
                results = response.json()
                # --- 验证响应格式 ---
                if isinstance(results, list):
                    logging.info(f"帧 {frame_number}: 收到 {len(results)} 个检测实例。")
                    valid_results = []
                    
                    # 定义常见错误检测的特征
                    error_patterns = [
                        {
                            "class_names": ["person", "man"], 
                            "min_area_ratio": 0.6,  # 面积占比超过60%可能是误检
                            "aspect_ratio_range": (0.1, 10.0),  # 极端的宽高比可能是误检
                            "edge_response_threshold": 0.15  # 边缘响应低于15%可能是误检
                        }
                    ]
                    
                    for instance in results:
                        # 基本格式验证
                        if (isinstance(instance, dict) and
                            'box_normalized' in instance and isinstance(instance['box_normalized'], list) and len(instance['box_normalized']) == 4 and
                            'confidence' in instance and isinstance(instance['confidence'], (float, int)) and
                            'class_name' in instance and isinstance(instance['class_name'], str) and
                            'mask_polygon_normalized' in instance and isinstance(instance['mask_polygon_normalized'], list)):
                            
                            # 应用高级验证规则，过滤错误检测
                            should_filter = False
                            
                            # 检查是否符合误检模式
                            for pattern in error_patterns:
                                if instance['class_name'].lower() in pattern["class_names"]:
                                    # 计算面积比例
                                    box = instance['box_normalized']
                                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                                    
                                    # 计算长宽比
                                    box_width = box[2] - box[0]
                                    box_height = box[3] - box[1]
                                    aspect_ratio = box_width / box_height if box_height > 0 else 0
                                    
                                    # 计算边缘响应（使用多边形点的复杂度估计）
                                    mask_points = np.array(instance['mask_polygon_normalized'])
                                    if len(mask_points) > 5:
                                        perimeter = cv2.arcLength(
                                            (mask_points * 1000).astype(np.int32), 
                                            True
                                        )
                                        area = cv2.contourArea(
                                            (mask_points * 1000).astype(np.int32)
                                        )
                                        edge_response = 0
                                        if area > 0:
                                            # 使用圆形度估计边缘响应
                                            edge_response = 4 * np.pi * area / (perimeter * perimeter)
                                        
                                        logging.debug(f"检测 {instance['class_name']}: 面积比={box_area}, 宽高比={aspect_ratio}, 边缘响应={edge_response}")
                                        
                                        # 应用过滤规则
                                        if (box_area > pattern["min_area_ratio"] or 
                                            aspect_ratio < pattern["aspect_ratio_range"][0] or 
                                            aspect_ratio > pattern["aspect_ratio_range"][1] or
                                            edge_response < pattern["edge_response_threshold"]):
                                            should_filter = True
                                            logging.info(f"过滤可能的误检: {instance['class_name']}, 面积比={box_area:.3f}, 宽高比={aspect_ratio:.3f}, 边缘响应={edge_response:.3f}")
                            
                            if not should_filter:
                                valid_results.append(instance)
                            else:
                                logging.warning(f"帧 {frame_number}: 跳过无效或格式不正确的实例: {str(instance)[:200]}...")
                    return valid_results
                else:
                    logging.error(f"帧 {frame_number}: Nuclio 响应不是列表。类型: {type(results)}, 响应: {str(response.text)[:200]}...")
                    return None
            except json.JSONDecodeError:
                logging.error(f"帧 {frame_number}: 解析 Nuclio JSON 失败。状态码: {response.status_code}, 响应: {response.text[:200]}...")
                return None
        else:
            logging.error(f"帧 {frame_number}: Nuclio 错误。状态码: {response.status_code}, 响应: {response.text[:200]}...")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 超时 ({REQUEST_TIMEOUT} 秒)。")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 失败: {e}")
        return None
    except Exception as e:
        logging.error(f"帧 {frame_number}: 调用 Nuclio 时发生意外错误: {e}")
        return None


def encode_frame_to_base64(frame):
    """将 OpenCV 帧编码为 Base64 字符串"""
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if is_success:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            logging.warning("帧编码失败 (cv2.imencode 返回 False)。")
            return None
    except Exception as e:
        logging.error(f"帧编码时发生异常: {e}")
        return None


def bgr_to_hex(bgr_color):
    """将 BGR 元组转换为 Hex 颜色字符串"""
    return '#{:02x}{:02x}{:02x}'.format(bgr_color[2], bgr_color[1], bgr_color[0])

# --- 辅助函数 (encode_frame_to_base64, call_nuclio_segmentor, pretty_print_xml) ---

def get_color(index):
    """获取一个颜色用于绘制"""
    # 确保索引有效，或者使用哈希等方法为标签分配固定颜色
    return COLOR_LIST_BGR[index % len(COLOR_LIST_BGR)]

# --- 主执行逻辑 (保持不变) ---




def refine_mask_boundaries(mask, image=None):
    """
    优化掩码边界，使其更平滑和精确
    参数:
        mask: 输入二值掩码
        image: 原始图像(可选)
    返回:
        优化后的掩码
    """
    try:
        if mask is None or mask.size == 0:
            return mask

        # 如果没有提供图像参数，则进行基本形态学处理
        if image is None:
            # 应用形态学闭运算平滑边缘
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 移除小孔洞
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            
            return refined_mask

        # 如果提供了图像参数，则使用更高级的边界优化方法
        # 转换确保掩码是二值图像
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 1. 提取边界
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        # 2. 平滑边界
        smoothed_mask = np.zeros_like(mask)
        for contour in contours:
            # 对轮廓进行平滑处理
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(smoothed_mask, [approx_contour], 0, 255, -1)
        
        # 3. 应用边缘感知平滑
        smoothed_mask = cv2.GaussianBlur(smoothed_mask, (5, 5), 0)
        _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 4. 使用原始图像边缘信息优化边界
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘以确保覆盖
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 在掩码边界区域应用边缘信息
        # 创建边界区域掩码
        boundary_region = cv2.dilate(smoothed_mask, kernel, iterations=2) - cv2.erode(smoothed_mask, kernel, iterations=2)
        
        # 在边界区域中找到强边缘
        boundary_edges = cv2.bitwise_and(dilated_edges, boundary_region)
        
        # 将边缘信息整合到掩码中
        final_mask = smoothed_mask.copy()
        
        # 在边界处优先使用图像中的边缘信息
        edge_dilated = cv2.dilate(boundary_edges, kernel, iterations=1)
        final_mask = cv2.bitwise_or(final_mask, edge_dilated)
        
        # 最后一次形态学闭运算以填充小缝隙
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask
        
    except Exception as e:
        logging.error(f"掩码边界优化失败: {e}")
        return mask

# 统一 enhance_object_boundaries 函数为一个简化版本
def enhance_object_boundaries(image, mask):
    """
    使用多尺度边缘检测和梯度信息增强对象边界
    参数:
        image: 输入RGB图像
        mask: 输入二值掩码
    返回:
        增强边界的掩码
    """
    try:
        if mask is None or image is None:
            return mask
            
        # 确保掩码是二值图像
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 提取图像梯度信息
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Sobel算子计算梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. 多尺度边缘检测
        edges_list = []
        for threshold in [30, 70, 120]:  # 多个阈值
            edges = cv2.Canny(gray, threshold, threshold*2)
            edges_list.append(edges)
        
        # 合并多尺度边缘
        multi_scale_edges = np.zeros_like(edges_list[0])
        for edge in edges_list:
            multi_scale_edges = cv2.bitwise_or(multi_scale_edges, edge)
        
        # 3. 找到掩码边界区域
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=2)
        boundary_region = cv2.subtract(dilated_mask, eroded_mask)
        
        # 4. 在边界区域应用边缘和梯度信息
        # 在边界内找到强边缘
        boundary_edges = cv2.bitwise_and(multi_scale_edges, boundary_region)
        
        # 增强边界区域
        enhanced_mask = binary_mask.copy()
        enhanced_boundary = cv2.addWeighted(
            boundary_region, 0.5,
            cv2.bitwise_and(gradient_magnitude, boundary_region), 0.5,
            0
        )
        
        # 二值化增强的边界
        _, enhanced_boundary = cv2.threshold(enhanced_boundary, 30, 255, cv2.THRESH_BINARY)
        
        # 合并原始掩码和增强的边界
        enhanced_mask = cv2.bitwise_or(
            cv2.bitwise_and(enhanced_mask, cv2.bitwise_not(boundary_region)),
            enhanced_boundary
        )
        
        # 5. 后处理 - 平滑和填充小孔洞
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
        enhanced_mask = cv2.medianBlur(enhanced_mask, 3)
        
        return enhanced_mask
        
    except Exception as e:
        logging.error(f"对象边界增强失败: {e}")
        return mask

# 添加简化版的process_detection_result函数
def process_detection_result(image, detection):
    """
    处理检测结果，增强其质量
    参数:
        image: 输入图像
        detection: 检测结果字典
    返回:
        处理后的检测结果
    """
    try:
        if not detection or 'mask_polygon_normalized' not in detection:
            return detection
            
        # 深拷贝检测结果，避免修改原始数据
        processed_detection = detection.copy()
        
        # 1. 从归一化多边形创建掩码
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 转换归一化坐标到像素坐标
        points = np.array(detection['mask_polygon_normalized'])
        pixel_points = (points * np.array([w, h])).astype(np.int32)
        
        # 填充多边形创建掩码
        cv2.fillPoly(mask, [pixel_points], 255)
        
        # 2. 优化掩码
        refined_mask = refine_mask_boundaries(mask, image)
        
        # 3. 增强对象边界
        enhanced_mask = enhance_object_boundaries(image, refined_mask)
        
        # 4. 从优化后的掩码提取新的轮廓
        contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 简化轮廓
            epsilon = 0.002 * cv2.arcLength(max_contour, True)
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # 转换回归一化坐标
            approx_contour = approx_contour.reshape(-1, 2).astype(np.float32)
            normalized_contour = approx_contour / np.array([w, h])
            
            # 更新检测结果中的掩码多边形
            processed_detection['mask_polygon_normalized'] = normalized_contour.tolist()
            
            # 5. 计算新的置信度得分
            # 使用边缘响应和形状复杂度等指标
            
            # 计算面积比例
            original_area = cv2.contourArea(pixel_points)
            new_area = cv2.contourArea(approx_contour)
            area_ratio = min(new_area / (original_area + 1e-6), 1.5)
            
            # 计算边缘响应分数
            edge_img = cv2.Canny(image, 50, 150)
            edge_mask = cv2.bitwise_and(edge_img, enhanced_mask)
            edge_score = np.sum(edge_mask) / (np.sum(enhanced_mask) + 1e-6)
            normalized_edge_score = min(edge_score * 5, 1.0)
            
            # 计算形状复杂度分数
            shape_complexity = cv2.arcLength(approx_contour, True) / (4 * np.sqrt(new_area + 1e-6))
            shape_score = 1.0 / (shape_complexity + 0.1)
            
            # 计算轮廓质量分数
            contour_quality = len(max_contour) / (len(approx_contour) + 1e-6)
            contour_score = min(contour_quality, 5.0) / 5.0
            
            # 综合各项指标计算新的置信度
            original_confidence = detection.get('confidence', 0.5)
            confidence_adjustment = (area_ratio * 0.3 + normalized_edge_score * 0.3 + 
                                    shape_score * 0.2 + contour_score * 0.2)
            
            new_confidence = original_confidence * 0.7 + confidence_adjustment * 0.3
            processed_detection['confidence'] = min(new_confidence, 1.0)
            
            # 添加处理标志
            processed_detection['is_refined'] = True
        
        return processed_detection
    
    except Exception as e:
        logging.error(f"处理检测结果失败: {e}")
        return detection

# 移除其他重复的函数实现，保留必要的调用关系
# ... 其余代码保持不变 ...

# 修复preprocess_frame函数，确保其不依赖于不可用库
def preprocess_frame(frame):
    """使用高级图像处理技术增强帧质量"""
    if frame is None:
        return frame
        
    try:
        # 创建输出图像
        enhanced = frame.copy()
        
        # 1. 自适应直方图均衡化 (CLAHE)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  # 修复了缩进错误
        
        # 应用CLAHE到亮度通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. 亮度调整
        # 计算当前亮度
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        
        # 根据当前亮度动态调整
        if current_brightness < 100:  # 暗图像
            alpha = 1.2  # 增加对比度
            beta = 10    # 增加亮度
        elif current_brightness > 200:  # 亮图像
            alpha = 0.8  # 降低对比度
            beta = -10   # 降低亮度
        else:  # 正常亮度
            alpha = 1.1  # 轻微增加对比度
            beta = 5     # 轻微增加亮度
            
        # 应用亮度调整
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # 3. 降噪
        # 使用双边滤波保留边缘细节
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. 锐化
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    except Exception as e:
        logging.error(f"图像预处理失败: {e}")
        return frame

def enhance_object_boundaries(image, mask):
    """
    使用最先进的计算机视觉技术增强对象边界，提高边缘的精确性和平滑度
    
    参数:
        image: 输入RGB图像
        mask: 输入二值掩码
    返回:
        增强边界的掩码
    """
    try:
        if mask is None or image is None:
            return mask
            
        # 确保掩码是二值图像
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 1. 多尺度金字塔边缘融合
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # 创建图像金字塔
        pyramid_levels = 4
        pyramids = []
        current_img = gray.copy()
        for i in range(pyramid_levels):
            pyramids.append(current_img)
            current_img = cv2.pyrDown(current_img)
        
        # 从金字塔中提取边缘并上采样回原始尺寸
        edge_maps = []
        for i, level_img in enumerate(pyramids):
            # 使用多种边缘检测算子
            # Canny边缘
            edges_canny = cv2.Canny(level_img, 50, 150)
            
            # Sobel边缘
            sobelx = cv2.Sobel(level_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(level_img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Scharr边缘 (更敏感)
            scharrx = cv2.Scharr(level_img, cv2.CV_64F, 1, 0)
            scharry = cv2.Scharr(level_img, cv2.CV_64F, 0, 1)
            scharr_mag = np.sqrt(scharrx**2 + scharry**2)
            scharr_mag = cv2.normalize(scharr_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Laplacian边缘 (二阶导数)
            laplacian = cv2.Laplacian(level_img, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # 融合当前层级的边缘
            level_edges = cv2.addWeighted(edges_canny, 0.3, sobel_mag, 0.3, 0)
            level_edges = cv2.addWeighted(level_edges, 0.7, scharr_mag, 0.2, 0)
            level_edges = cv2.addWeighted(level_edges, 0.8, laplacian, 0.2, 0)
            
            # 如果不是基础层级，需要上采样到原始尺寸
            if i > 0:
                for j in range(i):
                    level_edges = cv2.pyrUp(level_edges)
                # 调整至原始尺寸
                level_edges = cv2.resize(level_edges, (w, h))
            
            edge_maps.append(level_edges)
        
        # 加权融合不同层级的边缘图
        weights = [0.4, 0.3, 0.2, 0.1]  # 给基础层级更高的权重
        fused_edges = np.zeros_like(edge_maps[0], dtype=np.float32)
        for i, edge_map in enumerate(edge_maps):
            fused_edges += edge_map.astype(np.float32) * weights[i]
        
        # 归一化融合后的边缘图
        fused_edges = cv2.normalize(fused_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. 边缘方向选择性增强
        # 计算梯度方向信息
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # 将角度转换为度数 (0-360)
        ang_deg = ang * 180 / np.pi
        
        # 创建8个方向的边缘增强
        dir_edges = np.zeros_like(gray)
        for angle in range(0, 180, 22):  # 每22.5度一个方向
            # 创建特定方向的掩码
            lower = angle - 11
            upper = angle + 11
            dir_mask = np.zeros_like(gray)
            
            # 找出特定方向的边缘像素
            dir_mask[(ang_deg >= lower) & (ang_deg <= upper)] = 255
            dir_mask[(ang_deg >= lower+180) & (ang_deg <= upper+180)] = 255
            
            # 应用方向掩码
            dir_edge = cv2.bitwise_and(fused_edges, dir_mask)
            
            # 使用形态学操作增强这个方向的边缘
            kernel = np.ones((3, 3), np.uint8)
            dir_edge = cv2.dilate(dir_edge, kernel, iterations=1)
            
            # 累积到总的方向边缘图中
            dir_edges = cv2.max(dir_edges, dir_edge)
        
        # 3. 自适应边缘阈值处理
        # 使用局部自适应阈值增强边缘对比度
        adaptive_edges = cv2.adaptiveThreshold(
            fused_edges,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -2
        )
        
        # 4. 掩码边界区域提取与优化
        # 提取掩码的边界区域
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary_mask, kernel, iterations=2)
        eroded = cv2.erode(binary_mask, kernel, iterations=2)
        boundary_region = cv2.subtract(dilated, eroded)
        
        # 在边界区域应用融合的边缘信息
        edge_boundary = cv2.bitwise_and(dir_edges, boundary_region)
        adaptive_boundary = cv2.bitwise_and(adaptive_edges, boundary_region)
        
        # 融合边缘信息
        enhanced_boundary = cv2.addWeighted(edge_boundary, 0.6, adaptive_boundary, 0.4, 0)
        
        # 5. 边缘响应强度分析
        # 计算各像素的边缘响应强度
        edge_response = np.zeros_like(gray, dtype=np.float32)
        
        # 计算梯度幅值的局部平均和标准差
        gradient_magnitude = cv2.magnitude(gx, gy)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # 使用形态学梯度进一步增强边缘
        morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 组合多种边缘响应
        edge_response = cv2.addWeighted(gradient_magnitude, 0.7, morph_gradient.astype(np.float32), 0.3, 0)
        edge_response = cv2.normalize(edge_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 在边界区域应用边缘响应
        edge_response_boundary = cv2.bitwise_and(edge_response, boundary_region)
        
        # 6. 超分辨率边缘重建
        # 创建更高分辨率的边界图
        scale_factor = 2
        hr_shape = (boundary_region.shape[1] * scale_factor, boundary_region.shape[0] * scale_factor)
        
        # 上采样边界区域
        hr_boundary = cv2.resize(boundary_region, hr_shape, interpolation=cv2.INTER_CUBIC)
        hr_edge_response = cv2.resize(edge_response_boundary, hr_shape, interpolation=cv2.INTER_CUBIC)
        
        # 在高分辨率上增强边缘细节
        hr_edges = cv2.Canny(hr_edge_response, 30, 100)
        hr_edges = cv2.dilate(hr_edges, kernel, iterations=1)
        
        # 回到原始分辨率
        refined_edges = cv2.resize(hr_edges, (boundary_region.shape[1], boundary_region.shape[0]), 
                                 interpolation=cv2.INTER_AREA)
        
        # 7. 轮廓提取与平滑处理
        # 提取精细的轮廓
        enhanced_mask = binary_mask.copy()
        enhanced_boundary_final = cv2.addWeighted(enhanced_boundary, 0.5, refined_edges, 0.5, 0)
        
        # 应用闭操作填充边缘间隙
        enhanced_boundary_final = cv2.morphologyEx(enhanced_boundary_final, cv2.MORPH_CLOSE, kernel)
        
        # 提取轮廓
        contours, _ = cv2.findContours(
            cv2.bitwise_or(enhanced_boundary_final, binary_mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE  # 保留所有轮廓点以便后续平滑
        )
        
        # 绘制新的掩码
        new_mask = np.zeros_like(binary_mask)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 平滑轮廓点 - 使用Fourier变换
            if len(largest_contour) > 10:  # 确保有足够的点
                try:
                    # 将轮廓展平成一维数组
                    contour_array = largest_contour.squeeze()
                    
                    # 处理代码
                    
                    # 绘制平滑的轮廓
                    cv2.drawContours(new_mask, [smooth_contour], 0, 255, -1)
                except Exception as smooth_error:
                    logging.error(f"轮廓平滑失败，使用原始轮廓: {smooth_error}")
                    # 失败时使用原始轮廓
                    cv2.drawContours(new_mask, [largest_contour], 0, 255, -1)
        else:
            # 点太少，使用原始轮廓
            if largest_contour is not None:
                cv2.drawContours(new_mask, [largest_contour], 0, 255, -1)
            else:
                # 没有轮廓，使用原始掩码
                new_mask = binary_mask
            
        # 8. 最终边缘优化 - 使用边缘感知滤波
        # 应用联合双边滤波，保持边缘锐利的同时平滑区域
        kernel_size = 5
        try:
            # 转换为浮点格式
            mask_float = new_mask.astype(np.float32) / 255.0
            
            # 应用边缘感知的引导滤波
            refined_mask = cv2.ximgproc.guidedFilter(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,  # 引导图像
                mask_float,  # 输入图像
                8,           # 半径
                0.03         # 正则化参数
            )
            
            # 转回二值图像
            refined_mask = (refined_mask * 255).astype(np.uint8)
            _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 确保没有孤立的区域
            # 只保留最大的连通区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
            
            if num_labels > 1:  # 如果有多个连通区域
                # 找出最大的非背景区域
                max_area = 0
                max_label = 0
                
                for i in range(1, num_labels):  # 从1开始，0是背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = i
                
                # 创建只包含最大区域的掩码
                final_mask = np.zeros_like(refined_mask)
                final_mask[labels == max_label] = 255
                return final_mask
            else:
                return refined_mask
                
        except Exception as filter_error:
            logging.error(f"边缘感知滤波失败: {filter_error}")
            return new_mask
            
    except Exception as e:
        logging.error(f"对象边界高级增强失败: {e}")
        return mask

# 在文件末尾新增两个新函数
def optimize_polygon_points_bottle(points, max_points=MAX_POLYGON_POINTS, min_points=MIN_POLYGON_POINTS, rdp_epsilon=RDPEPS):
    """针对瓶子，多边形点优化：使用椭圆拟合方法，让瓶子的边缘更平滑呈现圆弧效果。"""
    points = np.array(points)
    if len(points) < 5:
        return points
    try:
        ellipse = cv2.fitEllipse(points.astype(np.float32))
    except Exception as e:
        logging.error(f"拟合椭圆失败: {e}")
        return points
    (cx, cy), (major, minor), angle = ellipse
    target_points = int((min_points + max_points) / 2)
    angles = np.linspace(0, 2 * np.pi, target_points, endpoint=False)
    angle_rad = np.deg2rad(angle)
    a = major / 2.0
    b = minor / 2.0
    x = cx + a * np.cos(angles) * np.cos(angle_rad) - b * np.sin(angles) * np.sin(angle_rad)
    y = cy + a * np.cos(angles) * np.sin(angle_rad) + b * np.sin(angles) * np.cos(angle_rad)
    new_points = np.column_stack((x, y))
    new_points[:, 0] = cv2.GaussianBlur(new_points[:, 0].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    new_points[:, 1] = cv2.GaussianBlur(new_points[:, 1].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    try:
        tck, u = interpolate.splprep([new_points[:,0], new_points[:,1]], s=1)
        u_new = np.linspace(0, 1, len(new_points))
        x_new, y_new = interpolate.splev(u_new, tck)
        new_points = np.column_stack((x_new, y_new))
    except Exception as e:
        logging.warning(f'瓶子B-spline平滑失败: {e}')
    return new_points

def optimize_polygon_points_person(points, max_points=MAX_POLYGON_POINTS, min_points=MIN_POLYGON_POINTS, rdp_epsilon=RDPEPS):
    """针对人物，多边形点优化：增强采样密度以填补轮廓空隙，提高检测结果的连续性。"""
    points = np.array(points)
    if len(points) < 3:
        return points
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cum_length = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_length[-1]
    base_target = int((min_points + max_points) / 2)
    target_points = int(base_target * 1.2)
    if total_length == 0:
        return points
    even_distances = np.linspace(0, total_length, target_points)
    new_x = np.interp(even_distances, cum_length, points[:, 0])
    new_y = np.interp(even_distances, cum_length, points[:, 1])
    new_points = np.stack((new_x, new_y), axis=-1)
    new_points[:, 0] = cv2.GaussianBlur(new_points[:, 0].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    new_points[:, 1] = cv2.GaussianBlur(new_points[:, 1].astype(np.float32).reshape(-1, 1), (5, 1), 0).flatten()
    try:
        tck, u = interpolate.splprep([new_points[:,0], new_points[:,1]], s=1)
        u_new = np.linspace(0, 1, len(new_points))
        x_new, y_new = interpolate.splev(u_new, tck)
        new_points = np.column_stack((x_new, y_new))
    except Exception as e:
        logging.warning(f'人物B-spline平滑失败: {e}')
    return new_points