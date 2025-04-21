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
from xml.dom import minidom # 用于美化打印XML
from pathlib import Path # 使用 Pathlib 处理路径
import random # 用于生成随机颜色 (可选)
import datetime
# import html # 如果需要解析SVG (在此版本中未使用)

# --- 配置区 ---
DEFAULT_NUCLIO_SEG_URL = "http://192.168.10.158:32792" # <<--- **修改这里** 指向你的分割函数 URL
DEFAULT_VIDEO_PATH = r"X:\视频样例\一般类\其他类\手工类\71341_segment_2.mp4"# <<--- **修改这"X:\视频样例\一般类\其他类\手工类\71341_segment_2.mp4"里** 默认输入视频路径
DEFAULT_OUTPUT_BASE_DIR = r"E:\视频样例其他类\手工类1" # 修改输出目录名
DEFAULT_FRAME_SKIP = 1
DEFAULT_SAVE_VIDEO = True # 默认保存带标注的视频
REQUEST_TIMEOUT = 120 # 分割可能更耗时，增加超时
CONF_THRESHOLD = 0.35 # 用于过滤检测结果和绘图标注的置信度阈值
VIDEO_CODEC = 'mp4v' # 输出视频的编码器 (如 'mp4v' for .mp4, 'XVID' for .avi)
VIDEO_EXTENSION = '.mp4' # 输出视频的文件扩展名
DRAW_BOX = True # 是否绘制边界框
DRAW_MASK = True # 是否绘制分割掩码
DRAW_LABEL = True # 是否绘制类别和置信度标签
MASK_ALPHA = 0.4 # 分割掩码的透明度

# *** 重要: 定义你的模型可能输出的所有类别标签 ***
# *** Update this set with ALL possible class names your model can output ***
ALL_POSSIBLE_LABELS = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"} # <-- !!! 修改这里 !!!

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
    # 原有19种颜色（保持不变）
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (255, 128, 0), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 255), (128, 0, 255),

    # 新增61种颜色（不含纯黑/纯白）
    (255, 165, 0), (75, 0, 130), (138, 43, 226), (147, 112, 219),
    (72, 61, 139), (123, 104, 238), (106, 90, 205), (176, 196, 222),
    (230, 230, 250), (255, 160, 122), (255, 192, 203), (255, 20, 147),
    (219, 112, 147), (199, 21, 133), (255, 105, 180), (255, 228, 225),  # 修改了接近纯白的颜色
    (139, 69, 19), (160, 82, 45), (210, 180, 140), (244, 164, 96),
    (188, 143, 143), (255, 215, 0), (218, 165, 32), (184, 134, 11),
    (205, 133, 63), (210, 105, 30), (139, 137, 112), (85, 107, 47),
    (107, 142, 35), (154, 205, 50), (50, 205, 50), (144, 238, 144),
    (152, 251, 152), (143, 188, 143), (0, 100, 0), (0, 250, 154),
    (46, 139, 87), (102, 205, 170), (32, 178, 170), (64, 224, 208),
    (72, 209, 204), (175, 238, 238), (127, 255, 212), (0, 255, 127),
    (0, 139, 139), (95, 158, 160), (70, 130, 180), (100, 149, 237),
    (30, 144, 255), (135, 206, 235), (135, 206, 250), (25, 25, 120),  # 加深接近纯黑的颜色
    (65, 105, 225), (0, 191, 255), (173, 216, 230), (240, 248, 245),  # 调整接近纯白的颜色
    (248, 248, 240), (245, 245, 240), (220, 220, 220), (211, 211, 211),
    (169, 169, 169), (128, 128, 128), (105, 105, 105), (20, 20, 20)  # 用深灰替代纯黑
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
                    for instance in results:
                        if (isinstance(instance, dict) and
                            'box_normalized' in instance and isinstance(instance['box_normalized'], list) and len(instance['box_normalized']) == 4 and
                            'confidence' in instance and isinstance(instance['confidence'], (float, int)) and
                            'class_name' in instance and isinstance(instance['class_name'], str) and
                            'mask_polygon_normalized' in instance and isinstance(instance['mask_polygon_normalized'], list)):
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
                cv2.fillPoly(overlay, [mask_points_px], color)
            except Exception as e:
                logging.warning(f"绘制掩码时出错 (实例 {i}): {e}. Mask data: {str(mask_poly_norm)[:100]}")

        # --- 绘制边界框 ---
        if DRAW_BOX and box_norm:
            try:
                x1, y1, x2, y2 = box_norm
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(annotated_frame, pt1, pt2, color, 2)
            except Exception as e:
                logging.warning(f"绘制边界框时出错 (实例 {i}): {e}. Box data: {box_norm}")

        # --- 绘制标签 ---
        if DRAW_LABEL and box_norm:
             try:
                x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_origin = (x1, max(0, y1 - text_height - baseline // 2))
                cv2.rectangle(annotated_frame, label_origin, (label_origin[0] + text_width, label_origin[1] + text_height + baseline), color, -1)
                cv2.putText(annotated_frame, label_text, (label_origin[0], label_origin[1] + text_height + baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
             except Exception as e:
                 logging.warning(f"绘制标签时出错 (实例 {i}): {e}. Label: {label_text}")

        instance_count += 1

    # --- 合并掩码图层 ---
    if instance_count > 0:
        cv2.addWeighted(overlay, MASK_ALPHA, annotated_frame, 1 - MASK_ALPHA, 0, annotated_frame)

    return annotated_frame


# --- XML 生成函数 (修改后) ---
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
    current_time_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "+00:00"
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


# --- 主执行逻辑 (保持不变) ---
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

    if not os.path.exists(video_path):
        logging.error(f"视频未找到: {video_path}"); return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"无法打开视频: {video_path}"); return

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
            b64_string = encode_frame_to_base64(frame)
            if not b64_string:
                logging.warning(f"因编码错误跳过帧 {frame_count} 的处理。")
                detections = None
            else:
                detections = call_nuclio_segmentor(b64_string, nuclio_url, frame_count)

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
    parser.set_defaults(save_video=DEFAULT_SAVE_VIDEO) # 设置 argparse 的默认值

    args = parser.parse_args()

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

    main(args.video, args.url, args.output_dir, frame_skip_value, CONF_THRESHOLD, save_video_flag)