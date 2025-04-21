# -*- coding: utf-8 -*-
"""
Time:     2025/4/10
Author:   ZhaoQi Cao(czq) - Modified for Manual-Like XML Output
Version:  V 1.9 (中文) (视频姿态估计 + 物体检测 - 输出类似手动标注的XML) # Version Updated
File:     model_http_object_detection.py # Renamed conceptually
Describe: 用于测试部署在Nuclio上的YOLOv8姿态估计和物体检测函数 (处理视频输入, 保存单个类似手动标注格式的CVAT XML)
"""
import cv2
import base64
import requests
import json
import os
import argparse
import logging
import time
from datetime import timedelta, datetime
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom # 用于美化打印XML
from pathlib import Path # 使用 Pathlib 处理路径
import html # 用于转义 SVG 中的特殊字符

# --- 配置区 ---
DEFAULT_NUCLIO_POSE_URL = "http://192.168.10.158:32772" # <<--- 确认姿态估计函数 URL
# !!! 添加物体检测服务 URL !!!
DEFAULT_NUCLIO_DETECTION_URL = "http://192.168.10.158:32772" # <<--- 修改为你部署的 *物体检测* 函数的 URL
DEFAULT_VIDEO_PATH = r"video_list\25. SHAHBAZIAN Suzanna (CAN) - 2022 Rhythmic Worlds, Sofia (BUL) - Qualifications Hoop_no_head_tail_no_head_tail_003.mp4" # 默认输入视频路径
DEFAULT_OUTPUT_BASE_DIR = r"C:\Users\DELL\Desktop\cam-prcess-data-1\out3" # 修改输出目录名
DEFAULT_FRAME_SKIP = 1
REQUEST_TIMEOUT = 60

# 姿态估计阈值
KEYPOINT_CONF_THRESHOLD = 0.5 # 用于XML中occluded标志

# !!! 添加物体检测配置 !!!
BOX_CONF_THRESHOLD = 0.5    # 用于过滤检测实例的边界框置信度阈值
# 边界框扩展系数 - 用于扩大检测到的边界框 (可选, 但推荐)
BOX_EXPANSION_RATIO = 0.05  # 边界框向各方向扩展的比例 (5%)
BOX_TOP_EXTRA_EXPANSION = 0.05  # 顶部额外扩展比例，用于覆盖头发/丸子头 (额外5%)
# 标签映射 (如果需要)
LABEL_MAPPING = {
    # "person": "person" # 如果检测器输出person, CVAT也用person, 则无需映射
}

# 创建日志和输出目录
os.makedirs('./logs/', exist_ok=True)
output_base_dir = Path(DEFAULT_OUTPUT_BASE_DIR)
output_xmls_dir = output_base_dir / "annotations"
output_xmls_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE = "./logs/video_combined_testing_manual_like.log" # 日志文件名 Updated

# --- 设置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- COCO 关键点名称列表 (顺序必须与模型输出一致) ---
COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# --- 辅助函数 (encode_frame_to_base64, call_nuclio_pose_estimator) ---
# (保持不变)
def encode_frame_to_base64(frame):
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success: logging.error("无法将帧编码为JPEG。"); return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e: logging.error(f"帧编码过程中出错: {e}"); return None

def call_nuclio_pose_estimator(base64_image_string, nuclio_url, frame_number):
    payload = json.dumps({"image": base64_image_string})
    headers = {'Content-Type': 'application/json'}
    logging.debug(f"帧 {frame_number}: 正在向 {nuclio_url} 发送请求...")
    try:
        start_time = time.time()
        response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        end_time = time.time()
        logging.debug(f"帧 {frame_number}: Nuclio 请求耗时 {end_time - start_time:.2f} 秒。")
        if response.status_code == 200:
            try:
                results = response.json()
                if isinstance(results, list):
                    logging.info(f"帧 {frame_number}: 收到 {len(results)} 个检测实例。")
                    valid_results = []
                    for instance in results:
                        # 确保关键数据存在
                        if isinstance(instance, dict) and \
                           'class_name' in instance and \
                           'keypoints' in instance and \
                           isinstance(instance['keypoints'], list):
                            # 检查关键点数量是否符合预期（例如 COCO 17 个点）
                            if len(instance['keypoints']) == 17:
                                valid_results.append(instance)
                            else:
                                logging.warning(f"帧 {frame_number}: 实例关键点数量 ({len(instance['keypoints'])}) 不符合预期 (17)，跳过。实例: {instance}")
                        else:
                            logging.warning(f"帧 {frame_number}: 跳过缺少关键点或类别名的实例: {instance}")
                    return valid_results
                else:
                    logging.error(f"帧 {frame_number}: Nuclio 响应不是列表格式。收到: {type(results)}")
                    return None
            except json.JSONDecodeError:
                logging.error(f"帧 {frame_number}: 解析 Nuclio JSON 失败。状态码: {response.status_code}, 响应体: {response.text[:200]}...")
                return None
        else:
            logging.error(f"帧 {frame_number}: Nuclio 函数错误。状态码: {response.status_code}, 响应体: {response.text[:200]}...")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 超时 ({REQUEST_TIMEOUT} 秒)。")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"帧 {frame_number}: 请求 Nuclio 失败: {e}")
        return None
    except Exception as e:
        logging.error(f"帧 {frame_number}: 调用 Nuclio 时意外错误: {e}")
        return None

def call_nuclio_object_detector(base64_image_string, nuclio_url, frame_number):
    """将Base64编码的图像发送给Nuclio物体检测函数并返回结果。"""
    payload = json.dumps({
        "image": base64_image_string,
        "threshold": BOX_CONF_THRESHOLD,
        "return_objects": True
    })
    headers = {'Content-Type': 'application/json'}
    logging.debug(f"帧 {frame_number}: 正在向 {nuclio_url} 发送物体检测请求...")
    
    try:
        start_time = time.time()
        response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        end_time = time.time()
        logging.debug(f"帧 {frame_number}: Nuclio 物体检测请求耗时 {end_time - start_time:.2f} 秒。")

        if response.status_code == 200:
            try:
                results = response.json()
                logging.info(f"帧 {frame_number}: 收到物体检测响应: {str(results)[:200]}...")
                
                # 处理不同格式的响应
                detection_instances = []
                if isinstance(results, list):
                    detection_instances = results
                elif isinstance(results, dict):
                    if 'predictions' in results: detection_instances = results['predictions']
                    elif 'objects' in results: detection_instances = results['objects']
                    elif 'detections' in results: detection_instances = results['detections']
                    else:
                        for value in results.values():
                            if isinstance(value, list) and value: detection_instances = value; break
                
                logging.info(f"帧 {frame_number}: 解析得到 {len(detection_instances)} 个物体检测实例。")
                
                # 标准化结果格式
                valid_results = []
                for instance in detection_instances:
                    if isinstance(instance, dict):
                        box = None; confidence = None; class_name = None
                        
                        # Find box
                        if 'box' in instance and len(instance['box']) == 4: box = instance['box']
                        elif 'bbox' in instance and len(instance['bbox']) == 4: box = instance['bbox']
                        # Find confidence
                        if 'confidence' in instance: confidence = float(instance['confidence'])
                        elif 'score' in instance: confidence = float(instance['score'])
                        # Find class name
                        if 'class_name' in instance: class_name = str(instance['class_name'])
                        elif 'label' in instance: class_name = str(instance['label'])
                        elif 'class' in instance: class_name = str(instance['class'])
                        
                        if box and class_name and confidence and confidence >= BOX_CONF_THRESHOLD:
                            # Apply label mapping
                            class_name = LABEL_MAPPING.get(class_name, class_name)
                            
                            valid_results.append({
                                'box': box,
                                'confidence': confidence,
                                'class_name': class_name
                            })
                            logging.debug(f"帧 {frame_number}: 添加了 {class_name} 物体检测结果 (置信度: {confidence:.2f})")
                        else:
                             logging.debug(f"帧 {frame_number}: 跳过无效的物体检测实例: {instance}")

                return valid_results
            except Exception as e:
                logging.error(f"帧 {frame_number}: 处理物体检测结果失败: {e}")
                return [] # Return empty list on error
        else:
            logging.error(f"帧 {frame_number}: 物体检测 HTTP 错误 {response.status_code}: {response.text[:200]}...")
            return [] # Return empty list on error
    except Exception as e:
        logging.error(f"帧 {frame_number}: 物体检测请求出错: {e}")
        return [] # Return empty list on error

# --- XML 生成辅助函数 ---
def pretty_print_xml(elem):
    """返回包含声明且格式化（美化）的XML字符串。"""
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        xml_str = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8')
        xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
        return xml_str
    except Exception as e:
        logging.error(f"XML美化打印过程中出错: {e}")
        return None

# --- 修改: 保存为类似手动标注的 CVAT Images XML ---
def save_results_to_cvat_manual_like_xml(all_results, xml_output_path, video_filename, frame_width, frame_height, total_processed_frames, original_total_frames):
    """将所有处理帧的检测结果保存为类似手动标注格式的单个 CVAT XML 1.1 for Images 文件。"""
    logging.info(f"正在为 {total_processed_frames} 个处理过的帧构建类似手动标注的 CVAT 图像 XML...")

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- Meta 信息 ---
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "N/A" # CVAT 会忽略导入的 ID
    task_name, _ = os.path.splitext(video_filename)
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(total_processed_frames)
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0" # 或者根据实际设置
    ET.SubElement(task, "bugtracker").text = ""
    current_time_utc = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "+00:00"
    ET.SubElement(task, "created").text = current_time_utc
    ET.SubElement(task, "updated").text = current_time_utc
    ET.SubElement(task, "start_frame").text = "0"
    last_processed_frame_num = max(all_results.keys()) if all_results else 0
    ET.SubElement(task, "stop_frame").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(task, "frame_filter").text = ""

    # --- Labels 定义 (关键修改) ---
    labels_elem = ET.SubElement(task, "labels")

    # 提取所有标签名 (pose 'person' + object detection labels)
    label_names = set(['person']) # Start with person for pose
    if 'object_detections' in all_results:
        for frame_num, detections in all_results['object_detections'].items():
            for det in detections:
                if 'class_name' in det:
                    label_names.add(det['class_name'])

    # 1. 定义主 Skeleton 标签 ('person')
    if 'person' in label_names:
        main_label = ET.SubElement(labels_elem, "label")
        ET.SubElement(main_label, "name").text = "person"
        ET.SubElement(main_label, "color").text = "#c06060" # Or leave empty
        ET.SubElement(main_label, "type").text = "skeleton"
        # Add SVG definition (optional but helpful for UI)
        # 需要确保 SVG 字符串中的 " 被正确处理或不出现
        svg_string = '<svg><line x1="44.89966583251953" y1="73.41136932373047" x2="44.89966583251953" y2="44.9832763671875" data-type="edge" data-node-from="14" data-node-to="12"></line><line x1="46.57190704345703" y1="93.97993469238281" x2="44.89966583251953" y2="73.41136932373047" data-type="edge" data-node-from="16" data-node-to="14"></line><line x1="55.10033416748047" y1="73.07691955566406" x2="53.762542724609375" y2="94.9832763671875" data-type="edge" data-node-from="15" data-node-to="17"></line><line x1="55.434783935546875" y1="44.64883041381836" x2="55.10033416748047" y2="73.07691955566406" data-type="edge" data-node-from="13" data-node-to="15"></line><line x1="64.13043212890625" y1="38.79598617553711" x2="63.96321105957031" y2="51.00334548950195" data-type="edge" data-node-from="9" data-node-to="11"></line><line x1="55.434783935546875" y1="44.64883041381836" x2="50.25083541870117" y2="14.381271362304688" data-type="edge" data-node-from="13" data-node-to="1"></line><line x1="55.434783935546875" y1="13.043478012084961" x2="64.13043212890625" y2="38.79598617553711" data-type="edge" data-node-from="5" data-node-to="9"></line><line x1="52.25752639770508" y1="12.207357406616211" x2="55.434783935546875" y2="13.043478012084961" data-type="edge" data-node-from="4" data-node-to="5"></line><line x1="50.25083541870117" y1="14.381271362304688" x2="52.25752639770508" y2="12.207357406616211" data-type="edge" data-node-from="1" data-node-to="4"></line><line x1="50.25083541870117" y1="14.381271362304688" x2="44.89966583251953" y2="44.9832763671875" data-type="edge" data-node-from="1" data-node-to="12"></line><line x1="38.21070098876953" y1="37.62541961669922" x2="34.698997497558594" y2="50.501670837402344" data-type="edge" data-node-from="8" data-node-to="10"></line><line x1="40.886287689208984" y1="22.909698486328125" x2="38.21070098876953" y2="37.62541961669922" data-type="edge" data-node-from="6" data-node-to="8"></line><line x1="45.066890716552734" y1="13.879598617553711" x2="40.886287689208984" y2="22.909698486328125" data-type="edge" data-node-from="3" data-node-to="6"></line><line x1="48.57859420776367" y1="12.709030151367188" x2="45.066890716552734" y2="13.879598617553711" data-type="edge" data-node-from="2" data-node-to="3"></line><line x1="50.25083541870117" y1="14.381271362304688" x2="48.57859420776367" y2="12.709030151367188" data-type="edge" data-node-from="1" data-node-to="2"></line><circle r="0.75" cx="50.25083541870117" cy="14.381271362304688" data-type="element node" data-element-id="1" data-node-id="1" data-label-name="1"></circle><circle r="0.75" cx="48.57859420776367" cy="12.709030151367188" data-type="element node" data-element-id="2" data-node-id="2" data-label-name="2"></circle><circle r="0.75" cx="45.066890716552734" cy="13.879598617553711" data-type="element node" data-element-id="3" data-node-id="3" data-label-name="3"></circle><circle r="0.75" cx="52.25752639770508" cy="12.207357406616211" data-type="element node" data-element-id="4" data-node-id="4" data-label-name="4"></circle><circle r="0.75" cx="55.434783935546875" cy="13.043478012084961" data-type="element node" data-element-id="5" data-node-id="5" data-label-name="5"></circle><circle r="0.75" cx="40.886287689208984" cy="22.909698486328125" data-type="element node" data-element-id="6" data-node-id="6" data-label-name="6"></circle><circle r="0.75" cx="59.44816207885742" cy="23.2441463470459" data-type="element node" data-element-id="7" data-node-id="7" data-label-name="7"></circle><circle r="0.75" cx="38.21070098876953" cy="37.62541961669922" data-type="element node" data-element-id="8" data-node-id="8" data-label-name="8"></circle><circle r="0.75" cx="64.13043212890625" cy="38.79598617553711" data-type="element node" data-element-id="9" data-node-id="9" data-label-name="9"></circle><circle r="0.75" cx="34.698997497558594" cy="50.501670837402344" data-type="element node" data-element-id="10" data-node-id="10" data-label-name="10"></circle><circle r="0.75" cx="63.96321105957031" cy="51.00334548950195" data-type="element node" data-element-id="11" data-node-id="11" data-label-name="11"></circle><circle r="0.75" cx="44.89966583251953" cy="44.9832763671875" data-type="element node" data-element-id="12" data-node-id="12" data-label-name="12"></circle><circle r="0.75" cx="55.434783935546875" cy="44.64883041381836" data-type="element node" data-element-id="13" data-node-id="13" data-label-name="13"></circle><circle r="0.75" cx="44.89966583251953" cy="73.41136932373047" data-type="element node" data-element-id="14" data-node-id="14" data-label-name="14"></circle><circle r="0.75" cx="55.10033416748047" cy="73.07691955566406" data-type="element node" data-element-id="15" data-node-id="15" data-label-name="15"></circle><circle r="0.75" cx="46.57190704345703" cy="93.97993469238281" data-type="element node" data-element-id="16" data-node-id="16" data-label-name="16"></circle><circle r="0.75" cx="53.762542724609375" cy="94.9832763671875" data-type="element node" data-element-id="17" data-node-id="17" data-label-name="17"></circle></svg>'
        # 注意：SVG 字符串中的 " 需要被正确处理，或者在生成 SVG 时避免使用引号
        # 如果直接从 XML 复制，可能需要 html.unescape() 处理
        # svg_elem = ET.SubElement(main_label, "svg")
        # svg_elem.text = svg_string # 直接赋值字符串
        # 或者，如果需要解析实体:
        try:
            # 尝试解析包含实体的 SVG 字符串，但这可能需要外部库或更复杂的处理
            # ET.fromstring() 对实体支持有限
            # 一个变通方法是先 unescape
            svg_elem = ET.fromstring(html.unescape(svg_string))
            main_label.append(svg_elem)
        except ET.ParseError:
            logging.warning("无法解析提供的 SVG 字符串以添加到 XML meta 中。骨架连接可能无法在 UI 中正确显示。")
            # 可以选择不添加错误的 SVG，或者只添加注释掉的字符串
            # ET.SubElement(main_label, "svg").text = "<!-- SVG data omitted due to parsing error -->"
        ET.SubElement(main_label, "attributes")

        # Define pose sub-labels ("1" to "17")
        for i in range(1, 18):
            sub_label = ET.SubElement(labels_elem, "label")
            ET.SubElement(sub_label, "name").text = str(i)
            ET.SubElement(sub_label, "color").text = ""
            ET.SubElement(sub_label, "type").text = "points"
            ET.SubElement(sub_label, "attributes")
            ET.SubElement(sub_label, "parent").text = "person"

    # 2. 定义其他 Object Detection 标签 (type rectangle)
    for name in sorted(list(label_names)):
        if name != 'person': # Avoid redefining 'person'
             obj_label = ET.SubElement(labels_elem, "label")
             ET.SubElement(obj_label, "name").text = name
             ET.SubElement(obj_label, "color").text = ""
             ET.SubElement(obj_label, "type").text = "rectangle"
             # Add attributes for object detection if needed (e.g., confidence)
             attributes = ET.SubElement(obj_label, "attributes")
             # Example attribute:
             # attr = ET.SubElement(attributes, "attribute")
             # ET.SubElement(attr, "name").text = "confidence"
             # ET.SubElement(attr, "input_type").text = "number"
             # ...

    # --- 结束 Labels 定义 ---

    # segments, owner, assignee, subset, original_size (保持不变)
    segments = ET.SubElement(task, "segments"); segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"; ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(segment, "url").text = "N/A" # CVAT 会用自己的
    ET.SubElement(task, "owner"); ET.SubElement(task, "assignee"); ET.SubElement(task, "subset").text = "Default"
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(frame_width); ET.SubElement(original_size, "height").text = str(frame_height)
    ET.SubElement(meta, "dumped").text = current_time_utc
    # --- 结束 Meta 信息 ---

    # --- 添加 <image> 元素 ---
    # Combine frame numbers from both detection types
    all_frame_numbers = set()
    if 'object_detections' in all_results: all_frame_numbers.update(all_results['object_detections'].keys())
    if 'pose_detections' in all_results: all_frame_numbers.update(all_results['pose_detections'].keys())

    if all_frame_numbers:
        for frame_number in sorted(list(all_frame_numbers)):
            # ... create <image> element ...
            image_elem = ET.SubElement(root, "image")
            frame_id_0_based = frame_number - 1
            image_elem.set("id", str(frame_id_0_based))
            image_elem.set("name", f"{frame_id_0_based:06d}") # Use 0-padded name
            image_elem.set("width", str(frame_width))
            image_elem.set("height", str(frame_height))

            # --- Add Object Detections as <box> ---
            if 'object_detections' in all_results and frame_number in all_results['object_detections']:
                object_detections = all_results['object_detections'][frame_number]
                for det in object_detections:
                     if isinstance(det, dict) and 'box' in det and 'class_name' in det:
                        try:
                            x_min, y_min, x_max, y_max = det['box']
                            class_name = det['class_name']

                            # Apply expansion (optional but recommended)
                            width = x_max - x_min
                            height = y_max - y_min
                            expansion_w = width * BOX_EXPANSION_RATIO
                            expansion_h = height * BOX_EXPANSION_RATIO
                            top_extra = height * BOX_TOP_EXTRA_EXPANSION

                            x_min = max(0, x_min - expansion_w)
                            y_min = max(0, y_min - expansion_h - top_extra)
                            x_max = min(frame_width, x_max + expansion_w)
                            y_max = min(frame_height, y_max + expansion_h)

                            if x_max > x_min and y_max > y_min:
                                box_elem = ET.SubElement(image_elem, "box")
                                box_elem.set("label", class_name)
                                box_elem.set("source", "auto") # Mark as automatic
                                box_elem.set("occluded", "0")
                                box_elem.set("xtl", f"{x_min:.2f}")
                                box_elem.set("ytl", f"{y_min:.2f}")
                                box_elem.set("xbr", f"{x_max:.2f}")
                                box_elem.set("ybr", f"{y_max:.2f}")
                                # Add confidence attribute if defined in labels
                                # if 'confidence' in det:
                                #     attr = ET.SubElement(box_elem, "attribute", name="confidence")
                                #     attr.text = f"{det['confidence']:.4f}"
                        except Exception as e:
                            logging.warning(f"帧 {frame_number}: 处理物体边界框时出错: {e}")

            # --- Add Pose Detections as <skeleton> ---
            if 'pose_detections' in all_results and frame_number in all_results['pose_detections']:
                pose_detections = all_results['pose_detections'][frame_number]
                for det_idx, det in enumerate(pose_detections):
                    label_name = str(det.get('class_name', 'unknown'))
                    keypoints = det.get('keypoints') # [[x, y, conf], ...]

                    if label_name != 'person' or not keypoints or len(keypoints) != 17:
                        continue # Skip if not a valid person pose

                    # Create <skeleton> tag
                    skeleton_elem = ET.SubElement(image_elem, "skeleton")
                    skeleton_elem.set("label", label_name)
                    skeleton_elem.set("source", "auto") # Mark as automatic
                    skeleton_elem.set("occluded", "0")
                    # ... other skeleton attributes ...

                    valid_point_added = False
                    for kpt_idx, kpt in enumerate(keypoints):
                         if isinstance(kpt, list) and len(kpt) == 3:
                            x, y, conf = kpt
                            point_label = str(kpt_idx + 1) # "1" to "17"
                            occluded = "1" if conf < KEYPOINT_CONF_THRESHOLD else "0"

                            point_elem = ET.SubElement(skeleton_elem, "points")
                            point_elem.set("label", point_label)
                            point_elem.set("source", "auto")
                            point_elem.set("occluded", occluded)
                            point_elem.set("outside", "0")
                            point_elem.set("points", f"{x:.2f},{y:.2f}")
                            valid_point_added = True

                    if not valid_point_added:
                        image_elem.remove(skeleton_elem) # Remove empty skeletons

    # --- 写入 XML 文件 ---
    # (写入逻辑保持不变)
    try:
        full_xml_string = pretty_print_xml(root)
        if full_xml_string:
            if not full_xml_string.strip().startswith("<?xml"):
                 logging.error("美化打印未能生成有效的XML开头。"); raise ValueError("...")
            with open(xml_output_path, "w", encoding="utf-8") as f:
                f.write(full_xml_string)
            logging.info(f"类似手动标注的 CVAT 图像 XML 已保存: {xml_output_path}")
        else:
            logging.warning("美化打印失败，回退到基础XML写入器。")
            tree = ET.ElementTree(root)
            if hasattr(ET, 'indent'): ET.indent(tree, space="  ", level=0)
            tree.write(str(xml_output_path), encoding='utf-8', xml_declaration=True)
            logging.info(f"类似手动标注的 CVAT 图像 XML 已保存 (后备方案): {xml_output_path}")
    except Exception as e:
        logging.error(f"保存类似手动标注的 CVAT XML 文件失败: {e}")


# --- 主执行逻辑 ---
def main(video_path, nuclio_pose_url, nuclio_detection_url, output_dir, frame_skip):
    """主函数，执行视频处理、保存类似手动标注格式 XML 的流程。"""
    output_dir = Path(output_dir)
    # output_images_dir = output_dir / "images" # 如果需要保存图像，取消注释
    output_xmls_dir = output_dir / "annotations"
    # output_images_dir.mkdir(parents=True, exist_ok=True)
    output_xmls_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始处理视频: {video_path}")
    logging.info(f"Nuclio 姿态估计 URL: {nuclio_pose_url}")
    logging.info(f"Nuclio 物体检测 URL: {nuclio_detection_url}")
    logging.info(f"处理帧间隔: 每 {frame_skip} 帧")
    logging.info(f"输出目录 (XML): {output_dir}")

    # 1. 检查视频
    if not os.path.exists(video_path): logging.error(f"视频文件未找到: {video_path}"); return

    # 2. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): logging.error(f"无法打开视频文件: {video_path}"); return

    # 3. 获取属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_total_frames <= 0:
         logging.warning("无法获取视频总帧数。")
         if frame_width == 0 or frame_height == 0:
             ret, frame = cap.read();
             if ret: frame_height, frame_width, _ = frame.shape; cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             else: logging.error("无法读取视频帧尺寸。"); cap.release(); return
    logging.info(f"视频信息 - 尺寸: {frame_width}x{frame_height}, FPS: {fps:.2f}, 总帧数: {original_total_frames if original_total_frames > 0 else '未知'}")

    # 4. 初始化
    frame_count = 0
    processed_frame_count = 0
    all_results = {
        'pose_detections': {},   # {frame_num: [pose_instances]}
        'object_detections': {}  # {frame_num: [object_instances]}
    }
    start_process_time = time.time()

    # 5. 逐帧处理
    while True:
        ret, frame = cap.read()
        if not ret: logging.info("视频处理到达结尾或读取错误。"); break
        frame_count += 1

        if frame_skip > 1 and frame_count % frame_skip != 0: continue

        processed_frame_count += 1
        logging.info(f"正在处理帧 {frame_count}{f'/{original_total_frames}' if original_total_frames > 0 else ''}...")

        # --- 如果需要保存图像，取消下面的注释 ---
        # frame_id_0_based = frame_count - 1
        # frame_filename = f"{frame_id_0_based:06d}.jpg"
        # output_image_path = output_images_dir / frame_filename
        # try:
        #     cv2.imwrite(str(output_image_path), frame)
        #     logging.debug(f"原始帧图像已保存: {output_image_path}")
        # except Exception as e:
        #     logging.error(f"保存帧 {frame_count} 图像失败: {e}")
        #     continue
        # ------------------------------------

        b64_string = encode_frame_to_base64(frame)
        if not b64_string: logging.warning(f"因编码错误跳过帧 {frame_count}。"); continue

        # Call Pose Estimator
        pose_detections = call_nuclio_pose_estimator(b64_string, nuclio_pose_url, frame_count)
        all_results['pose_detections'][frame_count] = pose_detections if pose_detections is not None else []

        # Call Object Detector
        object_detections = call_nuclio_object_detector(b64_string, nuclio_detection_url, frame_count)
        all_results['object_detections'][frame_count] = object_detections # Already returns [] on error

        if not all_results['pose_detections'][frame_count] and not all_results['object_detections'][frame_count]:
             logging.warning(f"帧 {frame_count}: 未收到任何有效结果。")

        # --- 进度估计 (可选) ---
        if original_total_frames > 0 and processed_frame_count > 0 and processed_frame_count % 10 == 0:
            try: pass
            except Exception as e_est: logging.warning(f"无法估计剩余时间: {e_est}")

    # --- 清理与结束 ---
    cap.release()
    cv2.destroyAllWindows()

    # --- 保存类似手动标注的 CVAT XML 文件 ---
    video_basename = os.path.basename(video_path)
    video_name_no_ext, _ = os.path.splitext(video_basename)
    xml_output_filename = f"{video_name_no_ext}_cvat_manual_like_combined.xml" # New filename
    xml_output_path = output_xmls_dir / xml_output_filename

    save_results_to_cvat_manual_like_xml( # Pass the combined results
        all_results,
        xml_output_path,
        video_basename,
        frame_width,
        frame_height,
        processed_frame_count,
        original_total_frames
    )
    # -------------------------------------

    end_process_time = time.time()
    total_time = max(0, end_process_time - start_process_time)
    logging.info("-" * 30)
    logging.info(f"视频处理完成。")
    logging.info(f"总耗时: {timedelta(seconds=int(total_time))}")
    logging.info(f"总共读取帧数: {frame_count}。")
    logging.info(f"实际处理帧数: {processed_frame_count}。")
    # logging.info(f"结果图像保存在: {output_images_dir}") # 如果保存了图像
    logging.info(f"结果 XML (类似手动格式) 保存在: {xml_output_path}")
    logging.info("-" * 30)


# --- 命令行参数解析和入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理视频, 调用Nuclio姿态和检测服务, 保存类似手动格式的CVAT XML")
    parser.add_argument("-v", "--video", type=str, default=DEFAULT_VIDEO_PATH, help=f"输入视频路径")
    parser.add_argument("-up", "--url-pose", type=str, default=DEFAULT_NUCLIO_POSE_URL, help=f"Nuclio姿态估计URL")
    parser.add_argument("-ud", "--url-detection", type=str, default=DEFAULT_NUCLIO_DETECTION_URL, help=f"Nuclio物体检测URL")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_BASE_DIR, help=f"保存结果XML的基础目录")
    parser.add_argument("-fs", "--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help=f"帧间隔 (1=处理所有)")
    parser.add_argument("-kpt", "--keypoint-threshold", type=float, default=KEYPOINT_CONF_THRESHOLD, help=f"关键点置信度阈值 (用于 occluded)")
    parser.add_argument("-bt", "--box-threshold", type=float, default=BOX_CONF_THRESHOLD, help=f"边界框置信度阈值")

    args = parser.parse_args()

    # Update global config from args
    KEYPOINT_CONF_THRESHOLD = args.keypoint_threshold
    BOX_CONF_THRESHOLD = args.box_threshold
    frame_skip_value = max(1, args.frame_skip)

    output_base_dir = Path(args.output_dir)
    # output_images_dir = output_base_dir / "images" # 如果需要保存图像
    output_xmls_dir = output_base_dir / "annotations"
    # output_images_dir.mkdir(parents=True, exist_ok=True)
    output_xmls_dir.mkdir(parents=True, exist_ok=True)

    main(args.video, args.url_pose, args.url_detection, args.output_dir, frame_skip_value)