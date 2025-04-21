# -*- coding: utf-8 -*-
"""
Time:     2025/4/17 (Fixed XML Generation & JSON Output)
Author:   ZhaoQi Cao(czq) & AI Assistant
Version:  V 1.3 (中文) (视频PaddleOCR测试 - 修复XML生成, 输出JSON和标注视频)
File:     model_http_paddleocr.py
Describe: 用于测试部署在Nuclio上的PaddleOCR函数 (处理视频输入, 保存JSON结果,
          生成包含所有帧的CVAT XML, 并可选生成带标注的视频)
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
from pathlib import Path
import random

# --- PaddleOCR 可视化工具 ---
try:
    from paddleocr import draw_ocr
except ImportError:
    print("警告: 未找到 paddleocr 库。如果需要绘制标注视频，请运行 'pip install paddleocr'")
    draw_ocr = None

# --- 配置区 ---
DEFAULT_NUCLIO_OCR_URL = "http://192.168.10.158:32799" # <<--- !!! 修改这里: 指向 PaddleOCR 函数 URL !!!
DEFAULT_VIDEO_PATH = r"E:\文字艺术\16个样例\87.mp4" # <<--- **修改这里** 默认输入视频路径
DEFAULT_OUTPUT_BASE_DIR =r"E:\文字艺术\16样例xml\7"
DEFAULT_FRAME_SKIP = 1
DEFAULT_SAVE_VIDEO = True
REQUEST_TIMEOUT = 180
VIDEO_CODEC = 'mp4v'
VIDEO_EXTENSION = '.mp4'
DEFAULT_FONT_PATH = '/Users/zg/PycharmProjects/CVAT_model_nuclio/src/可视化/ZhouZiFangTi241010-TTF-2.ttf' # <<--- !!! 修改为你的字体路径 !!!

# 创建日志和输出目录
os.makedirs('./logs/', exist_ok=True)
output_base_dir = Path(DEFAULT_OUTPUT_BASE_DIR)
output_jsons_dir = output_base_dir / "json_results"
output_xmls_dir = output_base_dir / "annotations" # XML 输出目录
output_videos_dir = output_base_dir / "annotated_videos"
output_jsons_dir.mkdir(parents=True, exist_ok=True)
output_xmls_dir.mkdir(parents=True, exist_ok=True) # 创建 XML 目录
output_videos_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE = "./logs/video_paddleocr_testing_annotated.log"

# --- 设置日志记录 ---
logging.basicConfig(
    level=logging.INFO, # 可以改为 logging.DEBUG 查看更详细信息
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # 添加函数名
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
    return COLOR_LIST_BGR[index % len(COLOR_LIST_BGR)]

def bgr_to_hex(bgr_color):
    return '#{:02x}{:02x}{:02x}'.format(bgr_color[2], bgr_color[1], bgr_color[0])

# --- 辅助函数 ---
def encode_frame_to_base64(frame):
    try:
        # 压缩图像以减少网络传输量，但保持较高质量
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        is_success, buffer = cv2.imencode(".jpg", frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8') if is_success else None
    except Exception as e:
        logging.error(f"帧编码时发生异常: {e}"); return None

def call_nuclio_paddleocr(base64_image_string, nuclio_url, frame_number, language='ch'):
    payload_dict = {"image": base64_image_string, "language": language}
    payload = json.dumps(payload_dict)
    headers = {'Content-Type': 'application/json'}
    
    # 添加简单的重试机制
    max_retries = 2
    retry_delay = 2
    
    for retry in range(max_retries + 1):
        try:
            start_time = time.time()
            response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
            end_time = time.time()
            logging.debug(f"帧 {frame_number}: Nuclio 请求耗时 {end_time - start_time:.2f} 秒。")

            if response.status_code == 200:
                try:
                    results_json = response.json()
                    if isinstance(results_json, list):
                        logging.info(f"帧 {frame_number}: 收到 {len(results_json)} 个检测结果 (请求语言: {language})。")
                        return results_json
                    else:
                        logging.warning(f"帧 {frame_number}: Nuclio 返回的不是预期的列表格式，类型: {type(results_json)}。")
                        return [] # 返回空列表表示成功请求但无有效结果
                except json.JSONDecodeError:
                    logging.error(f"帧 {frame_number}: 解析 Nuclio JSON 失败。状态码: {response.status_code}, 响应文本: {response.text[:200]}...")
                    if retry < max_retries:
                        logging.warning(f"帧 {frame_number}: 尝试第 {retry+1} 次重试...")
                        time.sleep(retry_delay)
                        continue
                    return None
            elif response.status_code == 429:  # 接口请求过多
                if retry < max_retries:
                    wait_time = (retry + 1) * retry_delay
                    logging.warning(f"帧 {frame_number}: 请求频率限制(429)，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"帧 {frame_number}: 多次尝试后仍受到请求频率限制。")
                    return None
            else:
                error_message = f"帧 {frame_number}: Nuclio 错误。状态码: {response.status_code}"
                try: error_body = response.json(); error_message += f", 响应: {json.dumps(error_body, indent=2, ensure_ascii=False)}"
                except json.JSONDecodeError: error_message += f", 响应文本: {response.text[:500]}"
                logging.error(error_message)
                
                if retry < max_retries and response.status_code >= 500:  # 只对服务器错误重试
                    logging.warning(f"帧 {frame_number}: 尝试第 {retry+1} 次重试...")
                    time.sleep(retry_delay)
                    continue
                return None
        except requests.exceptions.Timeout:
            if retry < max_retries:
                logging.warning(f"帧 {frame_number}: 请求超时({REQUEST_TIMEOUT}秒)，尝试第 {retry+1} 次重试...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"帧 {frame_number}: 请求 Nuclio 超时 ({REQUEST_TIMEOUT} 秒)，已达到最大重试次数。"); 
                return None
        except requests.exceptions.RequestException as e:
            if retry < max_retries:
                logging.warning(f"帧 {frame_number}: 请求异常 ({str(e)})，尝试第 {retry+1} 次重试...")
                time.sleep(retry_delay)
                continue
            else:
                logging.error(f"帧 {frame_number}: 请求 Nuclio 失败: {e}"); 
                return None
        except Exception as e:
            logging.error(f"帧 {frame_number}: 调用 Nuclio 时发生意外错误: {e}"); 
            return None
    
    # 如果所有重试都失败
    return None

def pretty_print_xml(elem):
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = '\n'.join([line for line in reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8').split('\n') if line.strip()])
        if not pretty_xml.strip().startswith("<?xml"): return '<?xml version="1.0" encoding="utf-8"?>\n' + pretty_xml
        return pretty_xml
    except Exception as e:
        logging.error(f"XML美化打印出错: {e}")
        try: ET.indent(elem, space="  "); xml_string = ET.tostring(elem, encoding='unicode'); return xml_string
        except Exception: return ET.tostring(elem, encoding='unicode')

def is_xml_char_valid(char):
    """检查字符是否为有效的 XML 字符"""
    code = ord(char)
    # XML 1.0 valid range: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    return (code == 0x9 or code == 0xA or code == 0xD or
            (0x20 <= code <= 0xD7FF) or
            (0xE000 <= code <= 0xFFFD) or
            (0x10000 <= code <= 0x10FFFF))

def clean_xml_text(text):
    """清理字符串，移除无效的 XML 字符"""
    if text is None: return ""
    cleaned = "".join(c for c in str(text) if is_xml_char_valid(c))
    # 可选：替换特殊 XML 字符（通常 ElementTree 会处理，但手动处理更安全）
    # cleaned = cleaned.replace("&", "&").replace("<", "<").replace(">", ">").replace("\"", """).replace("'", "'")
    return cleaned

# --- 绘图函数 ---
def draw_ocr_on_frame(frame, ocr_json_result, font_path, frame_count):
    """使用 paddleocr.draw_ocr 在帧上绘制 OCR 结果 (从 JSON 格式转换)。"""
    if not ocr_json_result: return frame
    if draw_ocr is None:
        logging.log(logging.WARNING if frame_count % 100 == 1 else logging.DEBUG, # 减少警告频率
                    "paddleocr.draw_ocr 未导入，仅绘制检测框。")
        vis_frame = frame.copy()
        for item in ocr_json_result:
             try: box = np.array(item['box']).astype(np.int32); cv2.polylines(vis_frame, [box], True, (0, 255, 0), 2)
             except Exception: pass # 忽略绘制错误
        return vis_frame
    if not os.path.exists(font_path):
        logging.log(logging.WARNING if frame_count % 100 == 1 else logging.DEBUG, # 减少警告频率
                    f"字体文件未找到: {font_path}。将只绘制检测框。")
        vis_frame = frame.copy()
        for item in ocr_json_result:
             try: box = np.array(item['box']).astype(np.int32); cv2.polylines(vis_frame, [box], True, (0, 255, 0), 2)
             except Exception: pass
        return vis_frame
    else:
        try:
            # 从 JSON 格式转换回 PaddleOCR draw_ocr 需要的格式
            boxes = [item['box'] for item in ocr_json_result]
            txts = [item['text'] for item in ocr_json_result]
            scores = [item['confidence'] for item in ocr_json_result]
            vis_frame = draw_ocr(frame, boxes, txts, scores, font_path=font_path)
            return vis_frame
        except Exception as e:
            logging.error(f"使用 paddleocr.draw_ocr 绘图时出错: {e}", exc_info=True)
            return frame

# --- CVAT XML 生成函数 (修复循环和文本清理) ---
def save_results_to_cvat_ocr_xml(all_results, xml_output_path, video_filename, frame_width, frame_height, original_total_frames):
    """
    将 PaddleOCR 结果保存为 CVAT for images 格式的 XML。
    确保为所有帧创建 <image> 标签，并清理文本内容。
    """
    logging.info(f"正在为 {original_total_frames} 帧构建 CVAT OCR XML (处理了 {len(all_results)} 帧的结果)...")
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- Meta (省略重复代码，与之前版本相同) ---
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "N/A"; task_name, _ = os.path.splitext(video_filename); ET.SubElement(task, "name").text = f"{task_name}_ocr_results"
    ET.SubElement(task, "size").text = str(original_total_frames); ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"; ET.SubElement(task, "bugtracker").text = ""
    current_time_utc = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "+00:00"
    ET.SubElement(task, "created").text = current_time_utc; ET.SubElement(task, "updated").text = current_time_utc
    ET.SubElement(task, "start_frame").text = "0"; ET.SubElement(task, "stop_frame").text = str(original_total_frames - 1 if original_total_frames > 0 else 0); ET.SubElement(task, "frame_filter").text = ""
    labels_elem = ET.SubElement(task, "labels"); label_elem = ET.SubElement(labels_elem, "label"); ET.SubElement(label_elem, "name").text = "text"; ET.SubElement(label_elem, "color").text = "#FF0000"; ET.SubElement(label_elem, "type").text = "polygon"
    attributes_elem = ET.SubElement(label_elem, "attributes"); attr_text = ET.SubElement(attributes_elem, "attribute"); ET.SubElement(attr_text, "name").text = "text_content"; ET.SubElement(attr_text, "mutable").text = "true"; ET.SubElement(attr_text, "input_type").text = "text"; ET.SubElement(attr_text, "default_value").text = ""; ET.SubElement(attr_text, "values").text = ""
    attr_conf = ET.SubElement(attributes_elem, "attribute"); ET.SubElement(attr_conf, "name").text = "confidence"; ET.SubElement(attr_conf, "mutable").text = "true"; ET.SubElement(attr_conf, "input_type").text = "number"; ET.SubElement(attr_conf, "default_value").text = "0.0"; ET.SubElement(attr_conf, "values").text = "0.0;1.0;0.01"
    segments = ET.SubElement(task, "segments"); segment = ET.SubElement(segments, "segment"); ET.SubElement(segment, "id").text = "0"; ET.SubElement(segment, "start").text = "0"; ET.SubElement(segment, "stop").text = str(original_total_frames - 1 if original_total_frames > 0 else 0); ET.SubElement(segment, "url").text = "N/A"
    ET.SubElement(task, "owner").text = "N/A"; ET.SubElement(task, "assignee").text = "N/A"; ET.SubElement(task, "subset").text = "Default"
    original_size = ET.SubElement(meta, "original_size"); ET.SubElement(original_size, "width").text = str(frame_width); ET.SubElement(original_size, "height").text = str(frame_height); ET.SubElement(meta, "dumped").text = current_time_utc
    # --- Meta End ---

    # --- 添加图像和多边形数据 ---
    # *** 修改: 循环遍历所有帧索引 ***
    for frame_idx in range(original_total_frames):
        frame_number = frame_idx + 1 # all_results 使用 1-based key
        # 安全地获取该帧的检测结果，如果该帧未处理或无结果，则返回空列表
        detections = all_results.get(frame_number, [])

        # --- 创建 <image> 标签 ---
        image_elem = ET.SubElement(root, "image")
        image_elem.set("id", str(frame_idx)) # 0-based ID
        image_elem.set("name", f"frame_{frame_idx:06d}")
        image_elem.set("width", str(frame_width))
        image_elem.set("height", str(frame_height))

        # 检查 detections 是否为 None 或非列表（防御性编程）
        if detections is None or not isinstance(detections, list):
             if detections is not None: # 记录非列表的意外情况
                  logging.warning(f"帧 {frame_number} (ID {frame_idx}): 预期结果为列表，但得到 {type(detections)}。跳过此帧的标注。")
             # else: detections 为 None 已在 call_nuclio_paddleocr 中记录错误
             continue # 处理下一帧

        if not detections: # 如果列表为空
            logging.debug(f"帧 {frame_number} (ID {frame_idx}): 无检测结果，写入空的 <image> 标签。")
            continue # 处理下一帧

        # --- 处理该帧的检测结果 ---
        logging.debug(f"帧 {frame_number} (ID {frame_idx}): 找到 {len(detections)} 个检测结果，开始处理...")
        for det_idx, detection_item in enumerate(detections):
            # detection_item 应该是 {"box": [[x,y],...], "text": "...", "confidence": ...}
            if not isinstance(detection_item, dict):
                 logging.warning(f"帧 {frame_number} (ID {frame_idx}), 索引 {det_idx}: 检测结果项不是字典，跳过。类型: {type(detection_item)}")
                 continue

            try:
                # 从字典中提取数据
                box = detection_item.get('box')
                text = detection_item.get('text')
                score = detection_item.get('confidence')

                # 检查提取的数据是否有效
                if box is None or text is None or score is None:
                    logging.warning(f"帧 {frame_number} (ID {frame_idx}), 索引 {det_idx}: 缺少 box, text 或 confidence，跳过。数据: {detection_item}")
                    continue
                if not isinstance(box, list) or len(box) != 4 or not all(isinstance(p, list) and len(p) == 2 for p in box):
                     logging.warning(f"帧 {frame_number} (ID {frame_idx}), 索引 {det_idx}: box 格式不正确，跳过。Box: {box}")
                     continue

                # --- 创建 <polygon> ---
                poly_elem = ET.SubElement(image_elem, "polygon")
                poly_elem.set("label", "text") # 使用通用标签
                poly_elem.set("source", "auto")
                poly_elem.set("occluded", "0")
                poly_elem.set("outside", "0")
                poly_elem.set("keyframe", "1")
                poly_elem.set("z_order", "0")

                # --- 处理多边形点 ---
                points_px = np.array(box).round(2) # 假设 box 已经是像素坐标
                points_str = ";".join([f"{p[0]},{p[1]}" for p in points_px])
                poly_elem.set("points", points_str)

                # --- 添加属性 (带清理) ---
                attr_text_elem = ET.SubElement(poly_elem, "attribute", name="text_content")
                cleaned_text = clean_xml_text(text) # *** 修改: 清理文本 ***
                if cleaned_text != str(text): # 记录清理操作
                     logging.log(logging.WARNING if frame_number % 50 == 1 else logging.DEBUG, # 减少警告频率
                                 f"帧 {frame_number}, 实例 {det_idx}: 原始文本包含无效XML字符，已清理。 Orig: '{text}', Cleaned: '{cleaned_text}'")
                attr_text_elem.text = cleaned_text

                attr_conf_elem = ET.SubElement(poly_elem, "attribute", name="confidence")
                attr_conf_elem.text = f"{float(score):.4f}" # 确保 score 是 float

            except Exception as e:
                 logging.error(f"帧 {frame_number} (ID {frame_idx}), 实例 {det_idx}: 生成 XML polygon 时意外错误: {e}", exc_info=True)
                 # 即使单个 polygon 出错，也继续处理下一个

    logging.info("CVAT XML 结构构建完成。")
    # --- 保存 XML ---
    try:
        xml_string = pretty_print_xml(root)
        xml_output_path_str = str(xml_output_path)
        with open(xml_output_path_str, "w", encoding="utf-8") as f:
            f.write(xml_string)
        logging.info(f"CVAT OCR XML 已保存: {xml_output_path_str}")
    except Exception as e:
        logging.error(f"保存 XML 失败: {e}")

# 添加一个辅助函数用于处理单个帧
def process_and_save_frame(frame, frame_number, nuclio_url, language, font_path, 
                          save_video_flag, video_writer, all_results):
    """处理单个帧并保存结果"""
    logging.info(f"正在处理帧 {frame_number}...")
    start_frame_time = time.time()
    
    # 优化图像大小以提高OCR性能
    height, width = frame.shape[:2]
    if width > 1280:  # 如果宽度大于1280，调整大小
        scale = 1280 / width
        new_height = int(height * scale)
        frame_for_ocr = cv2.resize(frame, (1280, new_height), interpolation=cv2.INTER_AREA)
    else:
        frame_for_ocr = frame
    
    b64_string = encode_frame_to_base64(frame_for_ocr)
    ocr_result_json = None
    
    if b64_string:
        ocr_result_json = call_nuclio_paddleocr(b64_string, nuclio_url, frame_number, language=language)
    else:
        logging.warning(f"因编码错误跳过帧 {frame_number} 的处理。")
    
    # 存储结果，确保是列表或空列表
    all_results[frame_number] = ocr_result_json if isinstance(ocr_result_json, list) else []
    
    # 处理视频输出
    if save_video_flag and video_writer:
        annotated_frame = frame
        if all_results[frame_number]:  # 只在有结果时绘制
            annotated_frame = draw_ocr_on_frame(frame, all_results[frame_number], font_path, frame_number)
        try: 
            video_writer.write(annotated_frame)
        except Exception as e: 
            logging.error(f"写入视频帧 {frame_number} 时出错: {e}")
    
    end_frame_time = time.time()
    logging.info(f"帧 {frame_number} 处理耗时: {end_frame_time - start_frame_time:.2f} 秒。")
    return all_results[frame_number]

# --- 主执行逻辑 ---
def main(video_path, nuclio_url, output_dir, frame_skip, font_path, save_video_flag, target_language):
    output_dir = Path(output_dir)
    output_jsons_dir = output_dir / "json_results"
    output_xmls_dir = output_dir / "annotations" # 添加 XML 输出目录
    output_videos_dir = output_dir / "annotated_videos"
    output_jsons_dir.mkdir(parents=True, exist_ok=True)
    output_xmls_dir.mkdir(parents=True, exist_ok=True) # 创建 XML 目录
    if save_video_flag:
        output_videos_dir.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(font_path):
             logging.warning(f"!!! 字体文件未找到: {font_path} !!!")
             logging.warning("将无法在输出视频中绘制识别的文本。")

    logging.info(f"开始处理视频: {video_path}")
    logging.info(f"Nuclio PaddleOCR URL: {nuclio_url}")
    logging.info(f"目标语言: {target_language}")
    logging.info(f"帧间隔: 每 {frame_skip} 帧")
    logging.info(f"JSON 输出目录: {output_jsons_dir}")
    logging.info(f"XML 输出目录: {output_xmls_dir}") # 记录 XML 目录
    if save_video_flag:
        logging.info(f"带标注视频输出目录: {output_videos_dir}")
        logging.info(f"绘图字体路径: {font_path}")
    else:
        logging.info("不保存带标注的视频。")

    if not os.path.exists(video_path): logging.error(f"视频未找到: {video_path}"); return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): logging.error(f"无法打开视频: {video_path}"); return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # ... (手动计算 total_frames 的逻辑保持不变) ...
    if original_total_frames <= 0:
        logging.warning("无法直接获取视频总帧数，尝试手动计数...")
        frame_temp_count = 0
        while True: ret_temp, _ = cap.read(); frame_temp_count += 1;
        original_total_frames = frame_temp_count; cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logging.info(f"手动计数得到总帧数: {original_total_frames}")
    else:
        logging.info(f"视频信息 - 尺寸: {frame_width}x{frame_height}, FPS: {fps:.2f}, 总帧数: {original_total_frames}")


    video_writer = None
    output_video_path = None
    if save_video_flag:
        # ... (VideoWriter 初始化逻辑保持不变) ...
        video_basename = os.path.basename(video_path)
        video_name_no_ext, _ = os.path.splitext(video_basename)
        output_video_filename = f"{video_name_no_ext}_ocr_annotated{VIDEO_EXTENSION}"
        output_video_path = output_videos_dir / output_video_filename
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        try:
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
            if not video_writer.isOpened(): raise RuntimeError("VideoWriter 未打开")
            logging.info(f"带标注视频将保存到: {output_video_path}")
        except Exception as e:
            logging.error(f"初始化 VideoWriter 时出错: {e}. 禁用视频保存。")
            video_writer = None; save_video_flag = False


    frame_count = 0
    processed_frame_count = 0
    all_results = {} # key: frame_number (1-based), value: ocr_json_result_list
    start_process_time = time.time()

    # 优化内存使用和批处理
    frame_buffer = []  # 建立一个小型帧缓冲区，用于处理
    buffer_size = 4    # 缓冲区大小
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            # 处理缓冲区中的剩余帧
            for buffered_frame, frame_num in frame_buffer:
                if frame_skip <= 1 or (frame_num - 1) % frame_skip == 0:
                    process_and_save_frame(buffered_frame, frame_num, nuclio_url, target_language, 
                                          font_path, save_video_flag, video_writer, all_results)
                    processed_frame_count += 1
                elif save_video_flag and video_writer:
                    try: video_writer.write(buffered_frame)
                    except Exception as e: logging.error(f"写入跳过的视频帧 {frame_num} 时出错: {e}")
            
            logging.info("视频处理结束或读取错误。")
            break
            
        frame_count += 1
        
        # 添加到缓冲区
        frame_buffer.append((frame.copy(), frame_count))
        
        # 当缓冲区达到指定大小时处理
        if len(frame_buffer) >= buffer_size:
            for buffered_frame, frame_num in frame_buffer:
                if frame_skip <= 1 or (frame_num - 1) % frame_skip == 0:
                    process_and_save_frame(buffered_frame, frame_num, nuclio_url, target_language, 
                                          font_path, save_video_flag, video_writer, all_results)
                    processed_frame_count += 1
                elif save_video_flag and video_writer:
                    try: video_writer.write(buffered_frame)
                    except Exception as e: logging.error(f"写入跳过的视频帧 {frame_num} 时出错: {e}")
            
            # 清空缓冲区
            frame_buffer = []

    cap.release()
    if video_writer: video_writer.release(); logging.info("带标注视频写入器已释放。")

    # --- 保存聚合的 JSON 结果 ---
    video_basename = os.path.basename(video_path)
    video_name_no_ext, _ = os.path.splitext(video_basename)
    json_output_filename = f"{video_name_no_ext}_ocr_results.json"
    json_output_path = output_jsons_dir / json_output_filename
    try:
        logging.info(f"正在将所有帧的 OCR 结果聚合保存到 JSON 文件: {json_output_path}")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logging.info(f"聚合 JSON 结果已保存。")
    except Exception as e:
        logging.error(f"保存聚合 JSON 文件失败: {e}")

    # --- 保存 CVAT XML 文件 ---
    xml_output_filename = f"{video_name_no_ext}_cvat_ocr.xml" # <--- XML 文件名
    xml_output_path = output_xmls_dir / xml_output_filename   # <--- XML 输出路径
    save_results_to_cvat_ocr_xml(
        all_results,
        xml_output_path,
        video_basename,
        frame_width,
        frame_height,
        original_total_frames # 使用原始总帧数确保所有 image 标签被创建
    )

    # --- 结束日志 ---
    end_process_time = time.time(); total_time = max(0, end_process_time - start_process_time)
    logging.info("-" * 30); logging.info(f"视频处理完成。")
    logging.info(f"总耗时: {timedelta(seconds=int(total_time))}")
    logging.info(f"总读取帧数: {frame_count}。")
    logging.info(f"实际处理帧数 (发送到Nuclio): {processed_frame_count}。")
    logging.info(f"结果 JSON 保存在: {json_output_path}")
    logging.info(f"结果 XML 保存在: {xml_output_path}") # <--- 添加 XML 路径日志
    if output_video_path and os.path.exists(output_video_path): logging.info(f"带标注视频保存在: {output_video_path}")
    elif save_video_flag and not output_video_path: logging.warning("设置了保存视频但未能生成视频文件路径。")
    logging.info("-" * 30)


# --- 命令行参数解析和入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理视频, 调用 Nuclio PaddleOCR 函数, 保存 JSON 和 CVAT XML 结果, 并可选生成带标注视频")
    parser.add_argument("-v", "--video", type=str, default=DEFAULT_VIDEO_PATH, help=f"输入视频路径 (默认: {DEFAULT_VIDEO_PATH})")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_NUCLIO_OCR_URL, help=f"Nuclio PaddleOCR 函数 URL (默认: {DEFAULT_NUCLIO_OCR_URL})")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_BASE_DIR, help=f"保存结果的基础目录 (默认: {DEFAULT_OUTPUT_BASE_DIR})")
    parser.add_argument("-fs", "--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help=f"帧间隔 (1=处理所有) (默认: {DEFAULT_FRAME_SKIP})")
    parser.add_argument("-fnt", "--font-path", type=str, default=DEFAULT_FONT_PATH, help=f"用于绘制文本的 TTF 字体文件路径 (默认: {DEFAULT_FONT_PATH})")
    parser.add_argument("-lang", "--language", type=str, default='ch', help="要传递给 PaddleOCR 的语言代码 (例如 'ch', 'en', 'korean', 'ch+en') (默认: 'ch')")
    parser.add_argument("--save-video", action='store_true', help=f"是否保存带标注的视频文件 (默认使用常量: {DEFAULT_SAVE_VIDEO})")
    parser.add_argument("--no-save-video", action='store_false', dest='save_video', help="明确指定不保存带标注的视频文件")
    parser.set_defaults(save_video=DEFAULT_SAVE_VIDEO)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志记录级别。默认: INFO")


    args = parser.parse_args()

    # 更新日志级别
    logging.getLogger().setLevel(args.log_level.upper())

    frame_skip_value = max(1, args.frame_skip)
    save_video_flag = args.save_video
    font_path_value = args.font_path
    target_language_arg = args.language

    output_base_dir_arg = Path(args.output_dir)
    output_jsons_dir_arg = output_base_dir_arg / "json_results"
    output_xmls_dir_arg = output_base_dir_arg / "annotations" # <--- 定义 XML 目录
    output_videos_dir_arg = output_base_dir_arg / "annotated_videos"
    output_jsons_dir_arg.mkdir(parents=True, exist_ok=True)
    output_xmls_dir_arg.mkdir(parents=True, exist_ok=True) # <--- 创建 XML 目录
    if save_video_flag:
        output_videos_dir_arg.mkdir(parents=True, exist_ok=True)

    if args.url == "http://<YOUR_NUCLIO_IP>:<PADDLEOCR_FUNCTION_PORT>":
        logging.warning("Nuclio URL 未通过命令行参数修改，请确保默认 URL 正确或使用 --url 参数指定!")

    main(args.video, args.url, args.output_dir, frame_skip_value, font_path_value, save_video_flag, target_language=target_language_arg)