# -*- coding: utf-8 -*-
"""
Time:     2025/4/10 (Adapted)
Author:   AI Assistant (Based on ZhaoQi Cao's pose script)
Version:  V 1.0 (中文) (视频通用物体检测 - 输出 CVAT XML)
File:     model_http_object_detection.py
Describe: 用于调用部署在 Nuclio 上的通用物体检测函数 (处理视频输入, 保存每帧图像和单个 CVAT XML for Images)。
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
import traceback # 用于打印详细错误

# --- 配置区 ---
# !!! 修改这里为你部署的 *物体检测* 函数的 URL !!!
DEFAULT_NUCLIO_DETECTION_URL = "http://YOUR_NUCLIO_OBJECT_DETECTION_URL:PORT" # <<--- 修改为你的服务 URL!
DEFAULT_VIDEO_PATH = r"path/to/your/video.mp4" # <<--- 修改为你的默认视频路径!
DEFAULT_OUTPUT_BASE_DIR = r"C:\\Users\\DELL\\Desktop\\cam-prcess-data-1\\out_detection" # <<--- 修改为你的输出目录!
DEFAULT_FRAME_SKIP = 1  # 默认处理每 N 帧 (1 = 处理所有)
REQUEST_TIMEOUT = 60  # 等待Nuclio响应的超时时间（秒）
BOX_CONF_THRESHOLD = 0.5    # 用于过滤检测实例的边界框置信度阈值

# --- 颜色 (用于可选的绘制，如果需要) ---
# 可以创建一个颜色映射函数，为不同的类分配不同的颜色
BOX_COLOR = (0, 255, 0) # 默认绿色

# --- 创建日志和输出目录 ---
os.makedirs('./logs/', exist_ok=True)
output_base_dir = Path(DEFAULT_OUTPUT_BASE_DIR)
output_images_dir = output_base_dir / "images"
output_xmls_dir = output_base_dir / "annotations"
output_images_dir.mkdir(parents=True, exist_ok=True)
output_xmls_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE = "./logs/video_object_detection.log" # 日志文件名

# --- 设置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 辅助函数 ---
def encode_frame_to_base64(frame):
    """将OpenCV帧编码为Base64字符串。"""
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            logging.error("无法将帧编码为JPEG。")
            return None
        b64_string = base64.b64encode(buffer).decode('utf-8')
        return b64_string
    except Exception as e:
        logging.error(f"帧编码过程中出错: {e}")
        return None

def call_nuclio_object_detector(base64_image_string, nuclio_url, frame_number):
    """将Base64编码的图像发送给Nuclio物体检测函数并返回结果。"""
    payload = json.dumps({"image": base64_image_string})
    headers = {'Content-Type': 'application/json'}
    logging.debug(f"帧 {frame_number}: 正在向 {nuclio_url} 发送物体检测请求...")
    try:
        start_time = time.time()
        response = requests.post(nuclio_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        end_time = time.time()
        logging.debug(f"帧 {frame_number}: Nuclio 请求耗时 {end_time - start_time:.2f} 秒。")

        if response.status_code == 200:
            try:
                results = response.json()
                if isinstance(results, list):
                    logging.info(f"帧 {frame_number}: 收到 {len(results)} 个原始检测实例。")
                    # 验证基本结构: box, class_name, confidence
                    valid_results = []
                    for instance in results:
                        if isinstance(instance, dict) and \
                           'box' in instance and isinstance(instance['box'], (list, tuple)) and len(instance['box']) == 4 and \
                           'confidence' in instance and isinstance(instance['confidence'], (int, float)) and \
                           'class_name' in instance and isinstance(instance['class_name'], str):
                            # 应用置信度阈值过滤
                            if instance['confidence'] >= BOX_CONF_THRESHOLD:
                                valid_results.append(instance)
                            else:
                                logging.debug(f"帧 {frame_number}: 跳过低置信度实例 ({instance['class_name']}: {instance['confidence']:.2f} < {BOX_CONF_THRESHOLD})")
                        else:
                            logging.warning(f"帧 {frame_number}: 跳过格式不完整的实例: {instance}")
                    return valid_results # 返回通过阈值过滤的有效结果
                else:
                    logging.error(f"帧 {frame_number}: Nuclio 响应不是预期的列表格式。收到: {type(results)}")
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

# --- XML 生成辅助函数 ---
def pretty_print_xml(elem):
    """返回包含声明且格式化（美化）的XML字符串。"""
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        # 使用 toxml() 替代 toprettyxml() 来避免额外的空行和保证声明
        xml_str = reparsed.toxml(encoding="utf-8").decode('utf-8')
        # 手动添加换行和缩进（如果需要更美观，但toxml通常足够）
        # lines = xml_str.splitlines()
        # pretty_lines = [line for line in lines if line.strip()]
        # return '\n'.join(pretty_lines)
        return xml_str
    except Exception as e:
        logging.error(f"XML美化打印过程中出错: {e}")
        # 回退到基础 tostring
        return ET.tostring(elem, encoding='unicode')


# --- 保存物体检测结果到 CVAT XML 函数 ---
def save_detection_results_to_cvat_xml(detection_results, xml_output_path, video_filename, frame_width, frame_height, total_processed_frames, original_total_frames):
    """将所有处理帧的物体检测结果保存为单个 CVAT XML 1.1 for Images 文件。"""
    logging.info(f"正在为 {total_processed_frames} 个处理过的帧构建合并的 CVAT 图像 XML (物体检测)...")

    # --- 构建XML结构 ---
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Meta元信息
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "N/A" # CVAT 会忽略或重新分配 ID
    task_name, _ = os.path.splitext(video_filename)
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(total_processed_frames) # 使用实际处理的帧数
    ET.SubElement(task, "mode").text = "annotation" # 图像模式
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    current_time_utc = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    ET.SubElement(task, "created").text = current_time_utc
    ET.SubElement(task, "updated").text = current_time_utc
    ET.SubElement(task, "start_frame").text = "0"
    last_processed_frame_num_1_based = max(detection_results.keys()) if detection_results else 0
    stop_frame_0_based = last_processed_frame_num_1_based - 1 if last_processed_frame_num_1_based > 0 else 0
    ET.SubElement(task, "stop_frame").text = str(stop_frame_0_based)
    ET.SubElement(task, "frame_filter").text = ""

    # --- 动态提取并定义标签信息 ---
    unique_labels = set()
    if detection_results:
        for frame_dets in detection_results.values():
            if frame_dets:
                for det in frame_dets:
                    # 使用 'unknown' 作为无法获取类名的备选项
                    class_name = str(det.get('class_name', 'unknown')).strip()
                    if class_name: # 确保类名不为空
                        unique_labels.add(class_name)

    labels_elem = ET.SubElement(task, "labels")
    if not unique_labels:
        # 如果没有任何检测结果，至少添加一个 'unknown' 标签以防万一
        unique_labels.add('unknown')

    # 为每个唯一的类名创建一个 label 条目
    for label_name in sorted(list(unique_labels)):
         label_elem = ET.SubElement(labels_elem, "label")
         ET.SubElement(label_elem, "name").text = label_name
         ET.SubElement(label_elem, "color").text = "" # 让 CVAT 自动分配颜色
         ET.SubElement(label_elem, "type").text = "rectangle" # 物体检测通常是矩形框
         ET.SubElement(label_elem, "attributes") # 可以根据需要添加属性定义

    # --- 添加其他元信息 ---
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(original_total_frames - 1 if original_total_frames > 0 else 0)
    ET.SubElement(segment, "url").text = "N/A" # CVAT 通常不需要这个

    ET.SubElement(task, "owner") # 可选，留空
    ET.SubElement(task, "assignee") # 可选，留空
    ET.SubElement(task, "subset").text = "Default" # CVAT 任务子集

    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(frame_width)
    ET.SubElement(original_size, "height").text = str(frame_height)
    ET.SubElement(meta, "dumped").text = current_time_utc

    # --- 为每个处理过的帧添加 <image> 和 <box> 标注 ---
    if detection_results:
        for frame_number in sorted(detection_results.keys()): # 按视频帧号 (1-based) 排序
            detections = detection_results[frame_number] # 获取当前帧的检测结果列表

            # 创建 <image> 元素
            image_elem = ET.SubElement(root, "image")
            frame_id_0_based = frame_number - 1 # 计算 0-based 索引
            image_elem.set("id", str(frame_id_0_based))
            # 关键: name 属性通常是相对于任务的文件名或路径，这里用 0-based 索引作为标识符
            image_elem.set("name", f"{frame_id_0_based:06d}.jpg") # 假设图片已按此格式保存
            image_elem.set("width", str(frame_width))
            image_elem.set("height", str(frame_height))

            # 在 <image> 元素内部为每个检测添加 <box>
            if detections:
                for det in detections:
                    label_name = str(det.get('class_name', 'unknown')).strip()
                    if not label_name: label_name = 'unknown' # 再次确保标签不为空
                    box = det.get('box')
                    confidence = det.get('confidence', 0.0)

                    if box and len(box) == 4:
                        # CVAT 需要 xtl, ytl, xbr, ybr 格式
                        xtl, ytl, xbr, ybr = map(float, box)

                        # 创建 <box> 元素
                        box_elem = ET.SubElement(image_elem, "box")
                        box_elem.set("label", label_name)
                        box_elem.set("occluded", "0") # 默认为0，可以根据需要修改逻辑
                        box_elem.set("source", "auto") # 标记为自动生成
                        box_elem.set("xtl", f"{xtl:.2f}")
                        box_elem.set("ytl", f"{ytl:.2f}")
                        box_elem.set("xbr", f"{xbr:.2f}")
                        box_elem.set("ybr", f"{ybr:.2f}")
                        box_elem.set("z_order", "0") # 控制堆叠顺序，默认为0

                        # 可以选择将置信度添加为属性
                        attr_conf = ET.SubElement(box_elem, "attribute", name="confidence")
                        attr_conf.text = f"{confidence:.4f}"
                    else:
                        logging.warning(f"帧 {frame_number}: 检测实例 '{label_name}' 的 box 数据无效: {box}")

            # else: 如果 detections 为空列表, 则该 <image> 标签下没有子元素

    # --- 结束构建XML结构 ---

    # --- 写入 XML 文件 ---
    try:
        # 使用 ElementTree 自带的写入器，并启用 indent (需要 Python 3.9+)
        tree = ET.ElementTree(root)
        if hasattr(ET, 'indent'):
            ET.indent(tree, space="  ", level=0) # 格式化输出
        tree.write(str(xml_output_path), encoding='utf-8', xml_declaration=True)
        logging.info(f"合并的 CVAT 物体检测 XML 已保存: {xml_output_path}")

        # # 或者使用之前的 pretty_print_xml (如果上面的格式不理想)
        # full_xml_string = pretty_print_xml(root)
        # if full_xml_string:
        #     with open(xml_output_path, "w", encoding="utf-8") as f:
        #         f.write(full_xml_string)
        #     logging.info(f"合并的 CVAT 物体检测 XML 已保存: {xml_output_path}")
        # else:
        #      raise ValueError("pretty_print_xml 返回空字符串")

    except Exception as e:
        logging.error(f"保存合并的 CVAT XML 文件失败: {e}")
        print("--- ERROR SAVING XML --- ")
        traceback.print_exc()
        print("--- END ERROR --- ")


# --- 主执行逻辑 ---
def main(video_path, nuclio_url, output_dir, frame_skip):
    """主函数，执行视频处理、物体检测、保存帧和合并XML的流程。"""
    output_dir = Path(output_dir)
    output_images_dir = output_dir / "images"
    output_xmls_dir = output_dir / "annotations"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_xmls_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始处理视频: {video_path}")
    logging.info(f"Nuclio 物体检测 URL: {nuclio_url}")
    logging.info(f"处理帧间隔: 每 {frame_skip} 帧")
    logging.info(f"输出目录 (图像和XML): {output_dir}")
    logging.info(f"边界框置信度阈值: {BOX_CONF_THRESHOLD}")

    # 1. 检查视频文件
    if not os.path.exists(video_path):
        logging.error(f"输入视频文件未找到: {video_path}")
        return

    # 2. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_path}")
        return

    # 3. 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_total_frames <= 0:
         logging.warning("无法获取视频总帧数，将尝试读取第一帧获取尺寸。")
         if frame_width == 0 or frame_height == 0:
             ret, frame = cap.read();
             if ret:
                 frame_height, frame_width = frame.shape[:2]
                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置回第一帧
                 logging.info("成功从第一帧读取尺寸。")
             else:
                 logging.error("无法读取视频帧尺寸。"); cap.release(); return
    logging.info(f"视频信息 - 尺寸: {frame_width}x{frame_height}, FPS: {fps:.2f}, 总帧数: {original_total_frames if original_total_frames > 0 else '未知'}")
    if frame_width == 0 or frame_height == 0:
        logging.error("无法确定视频帧尺寸，无法继续。"); cap.release(); return

    # 4. 初始化
    frame_count = 0
    processed_frame_count = 0
    all_detection_results = {} # 存储所有帧的检测结果: {frame_number (1-based): [detections]}
    start_process_time = time.time()

    # 5. 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("视频处理到达结尾或读取错误。")
            break
        frame_count += 1

        # --- 跳帧逻辑 ---
        if frame_skip > 1 and frame_count % frame_skip != 0:
            continue

        # --- 处理当前帧 ---
        processed_frame_count += 1
        logging.info(f"正在处理帧 {frame_count}{f'/{original_total_frames}' if original_total_frames > 0 else ''}...")

        # --- 保存原始帧图像 ---
        frame_id_0_based = frame_count - 1
        frame_filename = f"{frame_id_0_based:06d}.jpg"
        output_image_path = output_images_dir / frame_filename
        try:
            cv2.imwrite(str(output_image_path), frame)
            logging.debug(f"原始帧图像已保存: {output_image_path}")
        except Exception as e:
            logging.error(f"保存帧 {frame_count} 图像失败: {e}")
            # 可以选择跳过此帧的处理
            # all_detection_results[frame_count] = []
            # continue

        # 编码帧
        b64_string = encode_frame_to_base64(frame)
        if not b64_string:
            logging.warning(f"因编码错误跳过帧 {frame_count} 的处理。")
            all_detection_results[frame_count] = [] # 记录空结果
            continue

        # 调用 Nuclio 物体检测函数
        detections = call_nuclio_object_detector(b64_string, nuclio_url, frame_count)

        # --- 累积结果 ---
        if detections is not None:
             all_detection_results[frame_count] = detections # 存储过滤后的检测结果
             logging.info(f"帧 {frame_count}: 检测到 {len(detections)} 个有效物体实例。")
        else:
            # 如果调用失败或未返回有效结果，记录一个空列表
            all_detection_results[frame_count] = []
            logging.warning(f"帧 {frame_count}: 未收到有效检测结果，将在XML中标记为空帧。")
        # --------------------

    # --- 清理与结束 ---
    cap.release()
    # cv2.destroyAllWindows() # 在无 GUI 环境下不需要

    # --- 保存合并的 XML 文件 ---
    video_basename = os.path.basename(video_path)
    video_name_no_ext, _ = os.path.splitext(video_basename)
    xml_output_filename = f"{video_name_no_ext}_cvat_detections.xml" # 新文件名
    xml_output_path = output_xmls_dir / xml_output_filename

    # 调用保存函数
    save_detection_results_to_cvat_xml(
        all_detection_results,
        xml_output_path,
        video_basename, # 用于 <task><name>
        frame_width,
        frame_height,
        processed_frame_count, # 用于 <task><size>
        original_total_frames # 用于 <segments><segment><stop>
    )
    # -------------------------

    end_process_time = time.time()
    total_time = max(0, end_process_time - start_process_time)
    logging.info("-" * 30)
    logging.info(f"视频处理完成。")
    logging.info(f"总耗时: {timedelta(seconds=int(total_time))}")
    logging.info(f"总共读取帧数: {frame_count}。")
    logging.info(f"实际处理帧数: {processed_frame_count}。")
    logging.info(f"结果图像保存在: {output_images_dir}")
    logging.info(f"结果合并XML保存在: {xml_output_path}")
    if total_time > 0 and processed_frame_count > 0:
        fps_proc = processed_frame_count / total_time
        logging.info(f"平均处理速度: {fps_proc:.2f} 帧/秒")
    logging.info("-" * 30)


# --- 命令行参数解析和入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="处理视频, 调用Nuclio物体检测, 保存每帧图像和单个合并的CVAT XML (图像模式)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--video", type=str, default=DEFAULT_VIDEO_PATH, help="输入视频路径")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_NUCLIO_DETECTION_URL, help="Nuclio 物体检测函数 URL")
    parser.add_argument("-od", "--output-dir", type=str, default=DEFAULT_OUTPUT_BASE_DIR, help="保存结果的基础目录")
    parser.add_argument("-fs", "--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help="帧间隔 (1=处理所有)")
    parser.add_argument("-bt", "--box-threshold", type=float, default=BOX_CONF_THRESHOLD, help="边界框置信度阈值 (过滤检测实例)")

    args = parser.parse_args()

    # 更新全局配置（如果需要）
    BOX_CONF_THRESHOLD = args.box_threshold
    # 更新路径和跳帧值
    frame_skip_value = max(1, args.frame_skip)
    output_base_dir = Path(args.output_dir)
    output_images_dir = output_base_dir / "images"
    output_xmls_dir = output_base_dir / "annotations"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_xmls_dir.mkdir(parents=True, exist_ok=True)

    # 检查 Nuclio URL 是否被修改
    if args.url == "http://YOUR_NUCLIO_OBJECT_DETECTION_URL:PORT":
        logging.error("错误：请修改脚本中的 DEFAULT_NUCLIO_DETECTION_URL 为你实际的 Nuclio 服务地址，或使用 -u/--url 参数指定。")
    # 检查视频路径是否为默认值
    elif args.video == "path/to/your/video.mp4":
         logging.warning("警告：正在使用默认视频路径，请确认是否需要修改或使用 -v/--video 参数指定。")
         # 可以选择在这里退出，或者继续执行（取决于你的需求）
         # exit(1)
         main(args.video, args.url, args.output_dir, frame_skip_value)
    else:
        main(args.video, args.url, args.output_dir, frame_skip_value)