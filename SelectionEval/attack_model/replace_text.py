import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import difflib

# def smart_replace_text(
#     image_path,
#     target_bbox,  # (x, y, w, h)
#     new_text,
#     output_path="output.jpg",
#     font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
#     font_weight="normal",  # "normal" 或 "bold"
#     font_size=None,  # 新增：允许指定字体大小
#     text_color="black",
#     background_color="white",
#     padding=2,
#     line_height=None  # 新增：行高参数
# ):
#     """智能文字替换 - 精准适配字体大小和样式，并恢复背景"""
#     pil_img = Image.open(image_path).convert("RGB")
#     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

#     # 扩展文字框区域并生成掩码
#     x, y, w, h = target_bbox
#     x0, y0 = max(0, x - padding), max(0, y - padding)
#     x1, y1 = min(cv_img.shape[1], x + w + padding), min(cv_img.shape[0], y + h + padding)
#     mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
#     mask[y0:y1, x0:x1] = 255

#     # 背景恢复（Telea 算法）
#     restored = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
#     img = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)

#     # 清除原文字区域
#     draw.rectangle([x0, y0, x1, y1], fill=background_color)

#     # 字体选择支持粗体
#     def get_font_path(fp, weight):
#         if weight == "bold":
#             base, ext = os.path.splitext(fp)
#             candidates = [base + s + ext for s in ["-Bold", "_Bold", "Bold"]]
#             font_dir = os.path.dirname(fp)
#             candidates += glob.glob(os.path.join(font_dir, "*bold*" + ext))
#             for cand in candidates:
#                 if os.path.exists(cand): return cand
#         return fp

#     font_fp = get_font_path(font_path, font_weight)

#     # 计算最合适字体大小
#     if font_size:
#         best_size = font_size
#     else:
#         best_size, best_diff = 12, float('inf')
#         for size in range(int(h * 0.9), 6, -1):
#             try:
#                 f = ImageFont.truetype(font_fp, size)
#                 bbox_txt = draw.textbbox((0, 0), new_text, font=f)
#                 tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
#                 if tw <= w and th <= h:
#                     diff = abs(tw - w) + abs(th - h)
#                     if diff < best_diff:
#                         best_diff, best_size = diff, size
#                         if diff < w * 0.1:
#                             break
#             except:
#                 continue

#     # 绘制文字
#     try:
#         font = ImageFont.truetype(font_fp, best_size)
#     except:
#         font = ImageFont.load_default()
    
#     # 处理多行文本
#     lines = new_text.split('\n')
#     if len(lines) == 1:
#         # 单行文本，保持原有逻辑
#         bbox_txt = draw.textbbox((0, 0), new_text, font=font)
#         tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
#         text_x, text_y = x, y + (h - th) // 2
#         draw.text((text_x, text_y), new_text, fill=text_color, font=font)
#     else:
#         # 多行文本处理
#         if line_height is None:
#             # 默认行高为字体大小的1.2倍
#             line_height = int(best_size * 1.2)
        
#         # 计算每行的位置
#         total_height = len(lines) * line_height
#         start_y = y + (h - total_height) // 2
        
#         for i, line in enumerate(lines):
#             if line.strip():  # 跳过空行
#                 bbox_txt = draw.textbbox((0, 0), line, font=font)
#                 tw = bbox_txt[2] - bbox_txt[0]
#                 text_x = x  # 左对齐，而不是居中
#                 text_y = start_y + i * line_height
#                 draw.text((text_x, text_y), line, fill=text_color, font=font)

#     # 保存
#     img.save(output_path, quality=95)
#     print(f"Saved modified image to {output_path}")
#     # 更新返回信息
#     if len(lines) == 1:
#         bbox_txt = draw.textbbox((0, 0), new_text, font=font)
#         tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
#         return {"font_size": best_size, "text_size": (tw, th), "success": True}
#     else:
#         return {"font_size": best_size, "line_height": line_height, "lines_count": len(lines), "success": True}


def smart_replace_text(
    image_path,
    target_bbox,  # (x, y, w, h)
    new_text,
    output_path="output.jpg",
    font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    font_weight="normal",
    font_size=None,
    text_color="black",
    background_color="white",
    padding=2,
    line_height=2,
    adjust_config = {},
):
    pil_img = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = cv_img.shape[:2]

    # x, y, w, h = target_bbox
    # x0, y0 = max(0, x - padding), max(0, y - padding)
    # x1, y1 = min(orig_w, x + w + padding), min(orig_h, y + h + padding)
    
    
    x, y, _, h = target_bbox
    w = orig_w - x
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = orig_w, min(orig_h, y + h + padding)
    
    if adjust_config:
        if "y_offset" in adjust_config:
            x += adjust_config["y_offset"]
        if "x_offset" in adjust_config:
            y += adjust_config["x_offset"]

    # 加载字体
    def get_font_path(fp, weight):
        if weight == "bold":
            base, ext = os.path.splitext(fp)
            candidates = [base + s + ext for s in ["-Bold", "_Bold", "Bold"]]
            font_dir = os.path.dirname(fp)
            candidates += glob.glob(os.path.join(font_dir, "*bold*" + ext))
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return fp

    font_fp = get_font_path(font_path, font_weight)
    best_size = font_size
    font = ImageFont.truetype(font_fp, best_size)
    line_height_px = int(best_size * 1.2)

    # 分词换行
    draw = ImageDraw.Draw(pil_img)
    lines = []
    line = ""
    cur_width = 0
    for word in new_text.split():
        word_width = draw.textlength(word + " ", font=font)
        if cur_width + word_width > w:
            lines.append(line)
            line = word + " "
            cur_width = word_width
        else:
            line += word + " "
            cur_width += word_width
    if line:
        lines.append(line)

    total_needed_height = len(lines) * line_height_px
    new_h = max(h, total_needed_height)
    extend_down = max(0, y + total_needed_height - y1)

    # 重新生成 inpaint mask（如果需要扩展到底部）
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
    clear_y1 = y1 + extend_down
    if clear_y1 > orig_h:
        clear_y1 = orig_h
    mask[y0:clear_y1, x0:x1] = 255

    # 使用 inpaint 清理背景
    restored = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    img = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 填充背景为纯色（覆盖绘图区域）
    draw.rectangle([x0, y0, x1, y + total_needed_height], fill=background_color)

    # 逐行绘制文字
    for i, line in enumerate(lines):
        text_x = x
        text_y = y + i * line_height_px
        draw.text((text_x, text_y), line.strip(), font=font, fill=text_color)

    img.save(output_path, quality=95)
    return {
        "font_size": best_size,
        "lines": len(lines),
        "total_height": total_needed_height,
        "extended_bottom": extend_down,
        "success": True
    }

def detect_text_positions(image_path, min_confidence=50):
    """检测图片中所有文字的位置和置信度"""
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
    items = []
    for i, t in enumerate(data['text']):
        text = t.strip()
        if not text: continue
        try:
            conf = float(data['conf'][i])
        except:
            continue
        if conf < min_confidence: continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        items.append({'text': text, 'bbox': (x, y, w, h), 'confidence': conf})
    return items
    print(items)
    return sorted(items, key=lambda x: x['confidence'], reverse=True)


def create_visual_map(image_path, output_path="text_map.jpg"):
    """创建文字位置可视化图 - 带置信度标注"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    items = detect_text_positions(image_path)
    for idx, itm in enumerate(items):
        x, y, w, h = itm['bbox']; conf = itm['confidence']; txt = itm['text']
        color = "green" if conf>80 else ("orange" if conf>60 else "red")
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        draw.text((x, max(0,y-15)), f"{idx}:{txt}({int(conf)}%)", fill=color)
    img.save(output_path, quality=95)
    print(f"Saved visual map to {output_path}")
    return items



def replace_by_content(
    image_path,
    target_text,
    new_text,
    output_path="output.jpg",
    font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    font_weight="normal",
    font_size=None,
    text_color="black",
    background_color="white",
    max_phrase_len=45,
    similarity_threshold=0.5,
    line_height=None,
    upper_text=None,
    upper_distance=None,
    min_distance=0,
    adjust_config = {},
):
    from difflib import SequenceMatcher

    items = detect_text_positions(image_path)
    if not items:
        return {"success": False, "message": "未检测到任何文字"}

    best_match = None
    best_score = 0.0

    for i in range(len(items)):
        for j in range(i+1, min(i+max_phrase_len, len(items))+1):
            segment = items[i:j]
            phrase = ' '.join([s['text'] for s in segment])
            score = SequenceMatcher(None, target_text.lower(), phrase.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = segment

    if not best_match or best_score < similarity_threshold:
        return {
            "success": False,
            "message": f"未找到匹配 '{target_text}'，最高相似度为 {best_score:.2f}"
        }

    # 计算合并的 bbox
    x0 = min(i['bbox'][0] for i in best_match)
    y0 = min(i['bbox'][1] for i in best_match)
    x1 = max(i['bbox'][0] + i['bbox'][2] for i in best_match)
    y1 = max(i['bbox'][1] + i['bbox'][3] for i in best_match)
    w, h = x1 - x0, y1 - y0
    bbox = (x0, y0, w, h)

    if upper_text and upper_distance is not None:
        return smart_replace_with_upper_text(
            image_path, bbox, new_text, upper_text, upper_distance, output_path,
            font_path, font_weight, font_size,
            text_color, background_color, line_height
        )
    else:
        return smart_replace_text(
            image_path, bbox, new_text, output_path,
            font_path, font_weight, font_size,
            text_color, background_color, padding=2,
            line_height=line_height,
            adjust_config = adjust_config
        )




# def replace_by_content(
#     image_path,
#     target_text,
#     new_text,
#     output_path="output.jpg",
#     font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
#     font_weight="normal",
#     font_size=None,
#     text_color="black",
#     background_color="white",
#     similarity_threshold=0.6,
#     line_height=None,  # 新增：行高参数
#     upper_text=None,   # 新增：上方插入的文本
#     upper_distance=None,  # 新增：上方文本与目标文本的距离
#     min_distance=10  # 新增：最小匹配距离
# ):
#     """根据文字内容智能匹配替换，支持跨box多词匹配"""
#     items = detect_text_positions(image_path)
#     if not items:
#         raise ValueError("未检测到任何文字")
#     # 分行
#     lines = {}
#     for itm in items:
#         y = itm['bbox'][1]
#         key = min(lines.keys(), key=lambda k:abs(k-y)) if lines and min(abs(k-y) for k in lines) < min_distance else y
#         lines.setdefault(key, []).append(itm)

#     # 搜索最佳匹配
#     best, best_sim = None, 0.0
#     for line in lines.values():
#         line = sorted(line, key=lambda i:i['bbox'][0])
#         texts = [i['text'] for i in line]
#         for i in range(len(texts)):
#             for j in range(i+1, len(texts)+1):
#                 seq = ' '.join(texts[i:j])
#                 sim = difflib.SequenceMatcher(None, target_text.lower(), seq.lower()).ratio()
#                 if target_text.lower() in seq.lower(): sim = max(sim,0.8)
#                 if sim > best_sim and sim >= similarity_threshold:
#                     best, best_sim = line[i:j], sim
#     if not best:
#         print(f"未找到匹配 '{target_text}' (阈值 {similarity_threshold})")
#         return {"success": False, "message": f"未找到匹配 '{target_text}' (阈值 {similarity_threshold})"}
#     # 合并框
#     x0 = min(i['bbox'][0] for i in best); y0 = min(i['bbox'][1] for i in best)
#     x1 = max(i['bbox'][0]+i['bbox'][2] for i in best); y1 = max(i['bbox'][1]+i['bbox'][3] for i in best)
#     w, h = x1-x0, y1-y0
#     bbox = (x0, y0, w, h)
#     print(f"Match: '{' '.join(i['text'] for i in best)}' sim={best_sim:.2f} bbox={bbox}")
    
#     # 处理上方文本插入
#     if upper_text and upper_distance is not None:
#         return smart_replace_with_upper_text(
#             image_path, bbox, new_text, upper_text, upper_distance, output_path,
#             font_path, font_weight, font_size,
#             text_color, background_color, line_height
#         )
#     else:
#         return smart_replace_text(
#             image_path, bbox, new_text, output_path,
#             font_path, font_weight, font_size,
#             text_color, background_color, padding=2, line_height=line_height
#         )


def smart_replace_with_upper_text(
    image_path,
    target_bbox,  # (x, y, w, h)
    new_text,
    upper_text,
    upper_distance,
    output_path="output.jpg",
    font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    font_weight="normal",
    font_size=None,
    text_color="black",
    background_color="white",
    line_height=None,
    padding=2
):
    """智能文字替换并在上方插入浮动文本（不清除上方背景）"""
    pil_img = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 只对目标文本区域进行背景清理，不包含上方区域
    x, y, w, h = target_bbox
    
    # 扩展清理区域仅包含目标文本位置
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(cv_img.shape[1], x + w + padding), min(cv_img.shape[0], y + h + padding)
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255

    # 背景恢复（仅恢复目标文本区域）
    restored = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
    img = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # 清除原文字区域（仅目标文本区域）
    draw.rectangle([x0, y0, x1, y1], fill=background_color)

    # 字体选择支持粗体
    def get_font_path(fp, weight):
        if weight == "bold":
            base, ext = os.path.splitext(fp)
            candidates = [base + s + ext for s in ["-Bold", "_Bold", "Bold"]]
            font_dir = os.path.dirname(fp)
            candidates += glob.glob(os.path.join(font_dir, "*bold*" + ext))
            for cand in candidates:
                if os.path.exists(cand): return cand
        return fp

    font_fp = get_font_path(font_path, font_weight)

    # 计算最合适字体大小
    if font_size:
        best_size = font_size
    else:
        best_size, best_diff = 12, float('inf')
        for size in range(int(h * 0.9), 6, -1):
            try:
                f = ImageFont.truetype(font_fp, size)
                bbox_txt = draw.textbbox((0, 0), new_text, font=f)
                tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
                if tw <= w and th <= h:
                    diff = abs(tw - w) + abs(th - h)
                    if diff < best_diff:
                        best_diff, best_size = diff, size
                        if diff < w * 0.1:
                            break
            except:
                continue

    # 创建字体
    try:
        font = ImageFont.truetype(font_fp, best_size)
    except:
        font = ImageFont.load_default()
    
    # 设置默认行高
    if line_height is None:
        line_height = int(best_size * 1.2)
    
    # 1. 绘制上方浮动文本（直接在原背景上绘制，不清除背景）
    upper_lines = upper_text.split('\n')
    upper_total_height = len(upper_lines) * line_height
    upper_start_y = y - upper_distance - upper_total_height // 2
    
    for i, line in enumerate(upper_lines):
        if line.strip():
            bbox_txt = draw.textbbox((0, 0), line, font=font)
            tw = bbox_txt[2] - bbox_txt[0]
            text_x = x + (w - tw) // 2  # 水平居中
            text_y = upper_start_y + i * line_height
            # 直接在原背景上绘制文字（浮动效果）
            draw.text((text_x, text_y), line, fill=text_color, font=font)
    
    # 2. 绘制目标位置的新文本（在清除背景后的区域）
    lines = new_text.split('\n')
    if len(lines) == 1:
        # 单行文本
        bbox_txt = draw.textbbox((0, 0), new_text, font=font)
        tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
        text_x, text_y = x, y + (h - th) // 2
        draw.text((text_x, text_y), new_text, fill=text_color, font=font)
    else:
        # 多行文本处理
        total_height = len(lines) * line_height
        start_y = y + (h - total_height) // 2
        
        for i, line in enumerate(lines):
            if line.strip():
                bbox_txt = draw.textbbox((0, 0), line, font=font)
                tw = bbox_txt[2] - bbox_txt[0]
                text_x = x  # 左对齐，而不是居中
                text_y = start_y + i * line_height
                draw.text((text_x, text_y), line, fill=text_color, font=font)

    # 保存
    img.save(output_path, quality=95)
    return {
        "font_size": best_size,
        "line_height": line_height,
        "upper_text_lines": len(upper_lines),
        "main_text_lines": len(lines),
        "success": True
    }

