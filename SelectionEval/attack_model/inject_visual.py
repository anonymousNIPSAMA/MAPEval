from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualPromptInjector:
    """智能视觉提示注入器"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.default_fonts = self._find_system_fonts()
        
    def _find_system_fonts(self) -> List[str]:
        """递归查找系统所有ttf/ttc字体文件，去重，优先Noto/DejaVu等"""
        font_dirs = [
            '/usr/share/fonts', '/usr/local/share/fonts', str(Path('~/.fonts').expanduser()),
            '/System/Library/Fonts', 'C:/Windows/Fonts'
        ]
        font_paths = set()
        for font_dir in font_dirs:
            font_dir = Path(font_dir)
            if font_dir.exists():
                try:
                    for ext in ('*.ttf', '*.ttc', '*.otf'):
                        for font_file in font_dir.rglob(ext):
                            if font_file.is_file():
                                font_paths.add(str(font_file))
                except Exception:
                    continue
        # 优先Noto/DejaVu等
        sorted_fonts = sorted(font_paths, key=lambda x: (
            'noto' not in x.lower() and 'dejavu' not in x.lower(), x.lower()
        ))
        return sorted_fonts[:10]  # 限制返回数量
    
    def _get_best_font(self, text: str, font_size: int) -> ImageFont.FreeTypeFont:
        """根据文本内容智能选择最合适的字体，优先Noto/DejaVu，兼容ttc/ttf"""
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        # 优先级列表
        preferred_fonts = []
        if has_chinese:
            preferred_fonts = [
                'NotoSansCJK', 'NotoSansSC', 'NotoSans', 'msyh', 'simhei', 'simsun', 'PingFang', 'SourceHanSans'
            ]
        else:
            preferred_fonts = [
                'DejaVuSans', 'Arial', 'LiberationSans', 'FreeSans', 'NotoSans', 'Times', 'Ubuntu'
            ]
        # 先遍历优先字体
        for preferred in preferred_fonts:
            for font_path in self.default_fonts:
                if preferred.lower() in Path(font_path).name.lower():
                    try:
                        # ttc字体需指定index
                        if font_path.endswith('.ttc'):
                            font = ImageFont.truetype(font_path, font_size, index=0)
                        else:
                            font = ImageFont.truetype(font_path, font_size)
                        logger.info(f"加载字体: {font_path}")
                        return font
                    except Exception as e:
                        logger.warning(f"无法加载字体 {font_path}: {e}")
                        continue
        # fallback: 直接遍历所有可用字体
        for font_path in self.default_fonts:
            try:
                if font_path.endswith('.ttc'):
                    font = ImageFont.truetype(font_path, font_size, index=0)
                else:
                    font = ImageFont.truetype(font_path, font_size)
                logger.info(f"加载字体: {font_path}")
                return font
            except Exception:
                continue
        logger.warning("未找到合适的字体文件，使用PIL默认字体")
        return ImageFont.load_default()
    
    def _calculate_text_position(self, image_size: Tuple[int, int], text: str, 
                               font: ImageFont.FreeTypeFont, position: str = "auto") -> Tuple[int, int]:
        """智能计算文本放置位置"""
        img_width, img_height = image_size
        
        # 获取文本尺寸
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        positions = {
            "top_left": (20, 20),
            "top_right": (img_width - text_width - 20, 20),
            "bottom_left": (20, img_height - text_height - 20),
            "bottom_right": (img_width - text_width - 20, img_height - text_height - 20),
            "center": ((img_width - text_width) // 2, (img_height - text_height) // 2),
            "top_center": ((img_width - text_width) // 2, 20),
            "bottom_center": ((img_width - text_width) // 2, img_height - text_height - 20)
        }
        
        if position == "auto":
            # 自动选择较少遮挡的位置（这里简化为右上角）
            return positions["top_right"]
        
        return positions.get(position, positions["top_right"])
    
    def _analyze_image_colors(self, image: Image.Image, region: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """分析图像区域的主要颜色，用于选择对比色"""
        try:
            # 裁剪指定区域
            cropped = image.crop(region)
            # 转换为RGB模式
            cropped = cropped.convert('RGB')
            # 获取主要颜色（简化实现）
            colors = cropped.getcolors(maxcolors=256*256*256)
            if colors:
                # 获取最常见的颜色
                dominant_color = max(colors, key=lambda x: x[0])[1]
                return dominant_color
        except Exception as e:
            logger.warning(f"颜色分析失败: {e}")
        
        return (128, 128, 128)  # 默认灰色
    
    def _get_contrast_color(self, background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """根据背景色获取对比色"""
        r, g, b = background_color
        # 计算亮度
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        
        # 如果背景较亮，使用深色文字；如果背景较暗，使用浅色文字
        if brightness > 128:
            return (0, 0, 0)      # 黑色
        else:
            return (255, 255, 255)  # 白色


def inject_visual_prompt(image_path: str, text: str, x: Optional[int] = None, y: Optional[int] = None, 
                         font_path: Optional[str] = None, font_size: int = 24,
                         text_color: Optional[Tuple[int, int, int]] = None, 
                         background: Optional[Tuple[int, int, int]] = None,
                         opacity: int = 255, save_path: str = 'injected.jpg',
                         position: str = "auto", auto_contrast: bool = True,
                         add_shadow: bool = True, shadow_offset: Tuple[int, int] = (2, 2)):
    """
    智能视觉提示注入 - 在图片上添加诱导LLM agent的文字。

    参数:
    - image_path: 原始图像路径
    - text: 插入的诱导文字
    - x, y: 插入文字的起始坐标（None时自动计算）
    - font_path: 字体文件路径（None时自动选择）
    - font_size: 字体大小
    - text_color: 文本颜色 (R, G, B)（None时自动选择对比色）
    - background: 可选背景色 (R, G, B)，为 None 则透明
    - opacity: 文字图层的不透明度（0-255）
    - save_path: 输出文件路径
    - position: 文字位置 ("auto", "top_left", "top_right", "bottom_left", "bottom_right", "center")
    - auto_contrast: 是否自动选择对比色
    - add_shadow: 是否添加阴影效果
    - shadow_offset: 阴影偏移量
    """
    
    injector = VisualPromptInjector()
    
    # 验证输入
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    if not Path(image_path).suffix.lower() in injector.supported_formats:
        raise ValueError(f"不支持的图像格式，支持的格式: {injector.supported_formats}")
    
    logger.info(f"开始处理图像: {image_path}")
    
    try:
        # 打开图像
        base_image = Image.open(image_path).convert("RGBA")
        img_width, img_height = base_image.size
        logger.info(f"图像尺寸: {img_width} x {img_height}")

        # 创建同尺寸透明图层
        txt_layer = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)

        # 智能字体选择
        if font_path and Path(font_path).exists():
            try:
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"使用指定字体: {font_path}")
            except Exception as e:
                logger.warning(f"指定字体加载失败: {e}，使用自动选择")
                font = injector._get_best_font(text, font_size)
        else:
            font = injector._get_best_font(text, font_size)
            logger.info("使用自动选择的字体")

        # 智能位置计算
        if x is None or y is None:
            calculated_x, calculated_y = injector._calculate_text_position(
                (img_width, img_height), text, font, position
            )
            x = x if x is not None else calculated_x
            y = y if y is not None else calculated_y
            logger.info(f"自动计算文字位置: ({x}, {y})")

        # 获取文本尺寸（兼容Pillow新旧版本）
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 智能颜色选择
        if auto_contrast and text_color is None:
            # 分析文字区域的背景色
            text_region = (x, y, x + text_width, y + text_height)
            # 确保区域在图像范围内
            text_region = (
                max(0, text_region[0]), max(0, text_region[1]),
                min(img_width, text_region[2]), min(img_height, text_region[3])
            )
            
            if text_region[2] > text_region[0] and text_region[3] > text_region[1]:
                bg_color = injector._analyze_image_colors(base_image, text_region)
                text_color = injector._get_contrast_color(bg_color)
                logger.info(f"自动选择文字颜色: {text_color}（背景色: {bg_color}）")
            else:
                text_color = (255, 255, 255)  # 默认白色
        elif text_color is None:
            text_color = (255, 0, 0)  # 默认红色

        # 绘制文字（不再支持emoji，仅PIL）
        if add_shadow:
            shadow_x = x + shadow_offset[0]
            shadow_y = y + shadow_offset[1]
            shadow_color = (0, 0, 0, opacity // 2)
            draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
            logger.info("添加阴影效果")
        if background:
            bg_box = [x - 5, y - 5, x + text_width + 5, y + text_height + 5]
            draw.rectangle(bg_box, fill=background + (opacity,))
            logger.info(f"添加背景色: {background}")
        draw.text((x, y), text, font=font, fill=text_color + (opacity,))
        logger.info(f"绘制文字: '{text}' 在位置 ({x}, {y})（仅PIL，无emoji彩色支持）")

        # 合并原图与文字层
        combined = Image.alpha_composite(base_image, txt_layer)

        # 确保输出目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存输出
        combined.convert("RGB").save(save_path, quality=95)
        logger.info(f"提示注入成功，文件保存为：{save_path}")
        
        return str(save_path)

    except Exception as e:
        logger.error(f"视觉提示注入失败: {e}")
        raise


def batch_inject_prompts(image_dir: str, prompts: List[Dict[str, Any]], 
                        output_dir: str = "attacked_images") -> List[str]:
    """
    批量处理多个图像的视觉提示注入
    
    参数:
    - image_dir: 图像目录
    - prompts: 提示配置列表，每个包含 text, position, font_size 等参数
    - output_dir: 输出目录
    
    返回:
    - 处理后的图像路径列表
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    injector = VisualPromptInjector()
    results = []
    
    # 查找所有支持的图像文件
    image_files = []
    for ext in injector.supported_formats:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    for img_file in image_files:
        for i, prompt_config in enumerate(prompts):
            try:
                # 生成输出文件名
                output_name = f"{img_file.stem}_prompt_{i+1}{img_file.suffix}"
                output_path = output_dir / output_name
                
                # 执行注入
                result_path = inject_visual_prompt(
                    image_path=str(img_file),
                    save_path=str(output_path),
                    **prompt_config
                )
                results.append(result_path)
                
            except Exception as e:
                logger.error(f"处理 {img_file} 的提示 {i+1} 失败: {e}")
                continue
    
    logger.info(f"批量处理完成，共生成 {len(results)} 个文件")
    return results


def create_attack_variations(base_image: str, target_text: str, output_dir: str = "attack_variations") -> List[str]:
    """
    创建同一目标文本的多种攻击变体
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variations = [
        {
            "text": target_text,
            "position": "top_right",
            "font_size": 24,
            "text_color": (255, 0, 0),
            "add_shadow": True
        },
        {
            "text": target_text,
            "position": "bottom_left",
            "font_size": 32,
            "text_color": (0, 255, 0),
            "background": (255, 255, 255),
            "opacity": 200
        },
        {
            "text": f"⭐ {target_text} ⭐",
            "position": "center",
            "font_size": 28,
            "auto_contrast": True,
            "add_shadow": True
        },
        {
            "text": target_text.upper(),
            "position": "top_center",
            "font_size": 20,
            "text_color": (255, 255, 0),
            "background": (0, 0, 0),
            "opacity": 180
        }
    ]
    
    results = []
    for i, config in enumerate(variations, 1):
        try:
            output_path = output_dir / f"variation_{i}.jpg"
            result = inject_visual_prompt(
                image_path=base_image,
                save_path=str(output_path),
                **config
            )
            results.append(result)
        except Exception as e:
            logger.error(f"创建变体 {i} 失败: {e}")
    
    return results
