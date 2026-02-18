"""Image processing utilities"""
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
import os


class PosterGenerator:
    """Poster generator"""
    
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
    
    def create_poster(
        self,
        background_image_path: str,
        text: str,
        output_path: str,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_size: int = 48,
        text_position: str = "bottom"  # "top", "center", "bottom"
    ) -> str:
        """
        Overlay text on background image to generate final poster
        
        Args:
            background_image_path: Background image path
            text: Text to overlay
            output_path: Output path
            text_color: Text color RGB
            font_size: Font size
            text_position: Text position
        
        Returns:
            Output file path
        """
        # Open background image
        try:
            bg_image = Image.open(background_image_path)
            bg_image = bg_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        except Exception as e:
            # If image doesn't exist, create a solid color background
            bg_image = Image.new('RGB', (self.width, self.height), color=(30, 30, 30))
        
        # Create drawable object
        draw = ImageDraw.Draw(bg_image)
        
        # Try to load font, use default if fails
        try:
            # Windows system font paths
            font_path = "C:/Windows/Fonts/msyh.ttc"  # Microsoft YaHei
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/simhei.ttf"  # SimHei
            font = ImageFont.truetype(font_path, font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate text position
        if font:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Estimate text size
            text_width = len(text) * font_size // 2
            text_height = font_size
        
        # Calculate text position (centered, considering position parameter)
        x = (self.width - text_width) // 2
        
        if text_position == "top":
            y = 50
        elif text_position == "center":
            y = (self.height - text_height) // 2
        else:  # bottom
            y = self.height - text_height - 50
        
        # Draw text shadow (enhance readability)
        shadow_offset = 2
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            text,
            fill=(0, 0, 0),
            font=font
        )
        
        # Draw text
        draw.text(
            (x, y),
            text,
            fill=text_color,
            font=font
        )
        
        # Save image
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        bg_image.save(output_path, quality=95)
        
        return output_path
    
    def generate_placeholder_image(
        self,
        output_path: str,
        color: Tuple[int, int, int] = (30, 30, 30),
        text: Optional[str] = None
    ) -> str:
        """
        Generate placeholder image (used when image generation API is unavailable)
        
        Args:
            output_path: Output path
            color: Background color
            text: Optional text
        """
        image = Image.new('RGB', (self.width, self.height), color=color)
        
        if text:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
            except:
                font = None
            
            bbox = draw.textbbox((0, 0), text, font=font) if font else None
            if bbox:
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (self.width - text_width) // 2
                y = (self.height - text_height) // 2
                draw.text((x, y), text, fill=(200, 200, 200), font=font)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        image.save(output_path)
        
        return output_path

