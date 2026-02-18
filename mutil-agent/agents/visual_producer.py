"""Agent 3: Visual Imagery Extraction and Post Production"""
import json
import os
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from openai import OpenAI
import requests
import shutil

from config.settings import settings
from utils.prompt_templates import IMAGE_GENERATION_PROMPT_TEMPLATE
from utils.image_utils import PosterGenerator
from agents.transformer import TransformerOutput
from agents.visual_strategy_selector import VisualStrategySelector, VisualStrategyResult


class SemanticExtraction(BaseModel):
    """Semantic extraction result"""
    entities: list[str] = Field(default_factory=list, description="Entity keywords")
    emotions: list[str] = Field(default_factory=list, description="Emotion keywords")
    color_palette: list[str] = Field(default_factory=list, description="Color imagery")
    visual_style: str = Field(default="", description="Visual style description")


class VisualProducerAgent:
    """Visual Imagery Extraction and Post Production"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        image_gen_model: Optional[str] = None
    ):
        """
        Initialize Agent
        
        Args:
            model_name: Model name to use for prompt generation (deepseek)
            image_gen_model: Model name to use for image generation (chatgpt)
        """
        # Use deepseek for decision making (prompt generation)
        # Validate API key
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Please set it in your .env file or environment variables.\n"
                "Example: DEEPSEEK_API_KEY=your_api_key_here"
            )
        
        # Temporarily set environment variable for ChatOpenAI validation
        original_key = os.environ.get("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key
            
            llm = ChatOpenAI(
                model=model_name or settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.8  # Higher temperature for more creativity
            )
        finally:
            # Restore original environment variable
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        
        self.llm = llm
        
        # Use chatgpt for image generation
        # Note: OpenAI API key is optional - will use placeholder if not set
        if settings.openai_api_key:
            self.image_gen_client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
        else:
            self.image_gen_client = None
            print("⚠️  OpenAI API key not set - image generation will use placeholders")
        self.strategy_selector = VisualStrategySelector()
        self.poster_generator = PosterGenerator(
            width=settings.poster_width,
            height=settings.poster_height
        )
    
    def extract_semantics(
        self,
        transformer_output: TransformerOutput,
        strategy_result: VisualStrategyResult
    ) -> SemanticExtraction:
        """
        Extract key semantic information based on selected strategy
        
        Args:
            transformer_output: Transformer Agent output
            strategy_result: Selected visual strategy
        
        Returns:
            Semantic extraction result
        """
        # Extract entities and claims grounded in the original facts
        entities: list[str] = []
        claims = [transformer_output.opposite_claims]
        
        # Use facts (5W1H) as the primary source of visual entities
        try:
            facts = transformer_output.facts
            
            # Core people / organizations
            if getattr(facts, "who", None):
                entities.extend(facts.who[:3])
            
            # Core locations
            if getattr(facts, "where", None):
                entities.extend(facts.where[:2])
            
            # Core events (trim to keep phrases short)
            if getattr(facts, "what", None):
                for item in facts.what[:2]:
                    if isinstance(item, str):
                        entities.append(item[:30])
        except Exception:
            # Fallback: leave entities empty if anything goes wrong
            pass
        
        # Determine color palette and visual style based on opposite claims
        if "negative" in transformer_output.opposite_claims.lower() or "critical" in transformer_output.opposite_claims.lower():
            color_palette = ["dark", "gray", "red", "high contrast"]
            visual_style = "dramatic, somber, high contrast"
        elif "positive" in transformer_output.opposite_claims.lower() or "celebratory" in transformer_output.opposite_claims.lower():
            color_palette = ["bright", "vibrant", "warm"]
            visual_style = "energetic, vibrant, optimistic"
        else:
            color_palette = ["muted", "neutral", "balanced"]
            visual_style = "balanced, thoughtful, nuanced"
        
        # Adjust based on strategy
        if strategy_result.selected_strategy == 3:  # Data Visualization
            visual_style += ", infographic style, data-driven"
        elif strategy_result.selected_strategy == 5:  # Typographic
            visual_style += ", typographic, text-heavy"
        
        return SemanticExtraction(
            entities=entities,
            emotions=claims,  # Using claims instead of emotions
            color_palette=color_palette,
            visual_style=visual_style
        )
    
    def generate_image_prompt(
        self,
        semantic_extraction: SemanticExtraction,
        post_text: str,
        opposite_claims: str,
        strategy_result: VisualStrategyResult,
        mirrored_article: str
    ) -> str:
        """
        Generate image generation prompt based on selected strategy
        
        Args:
            semantic_extraction: Semantic extraction result
            post_text: Post text
            opposite_claims: Opposite claims
            strategy_result: Selected visual strategy
            mirrored_article: Mirrored article content
        
        Returns:
            Image generation prompt
        """
        # Extract key summary from mirrored article (first 300 chars + key facts)
        article_summary = mirrored_article[:500] + "..." if len(mirrored_article) > 500 else mirrored_article
        
        prompt = ChatPromptTemplate.from_template(IMAGE_GENERATION_PROMPT_TEMPLATE)
        
        messages = prompt.format_messages(
            opposite_claims=opposite_claims,
            strategy_name=strategy_result.strategy_name,
            strategy_details=strategy_result.strategy_details,
            entities=", ".join(semantic_extraction.entities),
            emotions=", ".join(semantic_extraction.emotions),
            color_palette=", ".join(semantic_extraction.color_palette),
            visual_style=semantic_extraction.visual_style,
            post_text=post_text,
            mirrored_article_summary=article_summary,
            width=settings.poster_width,
            height=settings.poster_height
        )
        
        response = self.llm.invoke(messages)
        result = response.content.strip()
        
        return result
    
    def generate_image(
        self,
        image_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate image using ChatGPT (OpenAI DALL-E)
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """

        if not self.image_gen_client or not settings.openai_api_key:
            print("⚠️ OpenAI API Key not found, using placeholder")
            # Generate placeholder image
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(40, 40, 50),
                text="[Image Placeholder]\nMissing OpenAI API Key"
            )
            return placeholder_path

        try:
            print(f"📸 Calling ChatGPT (DALL-E) with prompt: {image_prompt[:50]}...")
            
            response = self.image_gen_client.images.generate(
                model=settings.image_gen_model,
                prompt=image_prompt,
                size=f"{settings.poster_width}x{settings.poster_height}",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download image
            res = requests.get(image_url, stream=True)
            if res.status_code == 200:
                with open(output_path, 'wb') as f:
                    res.raw.decode_content = True
                    shutil.copyfileobj(res.raw, f)
                return output_path
            else:
                raise Exception(f"Failed to download image: {res.status_code}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Image generation failed: {error_msg}")
            
            # Check for common error types
            if "rejected" in error_msg.lower() or "policy" in error_msg.lower() or "content" in error_msg.lower():
                print("⚠️  Request rejected by content policy - prompt may contain sensitive content")
                detailed_error = "Content Policy Violation: The image prompt may contain content that violates OpenAI's usage policies. Consider simplifying or modifying the prompt."
            elif "invalid" in error_msg.lower() or "size" in error_msg.lower():
                detailed_error = f"Image size error: {error_msg}\nDALL-E-3 supports: 1024x1024, 1792x1024, 1024x1792"
            else:
                detailed_error = error_msg[:200]  # Show more characters
            
            print("⚠️  Using placeholder image")
            
            # Generate placeholder image with full error info
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            # Split long error messages into multiple lines for better display
            error_display = detailed_error.replace('\n', '\n')[:300]  # Limit display length
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(60, 20, 20),
                text=f"[Image Gen Failed]\n{error_display}"
            )
            return placeholder_path
    
    def create_final_post(
        self,
        background_image_path: str,
        post_text: str,
        output_path: str
    ) -> str:
        """
        Create final post
        
        Args:
            background_image_path: Background image path
            post_text: Post text
            output_path: Output path
        
        Returns:
            Final post path
        """
        return self.poster_generator.create_poster(
            background_image_path=background_image_path,
            text=post_text,
            output_path=output_path,
            text_color=(255, 255, 255),
            font_size=48,
            text_position="bottom"
        )
    
    def process(
        self,
        transformer_output: TransformerOutput,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline
        
        Args:
            transformer_output: Transformer Agent output
            output_dir: Output directory
        
        Returns:
            Dictionary containing all output information
        """
        # Step 1: Select visual strategy
        print("🎯 Selecting visual strategy...")
        strategy_result = self.strategy_selector.select_strategy(transformer_output)
        print(f"   Selected Strategy {strategy_result.selected_strategy}: {strategy_result.strategy_name}")
        print(f"   Reasoning: {strategy_result.reasoning[:100]}...")
        
        # Step 2: Extract semantics based on selected strategy
        print("🎨 Extracting visual semantics...")
        semantic_extraction = self.extract_semantics(transformer_output, strategy_result)
        
        # Step 3: Generate image prompt based on strategy
        print("📝 Generating image generation prompt...")
        image_prompt = self.generate_image_prompt(
            semantic_extraction,
            transformer_output.post_text,
            transformer_output.opposite_claims,
            strategy_result,
            transformer_output.mirrored_article
        )
        
        # Step 4: Generate image using ChatGPT
        print("🖼️  Generating image with ChatGPT (DALL-E)...")
        os.makedirs(output_dir, exist_ok=True)
        background_image_path = os.path.join(output_dir, "background.png")
        background_image_path = self.generate_image(image_prompt, background_image_path)
        
        # Step 5: Create final post
        print("🎬 Creating final post...")
        final_post_path = os.path.join(output_dir, "final_post.png")
        final_post_path = self.create_final_post(
            background_image_path,
            transformer_output.post_text,
            final_post_path
        )
        
        return {
            "strategy_result": strategy_result,
            "semantic_extraction": semantic_extraction,
            "image_prompt": image_prompt,
            "background_image_path": background_image_path,
            "final_post_path": final_post_path
        }

