"""Agent 3: Visual Imagery Extraction and Post Production - Local SD Version"""
import json
import os
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from openai import OpenAI
import requests
import shutil
import torch
from PIL import Image

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
        image_gen_model: Optional[str] = None,
        generation_lock: Optional[Any] = None
    ):
        """
        Initialize Agent
        
        Args:
            model_name: Model name to use for prompt generation (deepseek)
            image_gen_model: Model name to use for image generation
            generation_lock: Optional threading.Lock for thread-safe SD generation
        """
        # Use deepseek for decision making (prompt generation)
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
        
        # Store generation lock for thread safety
        self.generation_lock = generation_lock
        
        # Initialize image generation based on provider
        self.image_gen_provider = settings.image_gen_provider
        self.sd_pipeline = None  # Will be loaded on first use (lazy loading)
        self.image_gen_client = None
        
        if self.image_gen_provider == "sd_local":
            # Local Stable Diffusion - lazy loading to save memory
            print(f"✅ Using Local Stable Diffusion: {settings.sd_model_path}")
            print(f"   Device: {settings.sd_device}, Dtype: {settings.sd_dtype}")
            print(f"   Steps: {settings.sd_num_inference_steps}, Guidance: {settings.sd_guidance_scale}")
            print(f"   Model will be loaded on first use")
        elif self.image_gen_provider == "openai":
            # OpenAI DALL-E
            if settings.openai_api_key:
                self.image_gen_client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url
                )
                print("✅ Using OpenAI DALL-E for image generation")
            else:
                print("⚠️  OpenAI API key not set - image generation will use placeholders")
        else:
            print(f"⚠️  Unknown provider: {self.image_gen_provider}")
        
        self.strategy_selector = VisualStrategySelector()
        self.poster_generator = PosterGenerator(
            width=settings.poster_width,
            height=settings.poster_height
        )
    
    def _load_sd_pipeline(self):
        """Lazy load Stable Diffusion pipeline with thread-safe singleton pattern"""
        if self.sd_pipeline is not None:
            return self.sd_pipeline
        
        # Use lock to ensure only one thread loads the model
        if self.generation_lock is not None:
            with self.generation_lock:
                # Double-check after acquiring lock
                if self.sd_pipeline is not None:
                    return self.sd_pipeline
                return self._do_load_sd_pipeline()
        else:
            return self._do_load_sd_pipeline()
    
    def _do_load_sd_pipeline(self):
        """Actually load the SD pipeline (called within lock if multi-threaded)"""
        if self.sd_pipeline is not None:
            return self.sd_pipeline
        
        print(f"🔄 Loading Stable Diffusion model: {settings.sd_model_path}")
        print(f"   This may take a few minutes on first run...")
        
        try:
            # Determine dtype
            if settings.sd_dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            print(f"   Loading with dtype: {dtype}")
            
            # Determine target device first
            if settings.sd_device == "cuda" and torch.cuda.is_available():
                target_device = torch.device("cuda")
                print(f"   Target device: CUDA (GPU)")
            else:
                target_device = torch.device("cpu")
                print(f"   Target device: CPU")
            
            # Try to import SD3 pipeline first
            try:
                from diffusers import StableDiffusion3Pipeline
                pipeline_class = StableDiffusion3Pipeline
                print(f"   Using StableDiffusion3Pipeline")
            except ImportError:
                # Fallback to SDXL or SD2.1
                print(f"   ⚠️  StableDiffusion3Pipeline not available")
                print(f"   Trying fallback pipelines...")
                
                # Check model path to determine which pipeline to use
                model_path_lower = settings.sd_model_path.lower()
                
                if 'xl' in model_path_lower or 'sdxl' in model_path_lower:
                    from diffusers import StableDiffusionXLPipeline
                    pipeline_class = StableDiffusionXLPipeline
                    print(f"   Using StableDiffusionXLPipeline (SDXL)")
                else:
                    from diffusers import StableDiffusionPipeline
                    pipeline_class = StableDiffusionPipeline
                    print(f"   Using StableDiffusionPipeline (SD 2.1 or earlier)")
            
            # Load pipeline directly to target device
            print(f"   Loading model to {target_device}...")
            
            # Load pipeline directly to target device
            # This is the working approach from yesterday
            print(f"   Loading pipeline directly to {target_device}...")
            self.sd_pipeline = pipeline_class.from_pretrained(
                settings.sd_model_path,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(target_device)
            
            # Store device for later use
            self.sd_device = target_device
            
            if target_device.type == "cuda":
                print(f"✅ Model loaded on CUDA (GPU)")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"✅ Model loaded on CPU (slower)")
            
            # Enable memory optimizations
            if hasattr(self.sd_pipeline, "enable_attention_slicing"):
                self.sd_pipeline.enable_attention_slicing()
                print(f"   ✅ Attention slicing enabled")
            if hasattr(self.sd_pipeline, "enable_vae_slicing"):
                self.sd_pipeline.enable_vae_slicing()
                print(f"   ✅ VAE slicing enabled")
            
            print(f"✅ Stable Diffusion pipeline ready!")
            
            return self.sd_pipeline
            
        except ImportError as e:
            error_msg = f"Failed to import diffusers: {e}\n"
            error_msg += "\n💡 Your diffusers version is too old for Stable Diffusion 3.5"
            error_msg += "\n   Please upgrade: pip install --upgrade diffusers"
            error_msg += "\n   Or run: python fix_diffusers_version.py"
            print(f"❌ {error_msg}")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load Stable Diffusion model: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
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
            Image generation prompt (truncated to fit CLIP's 77 token limit)
        """
        # Extract key summary from mirrored article (first 500 chars)
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
        
        # Truncate to avoid CLIP's 77 token limit
        # Rough estimate: 1 token ≈ 4 characters, so 77 tokens ≈ 300 characters
        # Use conservative limit of 250 characters to be safe
        if len(result) > 250:
            print(f"   ⚠️  Prompt too long ({len(result)} chars), truncating to 250 chars")
            result = result[:247] + "..."
        
        return result
    
    def _generate_image_sd_local(
        self,
        image_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate image using local Stable Diffusion
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """
        try:
            print(f"📸 Generating image with Local Stable Diffusion...")
            print(f"   Prompt: {image_prompt[:100]}...")
            
            # Load pipeline (lazy loading)
            pipeline = self._load_sd_pipeline()
            
            # Get device from stored attribute
            device = self.sd_device if hasattr(self, 'sd_device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Generate image with lock protection for thread safety
            print(f"   Generating with {settings.sd_num_inference_steps} steps...")
            print(f"   Size: {settings.poster_width}x{settings.poster_height}")
            print(f"   Guidance scale: {settings.sd_guidance_scale}")
            print(f"   Device: {device}")
            
            # Use lock if provided (for multi-threaded environments)
            if self.generation_lock is not None:
                print(f"   🔒 Acquiring generation lock for thread safety...")
                with self.generation_lock:
                    print(f"   🔓 Lock acquired, generating...")
                    
                    with torch.no_grad():
                        result = pipeline(
                            prompt=image_prompt,
                            num_inference_steps=settings.sd_num_inference_steps,
                            guidance_scale=settings.sd_guidance_scale,
                            width=settings.poster_width,
                            height=settings.poster_height,
                        )
            else:
                # No lock needed (single-threaded)
                with torch.no_grad():
                    result = pipeline(
                        prompt=image_prompt,
                        num_inference_steps=settings.sd_num_inference_steps,
                        guidance_scale=settings.sd_guidance_scale,
                        width=settings.poster_width,
                        height=settings.poster_height,
                    )
            
            # Save image
            image = result.images[0]
            image.save(output_path)
            
            # Verify image quality
            import numpy as np
            img_array = np.array(image)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            
            print(f"✅ Image generated successfully: {output_path}")
            print(f"   Size: {image.size}, Mode: {image.mode}")
            print(f"   Unique colors: {unique_colors}")
            
            if unique_colors < 100:
                print(f"   ⚠️  WARNING: Image has very few colors ({unique_colors}), may be corrupted!")
                print(f"   This could indicate a model loading or generation issue.")
            
            return output_path
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Local SD image generation failed: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Generate placeholder image
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            error_display = error_msg[:300]
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(60, 20, 20),
                text=f"[SD Local Gen Failed]\n{error_display}"
            )
            return placeholder_path
    
    def _generate_image_openai(
        self,
        image_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate image using OpenAI DALL-E
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """
        if not self.image_gen_client or not settings.openai_api_key:
            print("⚠️ OpenAI API Key not found, using placeholder")
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(40, 40, 50),
                text="[Image Placeholder]\nMissing OpenAI API Key"
            )
            return placeholder_path
        
        try:
            print(f"📸 Calling OpenAI DALL-E with prompt: {image_prompt[:50]}...")
            
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
            print(f"❌ OpenAI image generation failed: {error_msg}")
            
            # Generate placeholder image
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            error_display = error_msg[:200]
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(60, 20, 20),
                text=f"[OpenAI Image Gen Failed]\n{error_display}"
            )
            return placeholder_path
    
    def generate_image(
        self,
        image_prompt: str,
        output_path: str
    ) -> str:
        """
        Generate image using configured provider
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """
        if self.image_gen_provider == "sd_local":
            return self._generate_image_sd_local(image_prompt, output_path)
        elif self.image_gen_provider == "openai":
            return self._generate_image_openai(image_prompt, output_path)
        else:
            print(f"⚠️ Unknown provider: {self.image_gen_provider}, using placeholder")
            placeholder_path = output_path.replace(".png", "_placeholder.png")
            self.poster_generator.generate_placeholder_image(
                placeholder_path,
                color=(40, 40, 50),
                text=f"[Unknown Provider]\n{self.image_gen_provider}"
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
        
        # Step 4: Generate image using configured provider
        provider_name = "Local Stable Diffusion" if self.image_gen_provider == "sd_local" else "OpenAI DALL-E"
        print(f"🖼️  Generating image with {provider_name}...")
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
