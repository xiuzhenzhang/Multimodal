"""Local visual helper used by SynthesisAgent with Hugging Face diffusion backends."""
import json
import os
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import torch
from PIL import Image

from generation.runtime import settings
from generation.utils.prompt_templates import IMAGE_GENERATION_PROMPT_TEMPLATE
from generation.utils.image_utils import PosterGenerator
from generation.agents.transformer import TransformerOutput
from generation.agents.visual_strategy_selector import VisualStrategySelector, VisualStrategyResult


class LocalImageBackendError(RuntimeError):
    """Raised when the local Hugging Face image backend is unavailable."""


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
        generation_lock: Optional[Any] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Agent
        
        Args:
            model_name: Model name to use for prompt generation (deepseek)
            image_gen_model: Model name to use for image generation
            generation_lock: Optional threading.Lock for thread-safe SD generation
        """
        # Use the synthesis runtime model for image reasoning/prompt generation.
        runtime_api_key = api_key or settings.synthesis_api_key or settings.deepseek_api_key
        runtime_base_url = base_url or settings.synthesis_base_url or settings.deepseek_base_url

        if not runtime_api_key:
            raise ValueError(
                "A synthesis runtime API key is not set. Please configure SYNTHESIS_API_KEY "
                "or the provider-specific API key in your environment."
            )
        
        # Temporarily set environment variable for ChatOpenAI validation
        original_key = os.environ.get("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = runtime_api_key
            
            llm = ChatOpenAI(
                model=model_name or settings.synthesis_model or settings.deepseek_model,
                api_key=runtime_api_key,
                base_url=runtime_base_url,
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
        if self.image_gen_provider != "sd_local":
            raise ValueError("VisualProducerAgent supports only the local sd_local image backend.")
        # Local Hugging Face diffusion backend - lazy loading to save memory.
        print(f"✅ Using local HF image model: {settings.sd_model_path}")
        print(f"   Device: {settings.sd_device}, Dtype: {settings.sd_dtype}")
        print(f"   Quantization enabled: {settings.sd_enable_quantization} ({settings.sd_quantization_mode})")
        if settings.sd_enable_quantization:
            print(f"   Quantized runtime repo: {settings.sd_quantized_model_path}")
        print(f"   CPU offload enabled: {settings.sd_enable_cpu_offload}")
        print(f"   Steps: {settings.sd_num_inference_steps}, Guidance: {settings.sd_guidance_scale}")
        print(f"   Reference images enabled: {settings.sd_use_reference_images}")
        print("   Model will be loaded on first use")

        self.strategy_selector = VisualStrategySelector(
            model_name=model_name,
            api_key=runtime_api_key,
            base_url=runtime_base_url
        )
        self.poster_generator = PosterGenerator(
            width=settings.poster_width,
            height=settings.poster_height
        )

    def _format_diffusers_import_error(self, exc: Exception) -> str:
        message = str(exc)
        if "No module named 'diffusers'" in message:
            return (
                "diffusers is not installed for the local FLUX backend.\n"
                "   Install the local image stack in your server environment:\n"
                "   pip install diffusers transformers accelerate safetensors sentencepiece"
            )
        return (
            f"Failed to import the required diffusers pipeline components: {message}\n"
            "   Your diffusers package may be missing FLUX support.\n"
            "   Try: pip install --upgrade diffusers transformers accelerate safetensors sentencepiece"
        )
    
    def _load_sd_pipeline(self):
        """Lazy load the local Hugging Face image pipeline with thread-safe singleton pattern."""
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

    def _is_flux_runtime(self) -> bool:
        return "flux.2" in settings.sd_model_path.lower() or "flux.2" in settings.sd_quantized_model_path.lower()

    def _should_use_quantized_flux(self) -> bool:
        return self._is_flux_runtime() and settings.sd_enable_quantization and settings.sd_quantization_mode == "bnb_4bit"

    def _load_quantized_flux2_pipeline(self, target_device: torch.device, dtype):
        try:
            from diffusers import Flux2Pipeline, Flux2Transformer2DModel
            from transformers import Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise LocalImageBackendError(
                "Quantized FLUX.2 loading requires recent diffusers + transformers.\n"
                "   Install/upgrade: pip install -U diffusers transformers bitsandbytes accelerate safetensors"
            ) from exc

        runtime_repo = settings.sd_quantized_model_path or settings.sd_model_path
        print(f"   Using 4-bit quantized FLUX.2 runtime repo: {runtime_repo}")
        print("   Loading quantized transformer and text encoder on CPU first...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            runtime_repo,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="cpu",
            local_files_only=settings.sd_local_files_only,
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            runtime_repo,
            subfolder="text_encoder",
            torch_dtype=dtype,
            device_map="cpu",
            local_files_only=settings.sd_local_files_only,
        )
        pipe = Flux2Pipeline.from_pretrained(
            runtime_repo,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=dtype,
            local_files_only=settings.sd_local_files_only,
        )

        if target_device.type == "cuda" and settings.sd_enable_cpu_offload:
            print("   Enabling model CPU offload for single-GPU memory savings...")
            pipe.enable_model_cpu_offload()
        else:
            print(f"   Moving quantized pipeline to {target_device}...")
            pipe = pipe.to(target_device)

        return pipe
    
    def _do_load_sd_pipeline(self):
        """Actually load the local image pipeline (called within lock if multi-threaded)."""
        if self.sd_pipeline is not None:
            return self.sd_pipeline
        
        print(f"🔄 Loading local Hugging Face image model: {settings.sd_model_path}")
        print(f"   This may take a few minutes on first run...")
        
        try:
            # Determine dtype
            if settings.sd_dtype == "float16":
                dtype = torch.float16
            elif settings.sd_dtype == "bfloat16":
                dtype = torch.bfloat16
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
            
            # Resolve pipeline class from the configured local model path.
            model_path_lower = settings.sd_model_path.lower()
            if self._should_use_quantized_flux():
                print("   Single-GPU quantized FLUX mode enabled.")
                self.sd_pipeline = self._load_quantized_flux2_pipeline(target_device, dtype)
                self.sd_device = target_device
            elif "flux.2" in model_path_lower:
                from diffusers import Flux2Pipeline

                pipeline_class = Flux2Pipeline
                print(f"   Using Flux2Pipeline")
                print(f"   Loading model to {target_device}...")
                print(f"   Loading pipeline directly to {target_device}...")
                self.sd_pipeline = pipeline_class.from_pretrained(
                    settings.sd_model_path,
                    torch_dtype=dtype,
                    local_files_only=settings.sd_local_files_only,
                ).to(target_device)
                self.sd_device = target_device
            elif "flux.1" in model_path_lower or "/flux" in model_path_lower:
                from diffusers import FluxPipeline

                pipeline_class = FluxPipeline
                print(f"   Using FluxPipeline")
                print(f"   Loading model to {target_device}...")
                print(f"   Loading pipeline directly to {target_device}...")
                self.sd_pipeline = pipeline_class.from_pretrained(
                    settings.sd_model_path,
                    torch_dtype=dtype,
                    local_files_only=settings.sd_local_files_only,
                ).to(target_device)
                self.sd_device = target_device
            else:
                try:
                    from diffusers import StableDiffusion3Pipeline

                    pipeline_class = StableDiffusion3Pipeline
                    print(f"   Using StableDiffusion3Pipeline")
                except ImportError:
                    # Fallback to other diffusers pipelines when FLUX-specific classes are unavailable.
                    print(f"   ⚠️  StableDiffusion3Pipeline not available")
                    print(f"   Trying fallback pipelines...")

                    if 'xl' in model_path_lower or 'sdxl' in model_path_lower:
                        from diffusers import StableDiffusionXLPipeline

                        pipeline_class = StableDiffusionXLPipeline
                        print(f"   Using StableDiffusionXLPipeline (SDXL)")
                    else:
                        from diffusers import StableDiffusionPipeline

                        pipeline_class = StableDiffusionPipeline
                        print(f"   Using StableDiffusionPipeline (SD 2.1 or earlier)")

                print(f"   Loading model to {target_device}...")
                print(f"   Loading pipeline directly to {target_device}...")
                self.sd_pipeline = pipeline_class.from_pretrained(
                    settings.sd_model_path,
                    torch_dtype=dtype,
                    local_files_only=settings.sd_local_files_only,
                ).to(target_device)
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
            
            print(f"✅ Local Hugging Face image pipeline ready!")
            
            return self.sd_pipeline
            
        except ImportError as e:
            error_msg = self._format_diffusers_import_error(e)
            print(f"❌ {error_msg}")
            raise LocalImageBackendError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load local Hugging Face image model: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise LocalImageBackendError(error_msg) from e
    
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

        # FLUX does not require the old CLIP-era 77 token truncation.
        # Keep the full prompt and only normalize excessive whitespace.
        normalized_lines = [line.strip() for line in result.splitlines() if line.strip()]
        normalized_prompt = " ".join(normalized_lines)

        if len(normalized_prompt) > 4000:
            print(f"   ⚠️  Prompt is very long ({len(normalized_prompt)} chars), keeping full content for FLUX.")

        return normalized_prompt
    
    def _generate_image_sd_local(
        self,
        image_prompt: str,
        output_path: str,
        reference_images: Optional[list[str]] = None
    ) -> str:
        """
        Generate image using the local Hugging Face image backend.
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """
        try:
            print(f"📸 Generating image with local Hugging Face model...")
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

            reference_payload = self._load_reference_images(reference_images)
            if reference_payload:
                print(f"   Using {len(reference_payload)} reference image(s) from the source article")

            pipeline_kwargs = {
                "prompt": image_prompt,
                "num_inference_steps": settings.sd_num_inference_steps,
                "guidance_scale": settings.sd_guidance_scale,
                "width": settings.poster_width,
                "height": settings.poster_height,
            }
            if reference_payload and "flux" in settings.sd_model_path.lower():
                pipeline_kwargs["image"] = reference_payload
            
            # Use lock if provided (for multi-threaded environments)
            if self.generation_lock is not None:
                print(f"   🔒 Acquiring generation lock for thread safety...")
                with self.generation_lock:
                    print(f"   🔓 Lock acquired, generating...")
                    
                    with torch.no_grad():
                        result = pipeline(**pipeline_kwargs)
            else:
                # No lock needed (single-threaded)
                with torch.no_grad():
                    result = pipeline(**pipeline_kwargs)
            
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
            
        except LocalImageBackendError:
            raise
        except Exception as e:
            error_msg = f"Local FLUX image generation failed: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise LocalImageBackendError(error_msg) from e

    def _load_reference_images(self, reference_images: Optional[list[str]]) -> list[Image.Image]:
        if not settings.sd_use_reference_images or not reference_images:
            return []

        loaded_images: list[Image.Image] = []
        for image_path in reference_images[: settings.sd_reference_max_images]:
            if not image_path or not os.path.exists(image_path):
                continue
            try:
                with Image.open(image_path) as image:
                    loaded_images.append(
                        image.convert("RGB").resize((settings.poster_width, settings.poster_height))
                    )
            except Exception as exc:
                print(f"⚠️  Skipping reference image {image_path}: {exc}")

        return loaded_images
    
    def generate_image(
        self,
        image_prompt: str,
        output_path: str,
        reference_images: Optional[list[str]] = None
    ) -> str:
        """
        Generate image using configured provider
        
        Args:
            image_prompt: Image generation prompt
            output_path: Output path
        
        Returns:
            Generated image path
        """
        return self._generate_image_sd_local(
            image_prompt,
            output_path,
            reference_images=reference_images,
        )
    
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
        provider_name = f"Local Hugging Face ({settings.sd_model_path})"
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
