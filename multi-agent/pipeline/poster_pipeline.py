"""Post generation Pipeline"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from agents.transformer import TransformerAgent
from agents.sentinel import SentinelAgent, CriticResult
from config.settings import settings

# Import appropriate visual producer based on configuration
if settings.image_gen_provider == "sd_local":
    from agents.visual_producer_sd_local import VisualProducerAgent
else:
    from agents.visual_producer import VisualProducerAgent


class PosterPipeline:
    """Mirror News Post Generation Pipeline"""
    
    def __init__(
        self,
        max_retries: int = 3,
        output_dir: Optional[str] = None,
        sd_generation_lock: Optional[Any] = None
    ):
        """
        Initialize Pipeline
        
        Args:
            max_retries: Maximum retry count for Agent 1
            output_dir: Output directory, defaults to directory in settings
            sd_generation_lock: Optional lock for thread-safe SD generation
        """
        self.max_retries = max_retries
        self.output_dir = output_dir or settings.output_dir
        
        # Initialize agents
        self.transformer = TransformerAgent()
        self.sentinel = SentinelAgent()
        
        # Pass lock to visual producer if using SD local
        if settings.image_gen_provider == "sd_local":
            self.visual_producer = VisualProducerAgent(generation_lock=sd_generation_lock)
        else:
            self.visual_producer = VisualProducerAgent()
    
    def process(
        self,
        news_article: str,
        save_intermediate: bool = True,
        post_method: int = 1,
        article_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process news article and generate post
        
        Args:
            news_article: Original news article
            save_intermediate: Whether to save intermediate results
            post_method: Post text generation method (1: modified facts, 2: modified evidence)
            article_id: Optional article ID to use as output folder name (instead of timestamp)
        
        Returns:
            Dictionary containing all processing results
        """
        print("=" * 60)
        print("🚀 Starting news post generation pipeline")
        print("=" * 60)
        
        # Create output directory
        if article_id:
            # Use article_id as folder name
            output_dir = os.path.join(self.output_dir, article_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # Use timestamp as folder name (backward compatibility)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"post_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: Stance reversal and text refinement (Agent 1)
        print("\n" + "=" * 60)
        print("Stage 1: Stance Reversal and Text Refinement (Agent 1)")
        print("=" * 60)
        
        transformer_output = None
        critic_result = None
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Agent 1 processing (only on first attempt, or if transformer_output is None)
                if transformer_output is None:
                    transformer_output = self.transformer.process(news_article, post_method=post_method)
                
                # Stage 2: Critical evaluation (Agent 2 - Critic)
                print("\n" + "=" * 60)
                print("Stage 2: Critical Evaluation (Agent 2 - Critic)")
                print("=" * 60)
                
                critic_result = self.sentinel.critique(transformer_output, original_article=news_article)
                
                print(f"📊 Quality Score: {critic_result.score}/100")
                print(f"✅ Evaluation: {'Excellent - Proceeding' if critic_result.is_excellent else 'Needs Improvement'}")
                
                if critic_result.strengths:
                    print(f"✨ Strengths: {', '.join(critic_result.strengths[:3])}")
                if critic_result.criticisms:
                    print(f"⚠️  Criticisms: {', '.join(critic_result.criticisms[:3])}")
                if critic_result.must_improve:
                    print(f"🔴 Must Improve: {', '.join(critic_result.must_improve)}")
                if critic_result.recommendations:
                    print(f"💡 Recommendations: {', '.join(critic_result.recommendations[:3])}")
                
                # Check if rollback is needed
                if not self.sentinel.should_rollback(critic_result):
                    print("\n✅ Content passed critical evaluation! Proceeding to visual generation...")
                    break
                else:
                    retry_count += 1
                    print(f"\n🔄 Content needs improvement, adjusting based on feedback (Attempt {retry_count}/{self.max_retries})...")
                    if critic_result.must_improve:
                        print(f"   Focus areas: {', '.join(critic_result.must_improve)}")
                    
                    # Use adjust method instead of complete regeneration
                    print("🔧 Applying targeted adjustments based on critic feedback...")
                    transformer_output = self.transformer.adjust(transformer_output, critic_result)
                    
            except Exception as e:
                print(f"❌ Error occurred during processing: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise
                # On error, regenerate from scratch
                transformer_output = None
        
        if transformer_output is None:
            raise RuntimeError("Unable to generate valid transformation result")
        
        # Save intermediate results
        if save_intermediate:
            intermediate_path = os.path.join(output_dir, "intermediate_results.json")
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump({
                    "original_article": news_article,
                    "facts": transformer_output.facts.model_dump(),
                    "mirrored_article": transformer_output.mirrored_article,
                    "post_text": transformer_output.post_text,
                    "opposite_claims": transformer_output.opposite_claims,
                    "critic_result": critic_result.model_dump() if critic_result else None
                }, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Intermediate results saved to: {intermediate_path}")
        
        # Stage 3: Visual strategy selection and image generation (Agent 3)
        print("\n" + "=" * 60)
        print("Stage 3: Visual Strategy Selection and Image Generation (Agent 3)")
        print("=" * 60)
        
        visual_result = self.visual_producer.process(transformer_output, output_dir)
        
        # Generate final report
        final_result = {
            "timestamp": timestamp,
            "output_dir": output_dir,
            "original_article": news_article,
            "facts": transformer_output.facts.model_dump(),
            "mirrored_article": transformer_output.mirrored_article,
            "post_text": transformer_output.post_text,
            "opposite_claims": transformer_output.opposite_claims,
            "critic_result": critic_result.model_dump() if critic_result else None,
            "strategy_result": visual_result["strategy_result"].model_dump(),
            "semantic_extraction": visual_result["semantic_extraction"].model_dump(),
            "image_prompt": visual_result["image_prompt"],
            "background_image_path": visual_result["background_image_path"],
            "final_post_path": visual_result["final_post_path"]
        }
        
        # Save final report
        report_path = os.path.join(output_dir, "final_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Processing completed!")
        print("=" * 60)
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Final post: {final_result['final_post_path']}")
        print(f"🎯 Selected Strategy: {visual_result['strategy_result'].strategy_name}")
        print(f"📊 Complete report: {report_path}")
        print("=" * 60)
        
        return final_result


def main():
    """Example usage"""
    # Sample news
    sample_news = """
    A tech company today released its financial report, showing significant growth in performance,
    with revenue up 50% year-over-year and net profit up 80%. The company's CEO stated that
    this achievement is due to the hard work of all employees and the company's innovation strategy.
    The company plans to continue expanding in the next quarter and expects to hire 500 new employees.
    """
    
    pipeline = PosterPipeline()
    result = pipeline.process(sample_news)
    
    print("\nGenerated post text:")
    print(result["post_text"])
    print(f"\nSelected visual strategy: {result['strategy_result']['strategy_name']}")


if __name__ == "__main__":
    main()

