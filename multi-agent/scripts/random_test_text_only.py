"""Random news article test script (Text only - no image generation)"""
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from agents.transformer import TransformerAgent
from agents.sentinel import SentinelAgent


def process_text_only(news_article: str, output_dir: str, article_id: str = None, post_method: int = None, max_retries: int = 3):
    """
    Process news article using only Agent 1 (Transformer) and Agent 2 (Sentinel)
    No image generation (Agent 3)
    
    Args:
        news_article: Original news article
        output_dir: Output directory for results
        article_id: Article ID for naming output directory
        post_method: Post text generation method (1: modified facts, 2: modified evidence). If None, uses settings.post_method
        max_retries: Maximum retry count
    
    Returns:
        Dictionary containing processing results
    """
    from config.settings import settings
    if post_method is None:
        post_method = settings.post_method
    # Initialize agents
    transformer = TransformerAgent()
    sentinel = SentinelAgent()
    
    transformer_output = None
    critic_result = None
    retry_count = 0
    
    # Stage 1: Stance reversal and text refinement (Agent 1)
    print("\n" + "=" * 60)
    print("Stage 1: Stance Reversal and Text Refinement (Agent 1)")
    print("=" * 60)
    
    while retry_count < max_retries:
        try:
            # Agent 1 processing (only on first attempt, or if transformer_output is None)
            if transformer_output is None:
                transformer_output = transformer.process(news_article, post_method=post_method)
            
            # Stage 2: Critical evaluation (Agent 2 - Critic)
            print("\n" + "=" * 60)
            print("Stage 2: Critical Evaluation (Agent 2 - Critic)")
            print("=" * 60)
            
            critic_result = sentinel.critique(transformer_output, original_article=news_article)
            
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
            if not sentinel.should_rollback(critic_result):
                print("\n✅ Content passed critical evaluation!")
                break
            else:
                retry_count += 1
                print(f"\n🔄 Content needs improvement, adjusting based on feedback (Attempt {retry_count}/{max_retries})...")
                if critic_result.must_improve:
                    print(f"   Focus areas: {', '.join(critic_result.must_improve)}")
                
                # Use adjust method instead of complete regeneration
                print("🔧 Applying targeted adjustments based on critic feedback...")
                transformer_output = transformer.adjust(transformer_output, critic_result)
                
        except Exception as e:
            print(f"❌ Error occurred during processing: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise
            # On error, regenerate from scratch
            transformer_output = None
    
    if transformer_output is None:
        raise RuntimeError("Unable to generate valid transformation result")
    
    # Save intermediate results
    os.makedirs(output_dir, exist_ok=True)
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
    
    # Generate final report (text only, no image)
    final_result = {
        "output_dir": output_dir,
        "original_article": news_article,
        "facts": transformer_output.facts.model_dump(),
        "mirrored_article": transformer_output.mirrored_article,
        "post_text": transformer_output.post_text,
        "opposite_claims": transformer_output.opposite_claims,
        "critic_result": critic_result.model_dump() if critic_result else None,
        "post_method": post_method,
        "note": "Text-only processing (no image generation)"
    }
    
    # Save final report
    report_path = os.path.join(output_dir, "final_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Complete report: {report_path}")
    print("=" * 60)
    
    return final_result


def main():
    """Test pipeline with a random news article (text only)"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Random news article test (text only)")
    parser.add_argument(
        "--post-method",
        type=int,
        choices=[1, 2],
        default=None,
        help="Post text generation method: 1=modified facts, 2=modified evidence (default: from settings)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: sample_data)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Random News Article Test (Text Only - No Image Generation)")
    print("=" * 80)
    
    # Initialize news loader
    print("\n📚 Loading news dataset...")
    loader = NewsLoader()
    
    article_count = loader.get_article_count()
    print(f"   Found {article_count} articles in dataset")
    
    if article_count == 0:
        print("❌ No articles found in dataset!")
        return
    
    # Load random article
    print("\n🎲 Selecting random article...")
    article_data = loader.load_random_article()
    
    if not article_data:
        print("❌ Failed to load article!")
        return
    
    print(f"\n📰 Selected Article:")
    print(f"   ID: {article_data['article_id']}")
    print(f"   Title: {article_data['title']}")
    print(f"   Content Length: {len(article_data['content'])} characters")
    print(f"\n   Content Preview (first 200 chars):")
    print(f"   {article_data['content'][:200]}...")
    
    # Initialize output directory
    from config.settings import settings
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = settings.output_dir if hasattr(settings, 'output_dir') else "sample_data"
    
    # Use article_id for naming, fallback to timestamp if not available
    if article_data.get('article_id'):
        output_dir = os.path.join(output_base, f"post_textonly_{article_data['article_id']}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"post_textonly_{timestamp}")
    
    # Process article
    print("\n" + "=" * 80)
    print("Processing Article (Text Only)...")
    print("=" * 80)
    
    try:
        result = process_text_only(
            article_data['content'],
            output_dir,
            article_id=article_data.get('article_id'),
            post_method=args.post_method,
            max_retries=3
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("Results Summary")
        print("=" * 80)
        print(f"\n📄 Original Article:")
        print(f"   Title: {article_data['title']}")
        print(f"   ID: {article_data['article_id']}")
        
        print(f"\n🔄 Opposite Claims:")
        print(f"   {result['opposite_claims']}")
        
        print(f"\n✍️  Mirrored Article (first 300 chars):")
        print(f"   {result['mirrored_article'][:300]}...")
        
        print(f"\n📝 Post Text ({len(result['post_text'])} characters):")
        print(f"   {result['post_text']}")
        
        if result.get('critic_result'):
            print(f"\n📊 Critic Score: {result['critic_result']['score']}/100")
            print(f"   Evaluation: {'Excellent' if result['critic_result']['is_excellent'] else 'Needs Improvement'}")
        
        print(f"\n📁 Output Files:")
        print(f"   Complete Report: {os.path.join(result['output_dir'], 'final_report.json')}")
        print(f"   Intermediate Results: {os.path.join(result['output_dir'], 'intermediate_results.json')}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

