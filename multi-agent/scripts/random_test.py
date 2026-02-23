"""Random news article test script"""
import sys
import os

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from pipeline.poster_pipeline import PosterPipeline


def main():
    """Test pipeline with a random news article"""
    
    print("=" * 80)
    print("Random News Article Test")
    print("=" * 80)
    
    # Initialize news loader
    # Use default path (fake_news_dataset/sample_dataset) or specify custom path
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
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    try:
        pipeline = PosterPipeline()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Please check your configuration:")
        print("   1. Create a .env file in the mutil-agent directory")
        print("   2. Add: DEEPSEEK_API_KEY=your_api_key_here")
        print("   3. Optionally add: OPENAI_API_KEY=your_openai_key_here (for image generation)")
        print("\n   You can also run: python scripts/check_config.py")
        return
    
    # Process article
    print("\n" + "=" * 80)
    print("Processing Article...")
    print("=" * 80)
    
    try:
        result = pipeline.process(article_data['content'])
        
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
        
        print(f"\n📝 Post Text:")
        print(f"   {result['post_text']}")
        
        print(f"\n🎯 Visual Strategy:")
        print(f"   Strategy {result['strategy_result']['selected_strategy']}: {result['strategy_result']['strategy_name']}")
        print(f"   Reasoning: {result['strategy_result']['reasoning'][:200]}...")
        
        print(f"\n📁 Output Files:")
        print(f"   Final Post: {result['final_post_path']}")
        print(f"   Background Image: {result['background_image_path']}")
        print(f"   Complete Report: {os.path.join(result['output_dir'], 'final_report.json')}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

