"""Random test script for direct summarization (no emotion reversal)"""
import sys
import os
import json
import shutil
from typing import Optional

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from config.settings import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.prompt_templates import DIRECT_SUMMARIZATION_PROMPT


def find_image_files(article_dir: str) -> list[str]:
    """Find image files in article directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = []
    
    if not os.path.exists(article_dir):
        return image_files
    
    for item in os.listdir(article_dir):
        item_path = os.path.join(article_dir, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item.lower())
            if ext in image_extensions:
                image_files.append(item_path)
    
    return image_files


def summarize_article(article_content: str, llm: ChatOpenAI) -> str:
    """Summarize article using DeepSeek API"""
    prompt = ChatPromptTemplate.from_template(DIRECT_SUMMARIZATION_PROMPT)
    messages = prompt.format_messages(original_article=article_content)
    
    response = llm.invoke(messages)
    post_text = response.content.strip()
    
    # Remove markdown formatting if present
    post_text = post_text.replace("**", "").strip()
    
    return post_text


def main():
    """Test summarization with a random article"""
    print("=" * 80)
    print("Direct Summarization Test (No Emotion Reversal)")
    print("=" * 80)
    
    # Validate API key
    if not settings.deepseek_api_key:
        print("\n❌ Configuration Error: DEEPSEEK_API_KEY is not set.")
        print("\n💡 Please check your configuration:")
        print("   1. Create a .env file in the mutil-agent directory")
        print("   2. Add: DEEPSEEK_API_KEY=your_api_key_here")
        return
    
    # Initialize LLM
    original_key = os.environ.get("OPENAI_API_KEY", None)
    try:
        os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key
        
        llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.7
        )
    finally:
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    # Load random article
    print("\n📰 Loading random article...")
    loader = NewsLoader()
    article_data = loader.load_random_article()
    
    if not article_data:
        print("❌ No articles found in dataset")
        return
    
    print(f"✅ Loaded article: {article_data['article_id']}")
    print(f"   Title: {article_data['title']}")
    print(f"   Content Length: {len(article_data['content'])} characters")
    
    # Find image files
    article_dir = None
    if article_data.get('html_path'):
        article_dir = os.path.dirname(article_data['html_path'])
    
    if not article_dir or not os.path.exists(article_dir):
        # Try to find article directory from article_id
        articles = loader.get_all_articles()
        for art_dir in articles:
            if os.path.basename(art_dir) == article_data['article_id']:
                article_dir = art_dir
                break
    
    image_files = find_image_files(article_dir) if article_dir and os.path.exists(article_dir) else []
    print(f"   Images found: {len(image_files)}")
    
    # Summarize
    print("\n🤖 Summarizing article (preserving original emotion)...")
    try:
        post_text = summarize_article(article_data['content'], llm)
        print(f"✅ Generated post text ({len(post_text.split())} words):")
        print(f"   {post_text}")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join(settings.output_dir, "summarized", article_data['article_id'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    output_data = {
        "article_id": article_data['article_id'],
        "title": article_data['title'],
        "original_article": article_data['content'],
        "post_text": post_text
    }
    
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved summary to: {json_path}")
    
    # Copy images
    if image_files:
        print(f"\n🖼️  Copying {len(image_files)} image(s)...")
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, img_name)
            shutil.copy2(img_path, dest_path)
            print(f"   ✅ Copied: {img_name}")
    else:
        print("\n📷 No images found to copy")
    
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

