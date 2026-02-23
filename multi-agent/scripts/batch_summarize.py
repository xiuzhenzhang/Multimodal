"""Batch summarization script for all articles (no emotion reversal)"""
import sys
import os
import json
import shutil
import argparse
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from config.settings import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.prompt_templates import DIRECT_SUMMARIZATION_PROMPT


class BatchSummarizer:
    """Batch summarizer for all news articles"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_processed: bool = False
    ):
        """
        Initialize batch summarizer
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            skip_processed: Whether to skip already processed articles
        """
        self.loader = NewsLoader(dataset_path=dataset_path)
        self.output_dir = output_dir or os.path.join(settings.output_dir, "summarized")
        self.skip_processed = skip_processed
        
        # Statistics
        self.stats = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        # Load processed articles if skip_processed is enabled
        self.processed_articles = set()
        if self.skip_processed:
            self._load_processed_articles()
        
        # Initialize LLM
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Please set it in your .env file or environment variables."
            )
        
        original_key = os.environ.get("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key
            
            self.llm = ChatOpenAI(
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
    
    def _load_processed_articles(self):
        """Load list of already processed article IDs"""
        if not self.output_dir:
            return
        
        if not os.path.exists(self.output_dir):
            return
        
        # Scan output directory for processed articles
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                json_path = os.path.join(item_path, "summary.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                            article_id = summary.get("article_id")
                            if article_id:
                                self.processed_articles.add(article_id)
                    except Exception:
                        pass
    
    def _is_processed(self, article_id: str) -> bool:
        """Check if article has already been processed"""
        if not self.skip_processed:
            return False
        return article_id in self.processed_articles
    
    def find_image_files(self, article_dir: str) -> List[str]:
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
    
    def summarize_article(self, article_content: str) -> str:
        """Summarize article using DeepSeek API"""
        prompt = ChatPromptTemplate.from_template(DIRECT_SUMMARIZATION_PROMPT)
        messages = prompt.format_messages(original_article=article_content)
        
        response = self.llm.invoke(messages)
        post_text = response.content.strip()
        
        # Remove markdown formatting if present
        post_text = post_text.replace("**", "").strip()
        
        return post_text
    
    def process_all(self, start_from: int = 0, limit: Optional[int] = None):
        """Process all articles in the dataset"""
        print("=" * 80)
        print("Batch Summarization (No Emotion Reversal)")
        print("=" * 80)
        
        # Get all articles
        article_dirs = self.loader.get_all_articles()
        self.stats["total"] = len(article_dirs)
        
        if not article_dirs:
            print("❌ No articles found in dataset")
            return
        
        print(f"\n📊 Found {len(article_dirs)} articles")
        if limit:
            print(f"   Processing {limit} articles (starting from index {start_from})")
        elif start_from > 0:
            print(f"   Starting from index {start_from}")
        
        # Filter articles
        articles_to_process = article_dirs[start_from:]
        if limit:
            articles_to_process = articles_to_process[:limit]
        
        print(f"   Will process {len(articles_to_process)} articles\n")
        
        # Process each article
        for idx, article_dir in enumerate(articles_to_process, start=start_from + 1):
            article_id = os.path.basename(article_dir)
            
            print("\n" + "=" * 80)
            print(f"Processing Article {idx}/{len(article_dirs)}: {article_id}")
            print("=" * 80)
            
            # Check if already processed
            if self._is_processed(article_id):
                print(f"⏭️  Article {article_id} already processed, skipping...")
                self.stats["skipped"] += 1
                continue
            
            # Parse article
            article_data = self.loader.parse_article(article_dir)
            
            if not article_data:
                print(f"❌ Failed to parse article {article_id}")
                self.stats["failed"] += 1
                self.stats["errors"].append({
                    "article_id": article_id,
                    "error": "Failed to parse article"
                })
                continue
            
            print(f"📰 Title: {article_data['title']}")
            print(f"   Content Length: {len(article_data['content'])} characters")
            
            # Find image files
            image_files = self.find_image_files(article_dir)
            print(f"   Images found: {len(image_files)}")
            
            # Summarize
            try:
                print("🤖 Summarizing article (preserving original emotion)...")
                post_text = self.summarize_article(article_data['content'])
                word_count = len(post_text.split())
                print(f"✅ Generated post text ({word_count} words):")
                print(f"   {post_text[:200]}..." if len(post_text) > 200 else f"   {post_text}")
                
                # Create output directory
                output_dir = os.path.join(self.output_dir, article_id)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save JSON
                output_data = {
                    "article_id": article_id,
                    "title": article_data['title'],
                    "original_article": article_data['content'],
                    "post_text": post_text
                }
                
                json_path = os.path.join(output_dir, "summary.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # Copy images
                if image_files:
                    print(f"🖼️  Copying {len(image_files)} image(s)...")
                    for img_path in image_files:
                        img_name = os.path.basename(img_path)
                        dest_path = os.path.join(output_dir, img_name)
                        shutil.copy2(img_path, dest_path)
                
                print(f"✅ Successfully processed article {article_id}")
                self.stats["success"] += 1
                self.stats["processed"] += 1
                
                # Mark as processed
                if self.skip_processed:
                    self.processed_articles.add(article_id)
                
            except Exception as e:
                print(f"❌ Error processing article {article_id}: {e}")
                self.stats["failed"] += 1
                self.stats["errors"].append({
                    "article_id": article_id,
                    "error": str(e)
                })
                continue
            
            # Print progress
            self._print_progress()
        
        # Print final summary
        self._print_final_summary()
    
    def _print_progress(self):
        """Print current progress"""
        processed = self.stats["processed"]
        total = self.stats["total"]
        success = self.stats["success"]
        failed = self.stats["failed"]
        skipped = self.stats["skipped"]
        
        if total > 0:
            progress_pct = (processed / total) * 100
            print(f"\n📊 Progress: {processed}/{total} ({progress_pct:.1f}%) | "
                  f"✅ Success: {success} | ❌ Failed: {failed} | ⏭️  Skipped: {skipped}")
    
    def _print_final_summary(self):
        """Print final processing summary"""
        print("\n" + "=" * 80)
        print("Batch Summarization Summary")
        print("=" * 80)
        print(f"📊 Total Articles: {self.stats['total']}")
        print(f"✅ Successfully Processed: {self.stats['success']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"📝 Processed: {self.stats['processed']}")
        
        if self.stats["errors"]:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:10]:  # Show first 10 errors
                print(f"   - {error['article_id']}: {error.get('error', 'Unknown error')}")
            if len(self.stats["errors"]) > 10:
                print(f"   ... and {len(self.stats['errors']) - 10} more errors")
        
        # Save summary to file
        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "batch_summary.json")
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "output_dir": self.output_dir
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Batch summarize all news articles (no emotion reversal)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset directory (optional)",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (optional)",
        default="true_dataset"
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip already processed articles"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Index to start from (for resuming, default: 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    try:
        processor = BatchSummarizer(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            skip_processed=args.skip_processed
        )
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Please check your configuration:")
        print("   1. Create a .env file in the mutil-agent directory")
        print("   2. Add: DEEPSEEK_API_KEY=your_api_key_here")
        return
    
    # Process all articles
    processor.process_all(start_from=args.start_from, limit=args.limit)


if __name__ == "__main__":
    main()

