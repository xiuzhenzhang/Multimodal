"""Batch process all news articles in the dataset"""
import sys
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from pipeline.poster_pipeline import PosterPipeline


class BatchProcessor:
    """Batch processor for all news articles"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_processed: bool = False,
        max_retries: int = 3
    ):
        """
        Initialize batch processor
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            skip_processed: Whether to skip already processed articles
            max_retries: Maximum retry count for each article
        """
        self.loader = NewsLoader(dataset_path=dataset_path)
        self.output_dir = output_dir
        self.skip_processed = skip_processed
        self.max_retries = max_retries
        
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
    
    def _load_processed_articles(self):
        """Load list of already processed article IDs"""
        if not self.output_dir:
            return
        
        if not os.path.exists(self.output_dir):
            return
        
        # Scan output directory for processed articles
        # New format: output_dir/article_id/news_data.json
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                json_path = os.path.join(item_path, "news_data.json")
                if os.path.exists(json_path):
                    # Directory name is the article_id
                    self.processed_articles.add(item)
    
    def _is_processed(self, article_id: str) -> bool:
        """Check if article has already been processed"""
        if not self.skip_processed:
            return False
        return article_id in self.processed_articles
    
    def process_all(self, start_from: int = 0, limit: Optional[int] = None):
        """
        Process all articles in the dataset
        
        Args:
            start_from: Index to start from (for resuming)
            limit: Maximum number of articles to process (None for all)
        """
        print("=" * 80)
        print("Batch Processing All News Articles")
        print("=" * 80)
        
        # Get all articles
        print("\n📚 Loading news dataset...")
        article_dirs = self.loader.get_all_articles()
        self.stats["total"] = len(article_dirs)
        
        print(f"   Found {self.stats['total']} articles in dataset")
        
        if self.stats["total"] == 0:
            print("❌ No articles found in dataset!")
            return
        
        # Initialize pipeline
        print("\n🚀 Initializing pipeline...")
        try:
            pipeline = PosterPipeline(
                max_retries=self.max_retries,
                output_dir=self.output_dir
            )
        except ValueError as e:
            print(f"\n❌ Configuration Error: {e}")
            print("\n💡 Please check your configuration:")
            print("   1. Create a .env file in the mutil-agent directory")
            print("   2. Add: DEEPSEEK_API_KEY=your_api_key_here")
            print("   3. Optionally add: OPENAI_API_KEY=your_openai_key_here (for image generation)")
            return
        
        # Determine processing range
        end_index = len(article_dirs)
        if limit:
            end_index = min(start_from + limit, len(article_dirs))
        
        articles_to_process = article_dirs[start_from:end_index]
        print(f"\n📋 Processing articles {start_from + 1} to {end_index} (total: {len(articles_to_process)})")
        
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
            
            # Process article
            try:
                result = pipeline.process(
                    article_data['content'],
                    article_id=article_id
                )
                
                # Save results in new format: article_id folder with JSON + image
                self._save_article_results(article_dir, article_data, result, article_id)
                
                print(f"✅ Successfully processed article {article_id}")
                self.stats["success"] += 1
                
                # Mark as processed
                if self.skip_processed:
                    self.processed_articles.add(article_id)
                
            except Exception as e:
                print(f"❌ Error processing article {article_id}: {e}")
                self.stats["failed"] += 1
                self.stats["errors"].append({
                    "article_id": article_id,
                    "title": article_data.get('title', 'Unknown'),
                    "error": str(e)
                })
                import traceback
                traceback.print_exc()
            
            self.stats["processed"] += 1
            
            # Print progress
            self._print_progress()
        
        # Print final summary
        self._print_final_summary()
    
    def _save_article_results(
        self,
        article_dir: str,
        article_data: Dict[str, Any],
        result: Dict[str, Any],
        article_id: str
    ):
        """
        Save article results in new format
        
        Args:
            article_dir: Original article directory
            article_data: Parsed article data
            result: Processing result from pipeline
            article_id: Article ID
        """
        # Pipeline already created the output directory with article_id
        article_output_dir = result['output_dir']
        
        # Load original JSON if exists
        json_path = os.path.join(article_dir, "news_data.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                original_json = json.load(f)
        else:
            # Create minimal JSON structure if no original JSON
            original_json = {
                "original_headline": article_data.get('title', ''),
                "original_content": article_data.get('content', ''),
                "source": article_data.get('source', 'unknown')
            }
        
        # Add post_text to JSON
        original_json["post_text"] = result.get('post_text', '')
        
        # Save updated JSON
        output_json_path = os.path.join(article_output_dir, "news_data.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(original_json, f, ensure_ascii=False, indent=4)
        
        # Copy generated image (without text overlay)
        # Use background_image_path instead of final_post_path
        if 'background_image_path' in result and os.path.exists(result['background_image_path']):
            image_filename = "generated_image.png"
            output_image_path = os.path.join(article_output_dir, image_filename)
            # If it's not already in the right place, copy it
            if os.path.abspath(result['background_image_path']) != os.path.abspath(output_image_path):
                shutil.copy2(result['background_image_path'], output_image_path)
            print(f"   💾 Saved image: {output_image_path}")
        
        print(f"   💾 Saved JSON: {output_json_path}")
    
    def _print_progress(self):
        """Print current progress"""
        processed = self.stats["processed"]
        total = self.stats["total"]
        success = self.stats["success"]
        failed = self.stats["failed"]
        skipped = self.stats["skipped"]
        
        progress_pct = (processed / total * 100) if total > 0 else 0
        
        print(f"\n📊 Progress: {processed}/{total} ({progress_pct:.1f}%) | "
              f"✅ Success: {success} | ❌ Failed: {failed} | ⏭️  Skipped: {skipped}")
    
    def _print_final_summary(self):
        """Print final processing summary"""
        print("\n" + "=" * 80)
        print("Batch Processing Summary")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process all news articles")
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
        default=None
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry count per article (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchProcessor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        skip_processed=args.skip_processed,
        max_retries=args.max_retries
    )
    
    # Process all articles
    processor.process_all(start_from=args.start_from, limit=args.limit)


if __name__ == "__main__":
    main()

