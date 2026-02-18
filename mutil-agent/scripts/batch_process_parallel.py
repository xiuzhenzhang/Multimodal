"""Parallel batch process all news articles in the dataset"""
import sys
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import traceback

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from pipeline.poster_pipeline import PosterPipeline


def process_single_article(
    article_dir: str,
    output_dir: Optional[str],
    max_retries: int,
    article_index: int,
    total_articles: int
) -> Tuple[str, bool, Optional[str], Dict[str, Any]]:
    """
    Process a single article (worker function for parallel processing)
    
    Args:
        article_dir: Path to article directory
        output_dir: Output directory for results
        max_retries: Maximum retry count
        article_index: Current article index
        total_articles: Total number of articles
    
    Returns:
        Tuple of (article_id, success, error_message, article_data)
    """
    article_id = os.path.basename(article_dir)
    
    try:
        # Parse article
        loader = NewsLoader()
        article_data = loader.parse_article(article_dir)
        
        if not article_data:
            return (article_id, False, "Failed to parse article", {})
        
        # Initialize pipeline for this worker
        pipeline = PosterPipeline(
            max_retries=max_retries,
            output_dir=output_dir
        )
        
        # Process article
        result = pipeline.process(
            article_data['content'],
            article_id=article_id
        )
        
        # Save results
        _save_article_results(article_dir, article_data, result, article_id)
        
        return (article_id, True, None, article_data)
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (article_id, False, error_msg, {})


def _save_article_results(
    article_dir: str,
    article_data: Dict[str, Any],
    result: Dict[str, Any],
    article_id: str
):
    """Save article results in new format"""
    article_output_dir = result['output_dir']
    
    # Load original JSON if exists
    json_path = os.path.join(article_dir, "news_data.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            original_json = json.load(f)
    else:
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
    
    # Copy generated image
    if 'background_image_path' in result and os.path.exists(result['background_image_path']):
        image_filename = "generated_image.png"
        output_image_path = os.path.join(article_output_dir, image_filename)
        if os.path.abspath(result['background_image_path']) != os.path.abspath(output_image_path):
            shutil.copy2(result['background_image_path'], output_image_path)


class ParallelBatchProcessor:
    """Parallel batch processor for all news articles"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_processed: bool = False,
        max_retries: int = 3,
        num_workers: Optional[int] = None
    ):
        """
        Initialize parallel batch processor
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            skip_processed: Whether to skip already processed articles
            max_retries: Maximum retry count for each article
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.loader = NewsLoader(dataset_path=dataset_path)
        self.output_dir = output_dir
        self.skip_processed = skip_processed
        self.max_retries = max_retries
        self.num_workers = num_workers or cpu_count()
        
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
        if not self.output_dir or not os.path.exists(self.output_dir):
            return
        
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                json_path = os.path.join(item_path, "news_data.json")
                if os.path.exists(json_path):
                    self.processed_articles.add(item)
    
    def _is_processed(self, article_id: str) -> bool:
        """Check if article has already been processed"""
        if not self.skip_processed:
            return False
        return article_id in self.processed_articles
    
    def process_all(self, start_from: int = 0, limit: Optional[int] = None):
        """
        Process all articles in parallel
        
        Args:
            start_from: Index to start from (for resuming)
            limit: Maximum number of articles to process (None for all)
        """
        print("=" * 80)
        print("Parallel Batch Processing All News Articles")
        print("=" * 80)
        
        # Get all articles
        print("\n📚 Loading news dataset...")
        article_dirs = self.loader.get_all_articles()
        self.stats["total"] = len(article_dirs)
        
        print(f"   Found {self.stats['total']} articles in dataset")
        print(f"   Using {self.num_workers} parallel workers")
        
        if self.stats["total"] == 0:
            print("❌ No articles found in dataset!")
            return
        
        # Determine processing range
        end_index = len(article_dirs)
        if limit:
            end_index = min(start_from + limit, len(article_dirs))
        
        articles_to_process = article_dirs[start_from:end_index]
        
        # Filter out already processed articles
        filtered_articles = []
        for article_dir in articles_to_process:
            article_id = os.path.basename(article_dir)
            if self._is_processed(article_id):
                self.stats["skipped"] += 1
            else:
                filtered_articles.append(article_dir)
        
        print(f"\n📋 Processing {len(filtered_articles)} articles (skipped {self.stats['skipped']})")
        
        if not filtered_articles:
            print("✅ All articles already processed!")
            self._print_final_summary()
            return
        
        # Process articles in parallel
        print("\n🚀 Starting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_article = {}
            for idx, article_dir in enumerate(filtered_articles):
                future = executor.submit(
                    process_single_article,
                    article_dir,
                    self.output_dir,
                    self.max_retries,
                    start_from + idx + 1,
                    len(article_dirs)
                )
                future_to_article[future] = article_dir
            
            # Process completed tasks
            for future in as_completed(future_to_article):
                article_dir = future_to_article[future]
                article_id = os.path.basename(article_dir)
                
                try:
                    article_id, success, error_msg, article_data = future.result()
                    
                    if success:
                        print(f"✅ [{self.stats['processed']+1}/{len(filtered_articles)}] {article_id}")
                        self.stats["success"] += 1
                        if self.skip_processed:
                            self.processed_articles.add(article_id)
                    else:
                        print(f"❌ [{self.stats['processed']+1}/{len(filtered_articles)}] {article_id}: {error_msg}")
                        self.stats["failed"] += 1
                        self.stats["errors"].append({
                            "article_id": article_id,
                            "error": error_msg
                        })
                    
                    self.stats["processed"] += 1
                    
                    # Print progress every 10 articles
                    if self.stats["processed"] % 10 == 0:
                        self._print_progress()
                
                except Exception as e:
                    print(f"❌ Unexpected error for {article_id}: {e}")
                    self.stats["failed"] += 1
                    self.stats["processed"] += 1
                    self.stats["errors"].append({
                        "article_id": article_id,
                        "error": str(e)
                    })
        
        # Print final summary
        self._print_final_summary()
    
    def _print_progress(self):
        """Print current progress"""
        processed = self.stats["processed"]
        total = self.stats["total"] - self.stats["skipped"]
        success = self.stats["success"]
        failed = self.stats["failed"]
        
        progress_pct = (processed / total * 100) if total > 0 else 0
        
        print(f"\n📊 Progress: {processed}/{total} ({progress_pct:.1f}%) | "
              f"✅ {success} | ❌ {failed}")
    
    def _print_final_summary(self):
        """Print final processing summary"""
        print("\n" + "=" * 80)
        print("Parallel Batch Processing Summary")
        print("=" * 80)
        print(f"📊 Total Articles: {self.stats['total']}")
        print(f"✅ Successfully Processed: {self.stats['success']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"📝 Total Processed: {self.stats['processed']}")
        print(f"⚡ Workers Used: {self.num_workers}")
        
        if self.stats["errors"]:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:10]:
                print(f"   - {error['article_id']}: {error.get('error', 'Unknown error')[:100]}")
            if len(self.stats["errors"]) > 10:
                print(f"   ... and {len(self.stats['errors']) - 10} more errors")
        
        # Save summary to file
        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "batch_summary_parallel.json")
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "output_dir": self.output_dir,
                "num_workers": self.num_workers
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel batch process all news articles")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: CPU count = {cpu_count()})"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = ParallelBatchProcessor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        skip_processed=args.skip_processed,
        max_retries=args.max_retries,
        num_workers=args.workers
    )
    
    # Process all articles
    processor.process_all(start_from=args.start_from, limit=args.limit)


if __name__ == "__main__":
    main()
