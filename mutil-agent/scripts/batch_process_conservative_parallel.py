"""Conservative parallel processing - uses small worker count with heavy protection"""
import sys
import os
import json
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.news_loader import NewsLoader
from pipeline.poster_pipeline import PosterPipeline


class ConservativeParallelProcessor:
    """Conservative parallel processor with heavy protection for SD stability"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_processed: bool = False,
        max_retries: int = 3,
        num_workers: int = 2,  # Conservative default
        delay_between_tasks: float = 2.0  # Delay between starting tasks
    ):
        """
        Initialize conservative parallel processor
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            skip_processed: Whether to skip already processed articles
            max_retries: Maximum retry count for each article
            num_workers: Number of parallel threads (recommended: 2-3)
            delay_between_tasks: Delay in seconds between starting new tasks
        """
        self.loader = NewsLoader(dataset_path=dataset_path)
        self.output_dir = output_dir
        self.skip_processed = skip_processed
        self.max_retries = max_retries
        self.num_workers = num_workers
        self.delay_between_tasks = delay_between_tasks
        
        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        # Load processed articles
        self.processed_articles = set()
        if self.skip_processed:
            self._load_processed_articles()
        
        # Shared pipeline with locks
        self.pipeline = None
        self.pipeline_lock = threading.Lock()
        self.sd_generation_lock = threading.Lock()
        
        # Task submission lock to control rate
        self.submission_lock = threading.Lock()
        self.last_submission_time = 0
    
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
    
    def _get_pipeline(self) -> PosterPipeline:
        """Get or create shared pipeline (thread-safe)"""
        if self.pipeline is None:
            with self.pipeline_lock:
                if self.pipeline is None:
                    print("🔧 Initializing shared pipeline...")
                    self.pipeline = PosterPipeline(
                        max_retries=self.max_retries,
                        output_dir=self.output_dir,
                        sd_generation_lock=self.sd_generation_lock
                    )
                    print("✅ Shared pipeline ready!")
        return self.pipeline
    
    def _rate_limited_submit(self, executor, func, *args):
        """Submit task with rate limiting"""
        with self.submission_lock:
            # Ensure minimum delay between submissions
            current_time = time.time()
            time_since_last = current_time - self.last_submission_time
            if time_since_last < self.delay_between_tasks:
                sleep_time = self.delay_between_tasks - time_since_last
                time.sleep(sleep_time)
            
            future = executor.submit(func, *args)
            self.last_submission_time = time.time()
            return future
    
    def _process_single_article(
        self,
        article_dir: str,
        article_index: int,
        total_articles: int
    ) -> Tuple[str, bool, Optional[str], Dict[str, Any]]:
        """
        Process a single article with retry logic
        
        Args:
            article_dir: Path to article directory
            article_index: Current article index
            total_articles: Total number of articles
        
        Returns:
            Tuple of (article_id, success, error_message, article_data)
        """
        article_id = os.path.basename(article_dir)
        thread_id = threading.current_thread().name
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                # Parse article
                article_data = self.loader.parse_article(article_dir)
                
                if not article_data:
                    return (article_id, False, "Failed to parse article", {})
                
                # Get shared pipeline
                pipeline = self._get_pipeline()
                
                # Process article
                if attempt > 0:
                    print(f"[{thread_id}] Retry {attempt}/{self.max_retries} for {article_id}...")
                else:
                    print(f"[{thread_id}] Processing {article_id}...")
                
                result = pipeline.process(
                    article_data['content'],
                    article_id=article_id
                )
                
                # Save results
                self._save_article_results(article_dir, article_data, result, article_id)
                
                return (article_id, True, None, article_data)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a recoverable error
                if "index" in error_msg and "out of bounds" in error_msg:
                    print(f"[{thread_id}] ⚠️  Recoverable error for {article_id}, attempt {attempt+1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        # Wait before retry
                        time.sleep(2 * (attempt + 1))
                        continue
                elif "device" in error_msg.lower():
                    print(f"[{thread_id}] ⚠️  Device error for {article_id}, attempt {attempt+1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(3 * (attempt + 1))
                        continue
                
                # Non-recoverable or max retries reached
                if attempt == self.max_retries - 1:
                    full_error = f"{error_msg}\n{traceback.format_exc()}"
                    return (article_id, False, full_error, {})
        
        return (article_id, False, "Max retries exceeded", {})
    
    def _save_article_results(
        self,
        article_dir: str,
        article_data: Dict[str, Any],
        result: Dict[str, Any],
        article_id: str
    ):
        """Save article results"""
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
        
        original_json["post_text"] = result.get('post_text', '')
        
        output_json_path = os.path.join(article_output_dir, "news_data.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(original_json, f, ensure_ascii=False, indent=4)
        
        if 'background_image_path' in result and os.path.exists(result['background_image_path']):
            image_filename = "generated_image.png"
            output_image_path = os.path.join(article_output_dir, image_filename)
            if os.path.abspath(result['background_image_path']) != os.path.abspath(output_image_path):
                shutil.copy2(result['background_image_path'], output_image_path)
    
    def _update_stats(self, success: bool, error: Optional[Dict] = None):
        """Thread-safe statistics update"""
        with self.stats_lock:
            self.stats["processed"] += 1
            if success:
                self.stats["success"] += 1
            else:
                self.stats["failed"] += 1
                if error:
                    self.stats["errors"].append(error)
    
    def process_all(self, start_from: int = 0, limit: Optional[int] = None):
        """
        Process all articles with conservative parallelism
        
        Args:
            start_from: Index to start from
            limit: Maximum number of articles to process
        """
        print("=" * 80)
        print("Conservative Parallel Processing")
        print("=" * 80)
        
        # Get all articles
        print("\n📚 Loading news dataset...")
        article_dirs = self.loader.get_all_articles()
        self.stats["total"] = len(article_dirs)
        
        print(f"   Found {self.stats['total']} articles")
        print(f"   Using {self.num_workers} parallel threads (conservative)")
        print(f"   Task delay: {self.delay_between_tasks}s between submissions")
        print(f"   Retry limit: {self.max_retries} attempts per article")
        
        if self.stats["total"] == 0:
            print("❌ No articles found!")
            return
        
        # Determine processing range
        end_index = len(article_dirs)
        if limit:
            end_index = min(start_from + limit, len(article_dirs))
        
        articles_to_process = article_dirs[start_from:end_index]
        
        # Filter processed
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
        
        # Process with conservative parallelism
        print("\n🚀 Starting conservative parallel processing...")
        print("   ⚠️  Using rate limiting to prevent race conditions")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks with rate limiting
            future_to_article = {}
            for idx, article_dir in enumerate(filtered_articles):
                future = self._rate_limited_submit(
                    executor,
                    self._process_single_article,
                    article_dir,
                    start_from + idx + 1,
                    len(article_dirs)
                )
                future_to_article[future] = article_dir
                
                if (idx + 1) % 10 == 0:
                    print(f"   Submitted {idx + 1}/{len(filtered_articles)} tasks...")
            
            # Process completed tasks
            for future in as_completed(future_to_article):
                article_dir = future_to_article[future]
                article_id = os.path.basename(article_dir)
                
                try:
                    article_id, success, error_msg, article_data = future.result()
                    
                    if success:
                        print(f"✅ [{self.stats['processed']+1}/{len(filtered_articles)}] {article_id}")
                        self._update_stats(True)
                        if self.skip_processed:
                            self.processed_articles.add(article_id)
                    else:
                        print(f"❌ [{self.stats['processed']+1}/{len(filtered_articles)}] {article_id}")
                        if error_msg:
                            print(f"   Error: {error_msg[:150]}")
                        self._update_stats(False, {
                            "article_id": article_id,
                            "error": error_msg
                        })
                    
                    if self.stats["processed"] % 5 == 0:
                        self._print_progress()
                
                except Exception as e:
                    print(f"❌ Unexpected error for {article_id}: {e}")
                    self._update_stats(False, {
                        "article_id": article_id,
                        "error": str(e)
                    })
        
        self._print_final_summary()
    
    def _print_progress(self):
        """Print current progress"""
        with self.stats_lock:
            processed = self.stats["processed"]
            total = self.stats["total"] - self.stats["skipped"]
            success = self.stats["success"]
            failed = self.stats["failed"]
            
            progress_pct = (processed / total * 100) if total > 0 else 0
            success_rate = (success / processed * 100) if processed > 0 else 0
            
            print(f"\n📊 Progress: {processed}/{total} ({progress_pct:.1f}%) | "
                  f"✅ {success} | ❌ {failed} | Success Rate: {success_rate:.1f}%")
    
    def _print_final_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("Conservative Parallel Processing Summary")
        print("=" * 80)
        print(f"📊 Total: {self.stats['total']}")
        print(f"✅ Success: {self.stats['success']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"📝 Processed: {self.stats['processed']}")
        print(f"🧵 Workers: {self.num_workers}")
        print(f"⏱️  Task Delay: {self.delay_between_tasks}s")
        
        if self.stats["processed"] > 0:
            success_rate = (self.stats["success"] / self.stats["processed"] * 100)
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.stats["errors"]:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:5]:
                print(f"   - {error['article_id']}: {error.get('error', 'Unknown')[:100]}")
            if len(self.stats["errors"]) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more")
        
        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "batch_summary_conservative.json")
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "output_dir": self.output_dir,
                "num_workers": self.num_workers,
                "delay_between_tasks": self.delay_between_tasks,
                "method": "conservative_parallel"
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Summary: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conservative parallel processing (SD safe)")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel threads (recommended: 2-3, max: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between task submissions (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Validate workers
    if args.workers > 4:
        print(f"⚠️  Warning: {args.workers} workers may cause instability")
        print(f"   Recommended: 2-3 workers")
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    processor = ConservativeParallelProcessor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        skip_processed=args.skip_processed,
        max_retries=args.max_retries,
        num_workers=args.workers,
        delay_between_tasks=args.delay
    )
    
    processor.process_all(start_from=args.start_from, limit=args.limit)


if __name__ == "__main__":
    main()
