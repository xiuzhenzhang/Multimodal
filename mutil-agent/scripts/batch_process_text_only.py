"""Batch process all news articles in the dataset (Text only - no image generation)"""
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

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
    
    while retry_count < max_retries:
        try:
            # Agent 1 processing (only on first attempt, or if transformer_output is None)
            if transformer_output is None:
                transformer_output = transformer.process(news_article, post_method=post_method)
            
            # Stage 2: Critical evaluation (Agent 2 - Critic)
            critic_result = sentinel.critique(transformer_output, original_article=news_article)
            
            # Check if rollback is needed
            if not sentinel.should_rollback(critic_result):
                break
            else:
                retry_count += 1
                # Use adjust method instead of complete regeneration
                transformer_output = transformer.adjust(transformer_output, critic_result)
                
        except Exception as e:
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
    
    return final_result


class BatchProcessorTextOnly:
    """Batch processor for all news articles (text only, no image generation)"""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_processed: bool = False,
        max_retries: int = 3,
        post_method: Optional[int] = None
    ):
        """
        Initialize batch processor
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            skip_processed: Whether to skip already processed articles
            max_retries: Maximum retry count for each article
            post_method: Post text generation method (1: modified facts, 2: modified evidence). If None, uses settings.post_method
        """
        from config.settings import settings
        self.loader = NewsLoader(dataset_path=dataset_path)
        self.output_dir = output_dir
        self.skip_processed = skip_processed
        self.max_retries = max_retries
        self.post_method = post_method if post_method is not None else settings.post_method
        
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
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                report_path = os.path.join(item_path, "final_report.json")
                if os.path.exists(report_path):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            report = json.load(f)
                            # Try to extract article ID from report if available
                            article_id = report.get("article_id")
                            if article_id:
                                self.processed_articles.add(article_id)
                    except Exception:
                        pass
    
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
        print("Batch Processing All News Articles (Text Only - No Image Generation)")
        print("=" * 80)
        
        # Get all articles
        print("\n📚 Loading news dataset...")
        article_dirs = self.loader.get_all_articles()
        self.stats["total"] = len(article_dirs)
        
        print(f"   Found {self.stats['total']} articles in dataset")
        print(f"   Post Method: {self.post_method} ({'Modified Facts' if self.post_method == 1 else 'Modified Evidence'})")
        
        if self.stats["total"] == 0:
            print("❌ No articles found in dataset!")
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
            article_data = self.loader.parse_sina_article(article_dir)
            
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
            
            # Create output directory for this article (use article_id for naming)
            if self.output_dir:
                article_output_dir = os.path.join(self.output_dir, f"post_textonly_{article_id}")
            else:
                from config.settings import settings
                output_base = settings.output_dir if hasattr(settings, 'output_dir') else "sample_data"
                article_output_dir = os.path.join(output_base, f"post_textonly_{article_id}")
            
            # Process article
            try:
                result = process_text_only(
                    article_data['content'],
                    article_output_dir,
                    article_id=article_id,
                    post_method=self.post_method,
                    max_retries=self.max_retries
                )
                
                # Add article_id to final report for skip_processed functionality
                if self.skip_processed:
                    report_path = os.path.join(article_output_dir, "final_report.json")
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, "r", encoding="utf-8") as f:
                                report = json.load(f)
                            report["article_id"] = article_id
                            report["title"] = article_data.get('title', 'Unknown')
                            with open(report_path, "w", encoding="utf-8") as f:
                                json.dump(report, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                
                print(f"✅ Successfully processed article {article_id}")
                if result.get('critic_result'):
                    print(f"   Score: {result['critic_result']['score']}/100")
                print(f"   Post Text Length: {len(result['post_text'])} characters")
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
            summary_path = os.path.join(self.output_dir, "batch_summary_textonly.json")
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "output_dir": self.output_dir,
                "post_method": self.post_method,
                "note": "Text-only processing (no image generation)"
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process all news articles (text only)")
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
        "--post-method",
        type=int,
        choices=[1, 2],
        default=None,
        help="Post text generation method: 1=modified facts, 2=modified evidence (default: from settings)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchProcessorTextOnly(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        skip_processed=args.skip_processed,
        max_retries=args.max_retries,
        post_method=args.post_method
    )
    
    # Process all articles
    processor.process_all(start_from=args.start_from, limit=args.limit)


if __name__ == "__main__":
    main()

