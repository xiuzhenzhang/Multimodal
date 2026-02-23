"""Usage tracking and cost control for API calls"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path


class UsageTracker:
    """Track API usage and enforce cost limits"""
    
    def __init__(self, usage_file: str = "./usage_log.json"):
        """
        Initialize usage tracker
        
        Args:
            usage_file: Path to JSON file storing usage data
        """
        self.usage_file = usage_file
        self.usage_data = self._load_usage_data()
        
        # Cost per 1K tokens (approximate, adjust based on your models)
        # DeepSeek pricing (example, check actual pricing)
        self.cost_per_1k_tokens = {
            "deepseek-chat": 0.00014,  # $0.14 per 1M tokens input, $0.28 per 1M tokens output
            "gpt-4-turbo-preview": 0.01,  # $10 per 1M tokens input, $30 per 1M tokens output
            "gpt-4": 0.03,  # $30 per 1M tokens input, $60 per 1M tokens output
            "dall-e-3": 0.04,  # $0.04 per image (1024x1024)
        }
    
    def _load_usage_data(self) -> Dict:
        """Load usage data from file"""
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load usage data: {e}")
        
        # Initialize default structure
        return {
            "daily_usage": {},
            "total_usage": {
                "total_requests": 0,
                "total_tokens_input": 0,
                "total_tokens_output": 0,
                "total_cost": 0.0,
                "total_images": 0
            },
            "limits": {
                "daily_cost_limit": float(os.getenv("DAILY_COST_LIMIT", "10.0")),  # $10 per day default
                "per_request_cost_limit": float(os.getenv("PER_REQUEST_COST_LIMIT", "1.0")),  # $1 per request default
                "daily_request_limit": int(os.getenv("DAILY_REQUEST_LIMIT", "100")),  # 100 requests per day default
            }
        }
    
    def _save_usage_data(self):
        """Save usage data to file"""
        try:
            os.makedirs(os.path.dirname(self.usage_file) if os.path.dirname(self.usage_file) else ".", exist_ok=True)
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save usage data: {e}")
    
    def _get_today_key(self) -> str:
        """Get today's date as string key"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_today_usage(self) -> Dict:
        """Get today's usage data"""
        today = self._get_today_key()
        if today not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][today] = {
                "requests": 0,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost": 0.0,
                "images": 0
            }
        return self.usage_data["daily_usage"][today]
    
    def estimate_cost(
        self,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        is_image: bool = False
    ) -> float:
        """
        Estimate cost for API call
        
        Args:
            model: Model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            is_image: Whether this is an image generation request
        
        Returns:
            Estimated cost in USD
        """
        if is_image:
            return self.cost_per_1k_tokens.get(model, 0.04)
        
        cost_per_1k = self.cost_per_1k_tokens.get(model, 0.01)
        input_cost = (tokens_input / 1000) * cost_per_1k
        output_cost = (tokens_output / 1000) * cost_per_1k * 2  # Output usually costs more
        return input_cost + output_cost
    
    def can_make_request(self, estimated_cost: float) -> Tuple[bool, Optional[str]]:
        """
        Check if a request can be made based on limits
        
        Args:
            estimated_cost: Estimated cost of the request
        
        Returns:
            Tuple of (can_proceed, reason_if_not)
        """
        limits = self.usage_data["limits"]
        today_usage = self._get_today_usage()
        
        # Check per-request limit
        if estimated_cost > limits["per_request_cost_limit"]:
            return False, f"Request cost ${estimated_cost:.4f} exceeds per-request limit ${limits['per_request_cost_limit']:.2f}"
        
        # Check daily cost limit
        if today_usage["cost"] + estimated_cost > limits["daily_cost_limit"]:
            return False, f"Daily cost limit ${limits['daily_cost_limit']:.2f} would be exceeded (current: ${today_usage['cost']:.4f}, request: ${estimated_cost:.4f})"
        
        # Check daily request limit
        if today_usage["requests"] >= limits["daily_request_limit"]:
            return False, f"Daily request limit {limits['daily_request_limit']} reached (current: {today_usage['requests']})"
        
        return True, None
    
    def record_usage(
        self,
        model: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        is_image: bool = False,
        actual_cost: Optional[float] = None
    ):
        """
        Record API usage
        
        Args:
            model: Model name
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            is_image: Whether this was an image generation
            actual_cost: Actual cost if known, otherwise will be estimated
        """
        if actual_cost is None:
            cost = self.estimate_cost(model, tokens_input, tokens_output, is_image)
        else:
            cost = actual_cost
        
        today_usage = self._get_today_usage()
        total_usage = self.usage_data["total_usage"]
        
        # Update today's usage
        today_usage["requests"] += 1
        today_usage["tokens_input"] += tokens_input
        today_usage["tokens_output"] += tokens_output
        today_usage["cost"] += cost
        if is_image:
            today_usage["images"] += 1
        
        # Update total usage
        total_usage["total_requests"] += 1
        total_usage["total_tokens_input"] += tokens_input
        total_usage["total_tokens_output"] += tokens_output
        total_usage["total_cost"] += cost
        if is_image:
            total_usage["total_images"] += 1
        
        self._save_usage_data()
    
    def get_usage_summary(self) -> Dict:
        """Get current usage summary"""
        today_usage = self._get_today_usage()
        limits = self.usage_data["limits"]
        total_usage = self.usage_data["total_usage"]
        
        return {
            "today": {
                "date": self._get_today_key(),
                "requests": today_usage["requests"],
                "tokens_input": today_usage["tokens_input"],
                "tokens_output": today_usage["tokens_output"],
                "cost": today_usage["cost"],
                "images": today_usage.get("images", 0),
                "cost_limit": limits["daily_cost_limit"],
                "request_limit": limits["daily_request_limit"],
                "cost_remaining": max(0, limits["daily_cost_limit"] - today_usage["cost"]),
                "requests_remaining": max(0, limits["daily_request_limit"] - today_usage["requests"])
            },
            "total": {
                "requests": total_usage["total_requests"],
                "tokens_input": total_usage["total_tokens_input"],
                "tokens_output": total_usage["total_tokens_output"],
                "cost": total_usage["total_cost"],
                "images": total_usage.get("total_images", 0)
            }
        }
    
    def print_usage_summary(self):
        """Print usage summary to console"""
        summary = self.get_usage_summary()
        
        print("\n" + "=" * 60)
        print("📊 API Usage Summary")
        print("=" * 60)
        print(f"\n📅 Today ({summary['today']['date']}):")
        print(f"   Requests: {summary['today']['requests']}/{summary['today']['request_limit']}")
        print(f"   Cost: ${summary['today']['cost']:.4f} / ${summary['today']['cost_limit']:.2f}")
        print(f"   Remaining: ${summary['today']['cost_remaining']:.2f} budget, {summary['today']['requests_remaining']} requests")
        print(f"   Tokens: {summary['today']['tokens_input']:,} input, {summary['today']['tokens_output']:,} output")
        if summary['today']['images'] > 0:
            print(f"   Images: {summary['today']['images']}")
        
        print(f"\n📈 Total (All Time):")
        print(f"   Requests: {summary['total']['requests']}")
        print(f"   Cost: ${summary['total']['cost']:.4f}")
        print(f"   Tokens: {summary['total']['tokens_input']:,} input, {summary['total']['tokens_output']:,} output")
        if summary['total']['images'] > 0:
            print(f"   Images: {summary['total']['images']}")
        print("=" * 60 + "\n")
