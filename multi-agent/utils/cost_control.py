"""Cost control wrapper for LLM calls"""
from typing import Optional, Callable, Any
from functools import wraps
import tiktoken


class CostControl:
    """Wrapper to add cost control to LLM calls"""
    
    def __init__(self, usage_tracker):
        """
        Initialize cost control
        
        Args:
            usage_tracker: UsageTracker instance
        """
        self.usage_tracker = usage_tracker
        self.encoding = None  # Lazy load encoding
    
    def _get_encoding(self):
        """Get tiktoken encoding (lazy load)"""
        if self.encoding is None:
            try:
                # Use cl100k_base which works for GPT-3.5, GPT-4, and most models
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Fallback to approximate counting
                self.encoding = None
        return self.encoding
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count
        
        Returns:
            Number of tokens
        """
        encoding = self._get_encoding()
        if encoding:
            return len(encoding.encode(text))
        else:
            # Fallback: approximate 1 token = 4 characters
            return len(text) // 4
    
    def estimate_request_cost(
        self,
        model: str,
        messages: list,
        is_image: bool = False
    ) -> float:
        """
        Estimate cost for a request
        
        Args:
            model: Model name
            messages: List of messages (for chat models)
            is_image: Whether this is an image generation
        
        Returns:
            Estimated cost
        """
        if is_image:
            return self.usage_tracker.estimate_cost(model, is_image=True)
        
        # Count input tokens
        total_text = ""
        for msg in messages:
            if isinstance(msg, dict):
                total_text += msg.get("content", "")
            elif hasattr(msg, "content"):
                total_text += str(msg.content)
            else:
                total_text += str(msg)
        
        tokens_input = self.count_tokens(total_text)
        # Estimate output tokens (typically 50-200 tokens for our use cases)
        tokens_output = 150  # Conservative estimate
        
        return self.usage_tracker.estimate_cost(model, tokens_input, tokens_output)
    
    def check_and_record(
        self,
        model: str,
        messages: list,
        response: Any = None,
        is_image: bool = False,
        actual_tokens_input: Optional[int] = None,
        actual_tokens_output: Optional[int] = None
    ):
        """
        Check if request is allowed and record usage
        
        Args:
            model: Model name
            messages: Input messages
            response: Response object (may contain token usage info)
            is_image: Whether this was an image generation
            actual_tokens_input: Actual input tokens if known
            actual_tokens_output: Actual output tokens if known
        
        Returns:
            Tuple of (tokens_input, tokens_output, cost)
        """
        # Get actual token counts if available
        if actual_tokens_input is None or actual_tokens_output is None:
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    actual_tokens_input = token_usage.get('prompt_tokens', 0)
                    actual_tokens_output = token_usage.get('completion_tokens', 0)
        
        # If still unknown, estimate
        if actual_tokens_input is None:
            total_text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    total_text += msg.get("content", "")
                elif hasattr(msg, "content"):
                    total_text += str(msg.content)
                else:
                    total_text += str(msg)
            actual_tokens_input = self.count_tokens(total_text)
        
        if actual_tokens_output is None and not is_image:
            # Estimate output tokens
            if hasattr(response, 'content'):
                actual_tokens_output = self.count_tokens(str(response.content))
            else:
                actual_tokens_output = 150  # Default estimate
        
        # Record usage
        self.usage_tracker.record_usage(
            model=model,
            tokens_input=actual_tokens_input or 0,
            tokens_output=actual_tokens_output or 0,
            is_image=is_image
        )
        
        return actual_tokens_input, actual_tokens_output, self.usage_tracker.estimate_cost(
            model, actual_tokens_input or 0, actual_tokens_output or 0, is_image
        )


def with_cost_control(usage_tracker, model: str, is_image: bool = False):
    """
    Decorator to add cost control to LLM call functions
    
    Args:
        usage_tracker: UsageTracker instance
        model: Model name
        is_image: Whether this is an image generation
    
    Returns:
        Decorator function
    """
    cost_control = CostControl(usage_tracker)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract messages from args/kwargs (adjust based on your function signature)
            messages = None
            if 'messages' in kwargs:
                messages = kwargs['messages']
            elif len(args) > 0:
                messages = args[0] if isinstance(args[0], list) else None
            
            if messages:
                # Estimate cost
                estimated_cost = cost_control.estimate_request_cost(model, messages, is_image)
                
                # Check if request is allowed
                can_proceed, reason = usage_tracker.can_make_request(estimated_cost)
                if not can_proceed:
                    raise RuntimeError(f"Cost limit exceeded: {reason}")
                
                # Make the call
                response = func(*args, **kwargs)
                
                # Record usage
                cost_control.check_and_record(
                    model=model,
                    messages=messages,
                    response=response,
                    is_image=is_image
                )
                
                return response
            else:
                # No messages, just call the function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

