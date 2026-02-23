import json
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import settings
from utils.prompt_templates import CRITIC_PROMPT
from agents.transformer import Facts, TransformerOutput


class CriticResult(BaseModel):
    """Critic result model"""
    is_excellent: bool = Field(description="Whether the content is excellent enough to proceed")
    criticisms: list[str] = Field(default_factory=list, description="Detailed criticisms")
    strengths: list[str] = Field(default_factory=list, description="Strengths identified")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations for improvement")
    score: int = Field(default=0, description="Quality score 0-100")
    must_improve: list[str] = Field(default_factory=list, description="Critical issues that must be fixed")


class SentinelAgent:
    """Harsh Critic and Quality Gatekeeper"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Agent
        
        Args:
            model_name: Model name to use
        """
        # Validate API key
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Please set it in your .env file or environment variables.\n"
                "Example: DEEPSEEK_API_KEY=your_api_key_here"
            )
        
        # Temporarily set environment variable for ChatOpenAI validation
        original_key = os.environ.get("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key
            
            llm = ChatOpenAI(
                model=model_name or settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.3  # Lower temperature for more rigorous criticism
            )
        finally:
            # Restore original environment variable
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        
        self.llm = llm
    
    def _count_characters(self, text: str) -> int:
        """
        Count characters in text
        
        Args:
            text: Text to count characters in
        
        Returns:
            Character count
        """
        # Remove markdown formatting
        text_clean = text.replace("**", "").replace("*", "").strip()
        return len(text_clean)
    
    def critique(
        self,
        transformer_output: TransformerOutput,
        original_article: Optional[str] = None
    ) -> CriticResult:
        """
        Perform critical evaluation
        
        Args:
            transformer_output: Transformer Agent output
        
        Returns:
            Critic result
        """
        # Check character count (hard limit check before LLM evaluation)
        char_count = self._count_characters(transformer_output.post_text)
        char_count_issue = None
        if char_count > 280:
            char_count_issue = f"Post text exceeds 280 character limit ({char_count} characters). This is a CRITICAL FAILURE."
            print(f"❌ {char_count_issue}")
        
        prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)
        
        facts_str = json.dumps(
            transformer_output.facts.model_dump(),
            ensure_ascii=False,
            indent=2
        )
        
        # Use original article if provided, otherwise use empty string
        original_article_text = original_article if original_article else ""
        
        messages = prompt.format_messages(
            original_article=original_article_text,
            facts=facts_str,
            opposite_claims=transformer_output.opposite_claims,
            mirrored_article=transformer_output.mirrored_article,
            post_text=transformer_output.post_text
        )
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Parse response
        try:
            # Try to extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            critic_dict = json.loads(json_str)
            critic_result = CriticResult(**critic_dict)
            
            # Check for fact modification issues in criticisms/must_improve
            fact_modification_issue = None
            for item in critic_result.criticisms + critic_result.must_improve:
                if "fact" in item.lower() and ("only stance" in item.lower() or "only opinion" in item.lower() or "not fake news" in item.lower()):
                    fact_modification_issue = item
                    break
            
            # If fact modification issue detected, apply strict penalties
            if fact_modification_issue:
                # Deduct 60 points and cap maximum score at 40
                critic_result.score = min(40, max(0, critic_result.score - 60))
                # Mark as not excellent
                critic_result.is_excellent = False
                # Ensure it's in must_improve
                if fact_modification_issue not in critic_result.must_improve:
                    critic_result.must_improve.insert(0, fact_modification_issue)
                # Add to criticisms if not already there
                if fact_modification_issue not in critic_result.criticisms:
                    critic_result.criticisms.insert(0, fact_modification_issue)
                # Add recommendation
                if not any("modify actual facts" in r.lower() for r in critic_result.recommendations):
                    critic_result.recommendations.insert(0, "CRITICAL: Must modify actual FACTS (data, numbers, results, events), not just opinions/stance. Only changing stance/opinion does NOT create fake news. Regenerate with modified facts.")
            
            # If character count exceeds limit, apply strict penalties
            if char_count_issue:
                # Deduct 50 points
                critic_result.score = max(0, critic_result.score - 50)
                # Add to must_improve (critical issue)
                if char_count_issue not in critic_result.must_improve:
                    critic_result.must_improve.insert(0, char_count_issue)
                # Add to criticisms
                if char_count_issue not in critic_result.criticisms:
                    critic_result.criticisms.insert(0, char_count_issue)
                # Mark as not excellent
                critic_result.is_excellent = False
                # Add recommendation
                critic_result.recommendations.insert(0, "Post text MUST be regenerated to be within 280 characters maximum. Count characters, not words.")
            
            return critic_result
        except Exception as e:
            # If parsing fails, perform simple checks
            print(f"Critic result parsing failed: {e}")
            # Simple heuristic checks
            criticisms = []
            
            # Check character count
            if char_count_issue:
                criticisms.append(char_count_issue)
            
            # Check if empty
            if not transformer_output.post_text.strip():
                criticisms.append("Post text is empty")
            
            # Check article length
            if len(transformer_output.mirrored_article) < 200:
                criticisms.append("Mirrored article is too short (less than 200 characters)")
            
            is_excellent = len(criticisms) == 0
            # Apply 50 point deduction if character count exceeds limit
            base_score = 50 if is_excellent else 30
            score = max(0, base_score - (50 if char_count_issue else 0))
            
            return CriticResult(
                is_excellent=is_excellent,
                criticisms=criticisms,
                strengths=[],
                recommendations=["Please regenerate content with complete news summary including 5W1H, maximum 280 characters"] if criticisms else [],
                score=score,
                must_improve=criticisms
            )
    
    def should_rollback(self, critic_result: CriticResult) -> bool:
        """
        Determine if rollback to Agent 1 is needed
        
        Args:
            critic_result: Critic result
        
        Returns:
            Whether rollback is needed
        """
        # If score is above 75, proceed to Agent 3 without rollback
        if critic_result.score > 75:
            return False
        
        # Rollback if not excellent OR score is too low OR there are critical issues
        return not critic_result.is_excellent or critic_result.score < 70 or len(critic_result.must_improve) > 0

