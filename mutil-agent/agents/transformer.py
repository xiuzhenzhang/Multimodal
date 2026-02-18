"""Agent 1: Mirror Creation and Post Copywriting Expert"""
import json
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from config.settings import settings
from utils.prompt_templates import (
    FACT_EXTRACTION_PROMPT,
    CLAIM_REVERSAL_PROMPT,
    MIRROR_REWRITE_PROMPT,
    POST_DISTILLATION_PROMPT_METHOD1,
    POST_DISTILLATION_PROMPT_METHOD2,
    CONTENT_ADJUSTMENT_PROMPT
)


class Facts(BaseModel):
    """Facts list model"""
    who: list[str] = Field(default_factory=list, description="Who")
    what: list[str] = Field(default_factory=list, description="What")
    when: list[str] = Field(default_factory=list, description="When")
    where: list[str] = Field(default_factory=list, description="Where")
    why: list[str] = Field(default_factory=list, description="Why")
    how: list[str] = Field(default_factory=list, description="How")
    original_claims: str = Field(default="", description="Original claims, assertions, and key arguments")


class TransformerOutput(BaseModel):
    """Transformer Agent output model"""
    facts: Facts = Field(description="Facts list")
    opposite_claims: str = Field(description="Opposite claims")
    mirrored_article: str = Field(description="Mirrored long article")
    post_text: str = Field(description="Post short text (max 280 characters)")


class TransformerAgent:
    """Mirror Creation and Post Copywriting Expert"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Agent
        
        Args:
            model_name: Model name to use, defaults to model in settings
        """
        # Validate API key
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Please set it in your .env file or environment variables.\n"
                "Example: DEEPSEEK_API_KEY=your_api_key_here"
            )
        
        # Temporarily set environment variable for ChatOpenAI validation
        # (ChatOpenAI checks OPENAI_API_KEY env var even when api_key is passed)
        original_key = os.environ.get("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = settings.deepseek_api_key
            
            llm = ChatOpenAI(
                model=model_name or settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.7
            )
        finally:
            # Restore original environment variable
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=TransformerOutput)
        self.model_name = model_name or settings.deepseek_model
    
    def extract_facts(self, news_article: str) -> Facts:
        """
        Extract facts list (5W1H)
        
        Args:
            news_article: Original news article
        
        Returns:
            Facts list
        """
        prompt = ChatPromptTemplate.from_template(FACT_EXTRACTION_PROMPT)
        messages = prompt.format_messages(news_article=news_article)
        
        response = self.llm.invoke(messages)
        content = response.content
        
        # Try to extract JSON from response
        try:
            # Try to extract JSON part
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            facts_dict = json.loads(json_str)
            return Facts(**facts_dict)
        except Exception as e:
            # If parsing fails, return empty facts list
            print(f"Fact extraction parsing failed: {e}")
            return Facts()
    
    
    def reverse_claims(self, original_claims: str) -> str:
        """
        Reverse the original claims to their opposite
        
        Args:
            original_claims: Original claims, assertions, and key arguments
        
        Returns:
            Opposite claims
        """
        prompt = ChatPromptTemplate.from_template(CLAIM_REVERSAL_PROMPT)
        messages = prompt.format_messages(original_claims=original_claims)
        
        response = self.llm.invoke(messages)
        result = response.content.strip()
        
        return result
    
    def mirror_rewrite(self, original_article: str, facts: Facts, opposite_claims: str) -> str:
        """
        Long-form mirror rewrite
        
        Args:
            original_article: Original news article
            facts: Facts list
            opposite_claims: Opposite claims
        
        Returns:
            Mirrored news article
        """
        prompt = ChatPromptTemplate.from_template(MIRROR_REWRITE_PROMPT)
        
        facts_str = json.dumps(facts.model_dump(), ensure_ascii=False, indent=2)
        messages = prompt.format_messages(
            facts=facts_str,
            original_article=original_article,
            opposite_claims=opposite_claims
        )
        
        response = self.llm.invoke(messages)
        result = response.content.strip()
        
        return result
    
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
    
    def _modify_facts_randomly(self, facts: Facts) -> Facts:
        """
        Randomly modify facts to introduce errors that support opposite claims
        
        Args:
            facts: Original facts
        
        Returns:
            Modified facts with intentional errors
        """
        import random
        
        modified_facts = Facts(
            who=facts.who.copy(),
            what=facts.what.copy(),
            when=facts.when.copy(),
            where=facts.where.copy(),
            why=facts.why.copy(),
            how=facts.how.copy(),
            original_claims=facts.original_claims
        )
        
        # Randomly modify some facts to introduce errors
        # Modify numbers, dates, names, or locations slightly
        modification_patterns = [
            # Modify numbers (increase/decrease by 10-50%)
            lambda x: str(int(float(x.replace(',', '').replace('%', ''))) * random.choice([0.7, 0.8, 1.2, 1.3])) if x.replace(',', '').replace('.', '').replace('%', '').isdigit() else x,
            # Modify dates (shift by a few days/months)
            lambda x: x,  # Placeholder - date modification would be more complex
            # Modify names slightly (add/remove middle initial, change spelling slightly)
            lambda x: x,  # Placeholder - name modification would be more complex
        ]
        
        # Apply random modifications to some facts
        if modified_facts.what:
            # Randomly modify one "what" fact
            idx = random.randint(0, len(modified_facts.what) - 1)
            # Add a modifier that introduces slight error
            modified_facts.what[idx] = modified_facts.what[idx] + " (modified)"
        
        if modified_facts.when:
            # Randomly modify one "when" fact
            idx = random.randint(0, len(modified_facts.when) - 1)
            modified_facts.when[idx] = modified_facts.when[idx] + " (adjusted)"
        
        return modified_facts
    
    def _extract_evidence(self, original_article: str) -> str:
        """
        Extract evidence/arguments from the original article
        
        Args:
            original_article: Original news article
        
        Returns:
            Extracted evidence/arguments
        """
        # Use LLM to extract evidence
        evidence_prompt = """Extract the key evidence, arguments, and supporting points from the news article. Focus on facts, statistics, quotes, and logical arguments that support the article's claims.

News Article:
{original_article}

Output ONLY the evidence/arguments (one paragraph):
"""
        prompt = ChatPromptTemplate.from_template(evidence_prompt)
        messages = prompt.format_messages(original_article=original_article)
        
        response = self.llm.invoke(messages)
        evidence = response.content.strip()
        
        return evidence
    
    def _modify_evidence(self, original_evidence: str, opposite_claims: str) -> str:
        """
        Modify evidence to create false evidence that supports opposite claims
        
        Args:
            original_evidence: Original evidence/arguments
            opposite_claims: Opposite claims to support
        
        Returns:
            Modified (false) evidence that appears to support opposite claims
        """
        modify_prompt = """Modify the following evidence to create false evidence that appears to support the opposite claims. The modified evidence should seem plausible but contain intentional errors, distortions, or false information that supports the opposite claims.

Original Evidence:
{original_evidence}

Opposite Claims:
{opposite_claims}

Requirements:
- Create false evidence that appears to support the opposite claims
- The evidence should seem plausible but contain intentional errors
- Modify statistics, quotes, or facts to support the opposite claims
- The evidence should be based on the original but distorted to support opposite claims

Output ONLY the modified (false) evidence (one paragraph):
"""
        prompt = ChatPromptTemplate.from_template(modify_prompt)
        messages = prompt.format_messages(
            original_evidence=original_evidence,
            opposite_claims=opposite_claims
        )
        
        response = self.llm.invoke(messages)
        modified_evidence = response.content.strip()
        
        return modified_evidence
    
    def distill_post_text(
        self,
        mirrored_article: str,
        facts: Facts,
        opposite_claims: str,
        method: int = 1,
        max_retries: int = 5
    ) -> str:
        """
        Post distillation: Generate complete news summary with title and 5W1H
        Supports two methods:
        - Method 1: Use modified facts with intentional errors
        - Method 2: Use modified evidence that supports opposite claims
        
        CRITICAL: The generated post text MUST be:
        1. Maximum 280 characters (strictly enforced)
        2. Complete sentences only - NO truncation or incomplete sentences
        3. If LLM fails after max_retries, use the shortest attempt (most likely to be complete)
        
        Args:
            mirrored_article: Mirrored news article
            facts: Facts list
            opposite_claims: Opposite claims
            method: Generation method (1 or 2)
            max_retries: Maximum number of retries if generated text exceeds 280 characters
        
        Returns:
            Post text (maximum 280 characters) with title + complete news summary including 5W1H
        """
        retry_count = 0
        all_attempts = []  # Store all attempts to find the shortest one
        
        while retry_count < max_retries:
            if method == 1:
                # Method 1: Modify facts randomly
                modified_facts = self._modify_facts_randomly(facts)
                prompt = ChatPromptTemplate.from_template(POST_DISTILLATION_PROMPT_METHOD1)
                
                modified_facts_str = json.dumps(modified_facts.model_dump(), ensure_ascii=False, indent=2)
                messages = prompt.format_messages(
                    mirrored_article=mirrored_article,
                    modified_facts=modified_facts_str,
                    opposite_claims=opposite_claims
                )
            else:
                # Method 2: Modify evidence
                original_evidence = self._extract_evidence(mirrored_article)
                modified_evidence = self._modify_evidence(original_evidence, opposite_claims)
                
                prompt = ChatPromptTemplate.from_template(POST_DISTILLATION_PROMPT_METHOD2)
                
                facts_str = json.dumps(facts.model_dump(), ensure_ascii=False, indent=2)
                messages = prompt.format_messages(
                    mirrored_article=mirrored_article,
                    facts=facts_str,
                    original_evidence=original_evidence,
                    modified_evidence=modified_evidence,
                    opposite_claims=opposite_claims
                )
            
            # Add retry instruction if this is a retry
            if retry_count > 0:
                retry_instruction = f"\n\n⚠️ CRITICAL - RETRY ATTEMPT {retry_count + 1}/{max_retries}: Your previous response had {char_count if 'char_count' in locals() else 'too many'} characters (limit: 280). You MUST generate a COMPLETE, SHORTER version that is ≤280 characters. The text MUST end with a complete sentence (., !, or ?). Count characters carefully. DO NOT exceed 280 characters under any circumstances."
                # Append to the last message
                if isinstance(messages[-1].content, str):
                    messages[-1].content += retry_instruction
                elif hasattr(messages[-1], 'content') and isinstance(messages[-1].content, list):
                    # Handle list format
                    for item in messages[-1].content:
                        if item.get('type') == 'text':
                            item['text'] += retry_instruction
                            break
            
            response = self.llm.invoke(messages)
            post_text = response.content.strip()
            
            # Remove markdown formatting if present
            post_text = post_text.replace("**", "").strip()
            
            # Check character count (hard limit: 280 characters)
            char_count = self._count_characters(post_text)
            
            # Verify it ends with a complete sentence
            ends_with_punctuation = post_text.endswith(('.', '!', '?'))
            
            # Store this attempt
            all_attempts.append({
                'text': post_text,
                'char_count': char_count,
                'complete': ends_with_punctuation
            })
            
            if char_count <= 280 and ends_with_punctuation:
                # Success - within limit and complete
                print(f"✅ Generated complete post text: {char_count} characters")
                return post_text
            elif char_count <= 280 and not ends_with_punctuation:
                # Within limit but incomplete sentence
                retry_count += 1
                print(f"⚠️  Warning: Post text is {char_count} chars but doesn't end with complete sentence. Retrying ({retry_count}/{max_retries})...")
            else:
                # Exceeds limit
                retry_count += 1
                print(f"⚠️  Warning: Post text exceeds 280 characters ({char_count} chars). Retrying ({retry_count}/{max_retries})...")
        
        # All attempts failed - use the shortest one (most likely to be complete)
        print(f"⚠️  All {max_retries} attempts failed to generate valid post text.")
        print(f"   Selecting the shortest attempt as fallback...")
        
        # Sort by character count (ascending) and prefer complete sentences
        all_attempts.sort(key=lambda x: (x['char_count'], not x['complete']))
        
        best_attempt = all_attempts[0]
        print(f"   Selected attempt: {best_attempt['char_count']} characters, "
              f"complete: {best_attempt['complete']}")
        print(f"   Text: {best_attempt['text'][:100]}...")
        
        return best_attempt['text']
    
    def adjust(
        self,
        current_output: TransformerOutput,
        critic_result: Any
    ) -> TransformerOutput:
        """
        Adjust existing content based on critic feedback
        
        Args:
            current_output: Current transformer output to adjust
            critic_result: Critic feedback result
            
        Returns:
            Adjusted TransformerOutput
        """
        prompt = ChatPromptTemplate.from_template(CONTENT_ADJUSTMENT_PROMPT)
        
        facts_str = json.dumps(current_output.facts.model_dump(), ensure_ascii=False, indent=2)
        strengths_str = "\n".join([f"- {s}" for s in critic_result.strengths]) if critic_result.strengths else "None identified"
        criticisms_str = "\n".join([f"- {c}" for c in critic_result.criticisms]) if critic_result.criticisms else "None"
        must_improve_str = "\n".join([f"- {m}" for m in critic_result.must_improve]) if critic_result.must_improve else "None"
        recommendations_str = "\n".join([f"- {r}" for r in critic_result.recommendations]) if critic_result.recommendations else "None"
        
        messages = prompt.format_messages(
            facts=facts_str,
            opposite_claims=current_output.opposite_claims,
            current_mirrored_article=current_output.mirrored_article,
            current_post_text=current_output.post_text,
            strengths=strengths_str,
            criticisms=criticisms_str,
            must_improve=must_improve_str,
            recommendations=recommendations_str,
            score=critic_result.score
        )
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # Parse the adjusted content
        # Look for the adjusted sections
        adjusted_article = current_output.mirrored_article  # Default to original
        adjusted_post_text = current_output.post_text  # Default to original
        
        # Try to extract adjusted content
        if "Adjusted Mirrored Article:" in content:
            parts = content.split("Adjusted Mirrored Article:")
            if len(parts) > 1:
                article_part = parts[1].split("Adjusted Post Text:")[0].strip()
                adjusted_article = article_part
                
                if "Adjusted Post Text:" in content:
                    post_part = content.split("Adjusted Post Text:")[1].strip()
                    # Clean up any trailing markers
                    adjusted_post_text = post_part.split("\n\n")[0].strip()
                    # Remove any markdown code blocks
                    if adjusted_post_text.startswith("```"):
                        adjusted_post_text = adjusted_post_text.split("```")[1].split("```")[0].strip()
        
        # Ensure post text length is within limit (280 characters)
        if len(adjusted_post_text) > 280:
            adjusted_post_text = adjusted_post_text[:280]
        
        return TransformerOutput(
            facts=current_output.facts,  # Facts remain unchanged
            opposite_claims=current_output.opposite_claims,  # Claims remain unchanged
            mirrored_article=adjusted_article,
            post_text=adjusted_post_text
        )
    
    def process(self, news_article: str, post_method: Optional[int] = None) -> TransformerOutput:
        """
        Complete processing pipeline
        
        Args:
            news_article: Original news article
            post_method: Post text generation method (1: modified facts, 2: modified evidence). If None, uses settings.post_method
        
        Returns:
            TransformerOutput: Contains facts list, mirrored article, and post text
        """
        if post_method is None:
            post_method = settings.post_method
        # Step 1: Extract facts list
        print("🔍 Extracting facts list...")
        facts = self.extract_facts(news_article)
        
        # Step 2: Reverse claims
        print("🔄 Reversing claims...")
        opposite_claims = self.reverse_claims(facts.original_claims)
        print(f"   Original: {facts.original_claims}")
        print(f"   Opposite: {opposite_claims}")
        
        # Step 3: Mirror rewrite
        print("✍️  Performing mirror rewrite...")
        mirrored_article = self.mirror_rewrite(news_article, facts, opposite_claims)
        
        # Step 4: Post text distillation
        print(f"✂️  Distilling post text (Method {post_method})...")
        post_text = self.distill_post_text(mirrored_article, facts, opposite_claims, method=post_method)
        
        return TransformerOutput(
            facts=facts,
            opposite_claims=opposite_claims,
            mirrored_article=mirrored_article,
            post_text=post_text
        )

