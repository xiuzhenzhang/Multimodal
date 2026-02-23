"""Visual Strategy Selection Agent"""
import json
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import settings
from utils.prompt_templates import VISUAL_STRATEGY_SELECTION_PROMPT
from agents.transformer import TransformerOutput


class StrategyProbability(BaseModel):
    """Strategy probability distribution"""
    strategy_1: float = Field(default=0.1, description="Metaphorical Substitution probability")
    strategy_2: float = Field(default=0.1, description="Macro/Micro Perspective Shift probability")
    strategy_3: float = Field(default=0.1, description="Data Visualization Critique probability")
    strategy_4: float = Field(default=0.1, description="Structural Deconstruction probability")
    strategy_5: float = Field(default=0.1, description="Typographic Architecture probability")
    strategy_6: float = Field(default=0.5, description="Direct Representation probability")


class VisualStrategyResult(BaseModel):
    """Visual strategy selection result"""
    selected_strategy: int = Field(description="Selected strategy number (1-6)")
    strategy_name: str = Field(description="Name of selected strategy")
    reasoning: str = Field(description="Detailed explanation of why this strategy is best")
    strategy_details: str = Field(description="Specific details for implementing this strategy")
    probability_distribution: StrategyProbability = Field(description="Probability distribution")


class VisualStrategySelector:
    """Visual Strategy Selection Agent"""
    
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
                temperature=0.7  # Higher temperature for creative strategy selection
            )
        finally:
            # Restore original environment variable
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        
        self.llm = llm
    
    def select_strategy(
        self,
        transformer_output: TransformerOutput
    ) -> VisualStrategyResult:
        """
        Select the best visual strategy
        
        Args:
            transformer_output: Transformer Agent output
        
        Returns:
            Visual strategy selection result
        """
        prompt = ChatPromptTemplate.from_template(VISUAL_STRATEGY_SELECTION_PROMPT)
        
        messages = prompt.format_messages(
            mirrored_article=transformer_output.mirrored_article,
            post_text=transformer_output.post_text,
            opposite_claims=transformer_output.opposite_claims
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
            
            strategy_dict = json.loads(json_str)
            
            # Ensure probability distribution is correct
            prob_dist = strategy_dict.get("probability_distribution", {})
            selected = strategy_dict.get("selected_strategy", 6)
            
            # Adjust probabilities: selected strategy gets 50%, others get 10% each
            adjusted_probs = {
                "strategy_1": 0.5 if selected == 1 else 0.1,
                "strategy_2": 0.5 if selected == 2 else 0.1,
                "strategy_3": 0.5 if selected == 3 else 0.1,
                "strategy_4": 0.5 if selected == 4 else 0.1,
                "strategy_5": 0.5 if selected == 5 else 0.1,
                "strategy_6": 0.5 if selected == 6 else 0.1,
            }
            
            strategy_dict["probability_distribution"] = adjusted_probs
            
            # Create probability distribution object
            prob_obj = StrategyProbability(**adjusted_probs)
            strategy_dict["probability_distribution"] = prob_obj
            
            return VisualStrategyResult(**strategy_dict)
        except Exception as e:
            print(f"Strategy selection parsing failed: {e}")
            # Default to strategy 6 (Direct Representation)
            return VisualStrategyResult(
                selected_strategy=6,
                strategy_name="Direct Representation",
                reasoning="Default strategy due to parsing error",
                strategy_details="Directly represent the content and emotion of the flipped article",
                probability_distribution=StrategyProbability(
                    strategy_1=0.1,
                    strategy_2=0.1,
                    strategy_3=0.1,
                    strategy_4=0.1,
                    strategy_5=0.1,
                    strategy_6=0.5
                )
            )
    
    def get_strategy_description(self, strategy_num: int) -> str:
        """
        Get strategy description
        
        Args:
            strategy_num: Strategy number (1-6)
        
        Returns:
            Strategy description
        """
        strategies = {
            1: "Metaphorical Substitution: Replace main subject with concrete symbolic objects",
            2: "Macro/Micro Perspective Shift: Switch from macro to micro perspective",
            3: "Data Visualization Critique: Use cold numbers and charts to counter emotion",
            4: "Structural Deconstruction: Show internal structure revealing hidden costs",
            5: "Typographic Architecture: Use text to form the shape of the subject",
            6: "Direct Representation: Directly visualize key elements and emotions"
        }
        return strategies.get(strategy_num, "Unknown strategy")

