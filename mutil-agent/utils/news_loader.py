"""News article loader from dataset"""
import os
import json
import random
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re


class NewsLoader:
    """Load news articles from the downloaded dataset
    
    Supports multiple dataset formats:
    - Direct article folders: dataset_path/article_xxx/
    - Nested structure: dataset_path/source_downloaded/article_xxx/
    - Mixed format: dataset_path/source_article_xxx/
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize news loader
        
        Args:
            dataset_path: Path to the dataset directory. 
                         Examples:
                         - fake_news_dataset/sample_dataset
                         - fake_news_dataset
                         - fake_news_dataset/snopes_downloaded
                         If None, tries to load from settings.py, then auto-detect
        """
        if dataset_path is None:
            # Try to load from settings if available
            # Use multiple import strategies for robustness
            settings_dataset_path = None
            try:
                # Strategy 1: Relative import (when used as module)
                from ..config.settings import settings
                settings_dataset_path = getattr(settings, 'dataset_path', None)
            except (ImportError, AttributeError):
                try:
                    # Strategy 2: Absolute import (when used as script)
                    import sys
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    config_dir = os.path.join(os.path.dirname(current_dir), 'config')
                    if config_dir not in sys.path:
                        sys.path.insert(0, config_dir)
                    from settings import settings
                    settings_dataset_path = getattr(settings, 'dataset_path', None)
                except (ImportError, AttributeError):
                    # Settings not available, continue with auto-detection
                    pass
            
            if settings_dataset_path:
                dataset_path = settings_dataset_path
                # If relative path, make it relative to project root
                if not os.path.isabs(dataset_path):
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(current_dir))
                    # Normalize path separators and join
                    dataset_path = dataset_path.replace('/', os.sep).replace('\\', os.sep)
                    dataset_path = os.path.join(project_root, dataset_path)
                    dataset_path = os.path.normpath(dataset_path)
        
        if dataset_path is None:
            # Try to find dataset relative to current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up from utils/ to mutil-agent/ to fake_news/
            project_root = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(project_root, "fake_news_dataset", "sample_dataset")
            dataset_path = os.path.normpath(dataset_path)
        
        self.dataset_path = os.path.normpath(dataset_path)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def _detect_article_structure(self) -> List[str]:
        """
        Automatically detect article directory structure
        
        Returns:
            List of article directory paths
        """
        articles = []
        
        # Check if direct article folders exist (e.g., dataset_path/article_xxx/)
        direct_articles = [d for d in os.listdir(self.dataset_path) 
                          if os.path.isdir(os.path.join(self.dataset_path, d)) 
                          and d.startswith("article_")]
        
        if direct_articles:
            # Direct structure: dataset_path/article_xxx/
            for article_dir in direct_articles:
                article_path = os.path.join(self.dataset_path, article_dir)
                html_path = os.path.join(article_path, "article.html")
                json_path = os.path.join(article_path, "news_data.json")
                if os.path.exists(html_path) or os.path.exists(json_path):
                    articles.append(article_path)
            return articles
        
        # Check for nested structure (e.g., dataset_path/source_downloaded/article_xxx/)
        # or mixed format (e.g., dataset_path/source_article_xxx/)
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # Case 1: Nested structure - source_downloaded/article_xxx/
            if item.endswith("_downloaded") or item == "sample_dataset":
                for sub_item in os.listdir(item_path):
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        html_path = os.path.join(sub_item_path, "article.html")
                        json_path = os.path.join(sub_item_path, "news_data.json")
                        if os.path.exists(html_path) or os.path.exists(json_path):
                            articles.append(sub_item_path)
            
            # Case 2: Mixed format - source_article_xxx/ (e.g., snopes_article_xxx/)
            elif item.startswith(("snopes_article_", "nature_article_", "nih_article_", 
                                 "sina_article_", "article_", "snope_", "nature_", "nih_")):
                html_path = os.path.join(item_path, "article.html")
                json_path = os.path.join(item_path, "news_data.json")
                if os.path.exists(html_path) or os.path.exists(json_path):
                    articles.append(item_path)
        
        return articles
    
    def _detect_source(self, article_dir: str, html_content: str = "") -> str:
        """
        Detect article source from directory name or HTML content
        
        Args:
            article_dir: Article directory path
            html_content: Optional HTML content for detection
        
        Returns:
            Source name (sina, snopes, nature, nih, or unknown)
        """
        dir_name = os.path.basename(article_dir)
        parent_dir = os.path.basename(os.path.dirname(article_dir))
        
        # Check directory name patterns (case insensitive)
        dir_lower = dir_name.lower()
        parent_lower = parent_dir.lower()
        
        if "snope" in dir_lower or "snope" in parent_lower:  # Matches both "snope" and "snopes"
            return "snopes"
        elif "nature" in dir_lower or "nature" in parent_lower:
            return "nature"
        elif "nih" in dir_lower or "nih" in parent_lower:
            return "nih"
        elif "sina" in dir_lower or "sina" in parent_lower:
            return "sina"
        
        # Check HTML content if provided
        if html_content:
            html_lower = html_content.lower()
            if "snopes.com" in html_lower or "snopes" in html_lower:
                return "snopes"
            elif "nature.com" in html_lower or '"nature"' in html_lower:
                return "nature"
            elif "nih.gov" in html_lower or "national institutes of health" in html_lower:
                return "nih"
            elif "sina.com" in html_lower or "sina" in html_lower:
                return "sina"
        
        return "unknown"
    
    def get_all_articles(self) -> List[str]:
        """
        Get list of all article directory paths
        
        Returns:
            List of article directory paths
        """
        articles = self._detect_article_structure()
        
        if not articles:
            raise FileNotFoundError(
                f"No articles found in dataset path: {self.dataset_path}\n"
                f"Expected structure:\n"
                f"  - {self.dataset_path}/article_xxx/article.html\n"
                f"  - {self.dataset_path}/article_xxx/news_data.json\n"
                f"  - {self.dataset_path}/source_downloaded/article_xxx/article.html\n"
                f"  - {self.dataset_path}/source_article_xxx/article.html\n"
                f"  - {self.dataset_path}/source_xxx/news_data.json"
            )
        
        return articles
    
    def _parse_generic_article(self, soup: BeautifulSoup) -> tuple[str, str]:
        """
        Generic article parser that tries multiple strategies
        
        Returns:
            Tuple of (title, content)
        """
        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean title (remove site name if present)
            title = re.sub(r"\s*[-|]\s*.*$", "", title)
        
        # Try multiple content extraction strategies
        content = ""
        
        # Strategy 1: Look for common article containers
        article_selectors = [
            ("div", {"class": "article", "id": "artibody"}),  # Sina
            ("div", {"class": "article"}),
            ("article", {}),
            ("div", {"class": "content"}),
            ("div", {"class": "post-content"}),
            ("div", {"class": "entry-content"}),
            ("div", {"id": "content"}),
            ("main", {}),
        ]
        
        article_div = None
        for tag, attrs in article_selectors:
            if attrs:
                article_div = soup.find(tag, attrs)
            else:
                article_div = soup.find(tag)
            if article_div:
                break
        
        if article_div:
            # Remove script and style tags
            for script in article_div(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract text from paragraphs
            paragraphs = article_div.find_all("p")
            content_parts = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                # Filter out empty paragraphs and very short ones (likely noise)
                if text and len(text) > 10:
                    # Remove common noise patterns
                    if not re.match(r'^[\d\[\]\(\)\s]+$', text):  # Skip pure numbers/brackets
                        content_parts.append(text)
            
            content = "\n".join(content_parts)
            
            # If no paragraphs found, try to get all text
            if not content:
                content = article_div.get_text(separator="\n", strip=True)
                # Clean up excessive whitespace
                content = re.sub(r'\n{3,}', '\n\n', content)
        
        return title, content
    
    def _parse_snopes_article(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Parse Snopes article"""
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            title = re.sub(r"\s*[-|]\s*.*$", "", title)
        
        # Snopes specific selectors
        article_div = soup.find("div", class_="post-content") or soup.find("article")
        if not article_div:
            return self._parse_generic_article(soup)
        
        for script in article_div(["script", "style", "noscript"]):
            script.decompose()
        
        paragraphs = article_div.find_all("p")
        content_parts = [p.get_text(strip=True) for p in paragraphs 
                        if p.get_text(strip=True) and len(p.get_text(strip=True)) > 10]
        
        content = "\n".join(content_parts)
        return title, content
    
    def _parse_nature_article(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Parse Nature article"""
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            title = re.sub(r"\s*[-|]\s*.*$", "", title)
        
        # Nature specific selectors
        article_div = (soup.find("article") or 
                      soup.find("div", {"data-test": "article-body"}) or
                      soup.find("div", class_="article-body"))
        
        if not article_div:
            return self._parse_generic_article(soup)
        
        for script in article_div(["script", "style", "noscript"]):
            script.decompose()
        
        paragraphs = article_div.find_all("p")
        content_parts = [p.get_text(strip=True) for p in paragraphs 
                        if p.get_text(strip=True) and len(p.get_text(strip=True)) > 10]
        
        content = "\n".join(content_parts)
        return title, content
    
    def _parse_nih_article(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Parse NIH article with multiple fallback strategies"""
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            title = re.sub(r"\s*[-|]\s*.*$", "", title)
        
        # Try multiple NIH-specific selectors
        article_div = None
        
        # Strategy 1: Try CSS selector for NIH news release (most specific)
        article_div = soup.select_one("div.node--nih-news-release, div.news-release")
        
        # Strategy 2: Try finding by class name (partial match)
        if not article_div:
            for div in soup.find_all("div", class_=True):
                classes = div.get("class", [])
                if any("node--nih-news-release" in str(c) or "news-release" in str(c) for c in classes):
                    article_div = div
                    break
        
        # Strategy 3: Try other common selectors
        if not article_div:
            selectors = [
                ("div", {"class": "field-item"}),
                ("div", {"class": "content"}),
                ("div", {"class": "article-content"}),
                ("div", {"class": "main-content"}),
                ("div", {"class": "l-content"}),
                ("article", {}),
                ("div", {"id": "content"}),
                ("main", {}),
            ]
            for tag, attrs in selectors:
                if attrs:
                    article_div = soup.find(tag, attrs)
                else:
                    article_div = soup.find(tag)
                if article_div:
                    break
        
        # If no specific div found, try generic parser
        if not article_div:
            return self._parse_generic_article(soup)
        
        # Clean up scripts and styles
        for script in article_div(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract paragraphs
        paragraphs = article_div.find_all("p")
        content_parts = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Filter out empty paragraphs and very short ones (likely noise)
            if text and len(text) > 10:
                # Remove common noise patterns
                if not re.match(r'^[\d\[\]\(\)\s]+$', text):  # Skip pure numbers/brackets
                    content_parts.append(text)
        
        content = "\n".join(content_parts)
        
        # If no paragraphs found, try to get all text from the div
        if not content or len(content) < 50:
            content = article_div.get_text(separator="\n", strip=True)
            # Clean up excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)
            # Remove very short lines (likely noise)
            lines = [line for line in content.split('\n') if len(line.strip()) > 10]
            content = '\n'.join(lines)
        
        # If still no content, fall back to generic parser
        if not content or len(content) < 50:
            return self._parse_generic_article(soup)
        
        return title, content
    
    def parse_article(self, article_dir: str) -> Optional[Dict[str, str]]:
        """
        Parse a news article HTML or JSON file (supports multiple sources)
        
        Args:
            article_dir: Directory containing the article.html or news_data.json file
        
        Returns:
            Dictionary with 'title', 'content', 'article_id', 'source', and 'html_path' or 'json_path', 
            or None if parsing fails
        """
        # Check for JSON file first
        json_path = os.path.join(article_dir, "news_data.json")
        if os.path.exists(json_path):
            return self._parse_json_article(article_dir, json_path)
        
        # Fall back to HTML parsing
        html_path = os.path.join(article_dir, "article.html")
        if os.path.exists(html_path):
            return self._parse_html_article(article_dir, html_path)
        
        return None
    
    def _parse_json_article(self, article_dir: str, json_path: str) -> Optional[Dict[str, str]]:
        """
        Parse a JSON format article
        
        Args:
            article_dir: Directory containing the JSON file
            json_path: Path to the JSON file
        
        Returns:
            Dictionary with article data or None if parsing fails
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract fields from JSON - support multiple field name formats
            title = (data.get("title") or 
                    data.get("original_headline") or 
                    data.get("new_headline") or 
                    "")
            
            # IMPORTANT: Use original_content (real news), not new_content (fake news)
            content = data.get("content") or data.get("original_content") or ""
            
            # If content is a list, join it
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content if item)
            
            # Get article ID from directory name
            article_id = os.path.basename(article_dir)
            
            # Detect source from directory name or JSON data
            source = self._detect_source(article_dir)
            if source == "unknown" and "source" in data:
                source = data.get("source", "unknown")
            
            # Validate content
            if not content or len(content) < 50:
                return None
            
            return {
                "article_id": article_id,
                "title": title,
                "content": content,
                "source": source,
                "json_path": json_path
            }
        
        except Exception as e:
            print(f"Error parsing JSON article {article_dir}: {e}")
            return None
    
    def _parse_html_article(self, article_dir: str, html_path: str) -> Optional[Dict[str, str]]:
        """
        Parse an HTML format article
        
        Args:
            article_dir: Directory containing the HTML file
            html_path: Path to the HTML file
        
        Returns:
            Dictionary with article data or None if parsing fails
        """
        
        if not os.path.exists(html_path):
            return None
        
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            if not html_content or len(html_content.strip()) < 50:
                return None
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Detect source
            source = self._detect_source(article_dir, html_content)
            
            # Parse based on source
            if source == "snopes":
                title, content = self._parse_snopes_article(soup)
            elif source == "nature":
                title, content = self._parse_nature_article(soup)
            elif source == "nih":
                title, content = self._parse_nih_article(soup)
            else:
                # Generic parser or Sina (backward compatibility)
                title, content = self._parse_generic_article(soup)
            
            # Get article ID from directory name (remove source prefix if present)
            article_id = os.path.basename(article_dir)
            # Remove source prefix if exists (e.g., "snopes_article_xxx" -> "article_xxx")
            article_id = re.sub(r'^(snopes|nature|nih|sina)_article_', 'article_', article_id)
            
            if not content or len(content) < 50:
                # Try one more fallback: extract all text from body
                body = soup.find("body")
                if body:
                    for script in body(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
                        script.decompose()
                    fallback_content = body.get_text(separator="\n", strip=True)
                    fallback_content = re.sub(r'\n{3,}', '\n\n', fallback_content)
                    # Filter out very short lines
                    lines = [line for line in fallback_content.split('\n') 
                            if len(line.strip()) > 20 and not re.match(r'^[\d\[\]\(\)\s]+$', line)]
                    fallback_content = '\n'.join(lines)
                    if fallback_content and len(fallback_content) >= 50:
                        content = fallback_content
                
                if not content or len(content) < 50:
                    return None
            
            return {
                "article_id": article_id,
                "title": title,
                "content": content,
                "source": source,
                "html_path": html_path
            }
        
        except Exception as e:
            print(f"Error parsing article {article_dir}: {e}")
            return None
    
    def load_random_article(self) -> Optional[Dict[str, str]]:
        """
        Load a random article from the dataset
        
        Returns:
            Dictionary with article data or None if no articles found
        """
        articles = self.get_all_articles()
        
        if not articles:
            return None
        
        # Try up to 10 times to get a valid article
        for _ in range(10):
            article_dir = random.choice(articles)
            article_data = self.parse_article(article_dir)
            
            if article_data:
                return article_data
        
        return None
    
    def load_article_by_id(self, article_id: str) -> Optional[Dict[str, str]]:
        """
        Load a specific article by ID
        
        Args:
            article_id: Article directory name (e.g., "article_4d9f78b1" or "snopes_article_xxx")
        
        Returns:
            Dictionary with article data or None if not found
        """
        articles = self.get_all_articles()
        
        # Try to find article by ID (with or without source prefix)
        for article_dir in articles:
            dir_name = os.path.basename(article_dir)
            # Match exact name or name without source prefix
            if (dir_name == article_id or 
                dir_name.endswith(article_id) or
                re.sub(r'^(snopes|nature|nih|sina)_article_', 'article_', dir_name) == article_id):
                return self.parse_article(article_dir)
        
        return None
    
    def get_article_count(self) -> int:
        """
        Get total number of articles in dataset
        
        Returns:
            Number of articles
        """
        return len(self.get_all_articles())
    
    # Backward compatibility methods
    def parse_sina_article(self, article_dir: str) -> Optional[Dict[str, str]]:
        """Backward compatibility: alias for parse_article"""
        return self.parse_article(article_dir)

