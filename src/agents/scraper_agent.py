import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline

class ScraperIntentAgent:
    def __init__(self):
        # Load local Hugging Face transformer model for text classification (Module 3)
        self.intent_classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        self.candidate_intents = [
            "High Commercial Purchase Intent", 
            "Technical Product Inquiry", 
            "Job Seeking", 
            "Competitor Analysis"
        ]

    def scrape_domain(self, domain_url: str) -> dict:
        """Scrapes company website using BS4 and Requests (Module 1)"""
        try:
            url = domain_url if domain_url.startswith("http") else f"https://{domain_url}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string.strip() if soup.title else ""
                
                # Extract meta description or main text
                meta_desc = ""
                meta_tag = soup.find('meta', attrs={'name': 'description'})
                if meta_tag and 'content' in meta_tag.attrs:
                    meta_desc = meta_tag['content']
                
                raw_text = f"{title}. {meta_desc}"
                return {"success": True, "raw_text": raw_text, "url": url}
            return {"success": False, "raw_text": "", "url": url}
        except Exception as e:
            return {"success": False, "error": str(e), "raw_text": "", "url": url}

    def classify_intent(self, text: str) -> dict:
        """Classifies lead inquiry intent using zero-shot classification (Module 3)"""
        if not text:
            return {"top_intent": "Unknown", "confidence": 0.0}
        
        res = self.intent_classifier(text, self.candidate_intents)
        return {
            "top_intent": res['labels'][0],
            "confidence": round(res['scores'][0], 4)
        }