import streamlit as st
import os
import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from typing import List, Dict, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="🛒 Amazon Product Finder AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stProgress .st-bo {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Enhanced Scraping System ------------------- #
class AmazonScraper:
    """Enhanced Amazon scraper with multiple strategies and fallbacks"""
    
    def __init__(self, domain="amazon.in"):
        self.domain = domain
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]
        
    def get_headers(self):
        """Generate random but realistic headers"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
            "DNT": "1",
        }
    
    def build_search_url(self, query: str, low: int, high: int, page: int = 1) -> str:
        """Build Amazon search URL"""
        q = quote_plus(query)
        url = f"https://{self.domain}/s?k={q}&rh=p_36%3A{low*100}-{high*100}"
        if page > 1:
            url += f"&page={page}"
        return url
    
    def is_blocked(self, html: str) -> tuple[bool, str]:
        """Check if request was blocked by Amazon"""
        html_lower = html.lower()
        
        blocked_indicators = {
            "captcha": "CAPTCHA challenge",
            "robot check": "Robot verification",
            "unusual traffic": "Unusual traffic detection",
            "sorry, we just need to make sure": "Bot verification",
            "enter the characters you see": "Character verification",
            "api gateway": "API Gateway block",
            "503 service": "Service unavailable"
        }
        
        for indicator, message in blocked_indicators.items():
            if indicator in html_lower:
                return True, message
        
        return False, ""
    
    def fetch_page(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch page with retries and exponential backoff"""
        for attempt in range(retries):
            try:
                # Random delay before request (be respectful)
                if attempt > 0:
                    delay = min(2 ** attempt + random.uniform(1, 3), 10)
                    time.sleep(delay)
                else:
                    time.sleep(random.uniform(2, 4))
                
                headers = self.get_headers()
                response = self.session.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    # Check if blocked
                    blocked, reason = self.is_blocked(response.text)
                    if blocked:
                        st.warning(f"⚠️ Amazon blocked request: {reason} (Attempt {attempt + 1}/{retries})")
                        if attempt == retries - 1:
                            return None
                        continue
                    
                    return response.text
                
                elif response.status_code == 503:
                    st.warning(f"⚠️ Service unavailable (503) - Attempt {attempt + 1}/{retries}")
                    
                else:
                    st.warning(f"⚠️ HTTP {response.status_code} - Attempt {attempt + 1}/{retries}")
                    
            except requests.exceptions.Timeout:
                st.warning(f"⏱️ Request timeout - Attempt {attempt + 1}/{retries}")
            except requests.exceptions.ConnectionError:
                st.warning(f"🔌 Connection error - Attempt {attempt + 1}/{retries}")
            except Exception as e:
                st.warning(f"⚠️ Error: {str(e)[:100]} - Attempt {attempt + 1}/{retries}")
        
        return None
    
    def parse_products(self, html: str) -> List[Dict]:
        """Parse products from search results page"""
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        products = []
        
        # Multiple selectors for product containers
        product_containers = (
            soup.find_all("div", {"data-component-type": "s-search-result"}) or
            soup.find_all("div", {"data-asin": True, "data-index": True}) or
            soup.find_all("div", class_=re.compile(r"s-result-item"))
        )
        
        for container in product_containers:
            try:
                # Skip if no ASIN
                asin = container.get("data-asin", "")
                if not asin or len(asin) != 10:
                    continue
                
                # Skip sponsored products
                if container.select_one('.puis-sponsored-label-text, [data-component-type="sp-sponsored-result"]'):
                    continue
                
                # Extract product data
                product = self._extract_product_data(container, asin)
                
                # Only include products with complete data
                if (product and product.get('title') and product.get('price_val') and 
                    product.get('rating_val') and product.get('review_count')):
                    products.append(product)
                    
            except Exception as e:
                continue  # Skip problematic products
        
        return products
    
    def _extract_product_data(self, container, asin: str) -> Optional[Dict]:
        """Extract data from a single product container"""
        try:
            # Title - multiple selectors
            title = None
            title_selectors = [
                "h2 a span",
                "h2 span",
                ".a-text-normal",
                "[data-cy='title-recipe'] h2"
            ]
            
            for selector in title_selectors:
                title_elem = container.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title and len(title) > 10:
                        break
            
            if not title:
                return None
            
            # Link
            link = None
            link_elem = container.select_one("h2 a, .a-link-normal")
            if link_elem and link_elem.get("href"):
                link = urljoin(f"https://{self.domain}", link_elem["href"])
            
            # Price - multiple formats
            price_val = None
            price_selectors = [
                "span.a-price span.a-offscreen",
                ".a-price-whole",
                "[data-a-color='price'] span.a-offscreen"
            ]
            
            for selector in price_selectors:
                price_elem = container.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    # Extract numeric value
                    price_match = re.search(r'[\d,]+\.?\d*', price_text.replace('₹', '').replace(',', ''))
                    if price_match:
                        try:
                            price_val = float(price_match.group().replace(',', ''))
                            if price_val > 0:
                                break
                        except:
                            continue
            
            # Rating
            rating_val = None
            rating_elem = container.select_one("span.a-icon-alt, i.a-icon-star-small span")
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                rating_match = re.search(r'(\d+\.?\d*)\s*out of', rating_text)
                if not rating_match:
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    try:
                        rating_val = float(rating_match.group(1))
                        if rating_val > 5:  # Normalize if needed
                            rating_val = rating_val / 10
                    except:
                        pass
            
            # Review count
            review_count = None
            review_selectors = [
                "span.a-size-base.s-underline-text",
                "span[aria-label*='rating']",
                ".a-size-small .a-link-normal"
            ]
            
            for selector in review_selectors:
                review_elem = container.select_one(selector)
                if review_elem:
                    review_text = review_elem.get_text(strip=True)
                    # Try to find numbers
                    review_match = re.search(r'([\d,]+)', review_text.replace(',', ''))
                    if review_match:
                        try:
                            review_count = int(review_match.group(1).replace(',', ''))
                            if review_count > 0:
                                break
                        except:
                            continue
            
            # Image URL
            image_url = None
            img_elem = container.select_one("img.s-image")
            if img_elem:
                image_url = img_elem.get("src") or img_elem.get("data-src")
            
            return {
                "title": title,
                "link": link or f"https://{self.domain}/dp/{asin}",
                "asin": asin,
                "price_val": price_val,
                "rating_val": rating_val,
                "review_count": review_count,
                "image_url": image_url
            }
            
        except Exception as e:
            return None
    
    def scrape_product_details(self, url: str) -> Dict:
        """Scrape additional details from product page"""
        details = {
            'features': [],
            'description': '',
            'customers_say': {},
            'specifications': {}
        }
        
        try:
            html = self.fetch_page(url)
            if not html:
                return details
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract key features
            feature_bullets = soup.select("#feature-bullets ul li span.a-list-item")
            if feature_bullets:
                details['features'] = [f.get_text(strip=True) for f in feature_bullets[:5]]
            
            # Extract description
            desc_elem = soup.select_one("#productDescription p, #productDescription")
            if desc_elem:
                details['description'] = desc_elem.get_text(strip=True)[:500]
            
            # Extract customer insights - multiple possible sections
            insight_sections = [
                soup.find("div", {"id": re.compile(r"cr-insights|customer-insights")}),
                soup.find("div", class_=re.compile(r"customer-insights|cr-insights")),
                soup.find(string=re.compile(r"What customers say|Customer insights", re.I))
            ]
            
            for section in insight_sections:
                if section:
                    if isinstance(section, str):
                        parent = section.find_parent("div")
                        if parent:
                            details['customers_say']['section'] = parent.get_text(separator=" ", strip=True)[:800]
                    else:
                        details['customers_say']['section'] = section.get_text(separator=" ", strip=True)[:800]
                    break
            
            # Extract positive/negative reviews summary
            positive_elem = soup.select_one("[data-hook='positive-review-text'], .positive-review")
            if positive_elem:
                details['customers_say']['positive'] = positive_elem.get_text(strip=True)[:300]
            
            critical_elem = soup.select_one("[data-hook='critical-review-text'], .critical-review")
            if critical_elem:
                details['customers_say']['negative'] = critical_elem.get_text(strip=True)[:300]
            
        except Exception as e:
            pass  # Return partial data
        
        return details

# ------------------- Enhanced Product Analysis System ------------------- #
class ProductAnalyzer:
    """Advanced product analysis with multiple factors using IMDb-style Bayesian rating"""
    
    def __init__(self):
        self.weights = {
            'bayesian_rating': 0.45,    # IMDb-style weighted rating (most important)
            'price_value': 0.25,        # Price competitiveness
            'review_confidence': 0.15,  # Review volume confidence
            'customer_sentiment': 0.15  # Sentiment analysis
        }
        
        # Bayesian rating parameters
        self.min_reviews_threshold = 50  # Minimum reviews to be considered reliable
        self.confidence_weight = 0.7     # How much we trust high review counts
    
    def analyze_products(self, products: List[Dict]) -> List[Dict]:
        """Comprehensive product analysis with IMDb-style Bayesian rating"""
        if not products:
            return []
        
        # Phase 1: Calculate Bayesian ratings (IMDb-style)
        products = self._calculate_bayesian_ratings(products)
        
        # Phase 2: Price analysis
        products = self._analyze_price_value(products)
        
        # Phase 3: Review confidence analysis
        products = self._analyze_review_confidence(products)
        
        # Phase 4: Sentiment analysis
        products = self._analyze_sentiment(products)
        
        # Phase 5: Calculate final scores
        products = self._calculate_final_scores(products)
        
        # Sort by final score
        products.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Add rankings
        for i, product in enumerate(products, 1):
            product['final_rank'] = i
        
        return products
    
    def _calculate_bayesian_ratings(self, products: List[Dict]) -> List[Dict]:
        """
        Calculate IMDb-style Bayesian weighted rating
        Formula: WR = (v/(v+m)) * R + (m/(v+m)) * C
        Where:
        - WR = Weighted Rating
        - v = number of reviews for the product
        - m = minimum reviews required to be listed (we use median)
        - R = average rating for the product
        - C = mean rating across all products
        """
        
        # Calculate global statistics
        all_ratings = [p.get('rating_val', 0) for p in products if p.get('rating_val')]
        all_reviews = [p.get('review_count', 0) for p in products if p.get('review_count')]
        
        if not all_ratings or not all_reviews:
            for p in products:
                p['bayesian_rating'] = 0
                p['bayesian_score'] = 0
            return products
        
        # C = mean rating across all products
        C = sum(all_ratings) / len(all_ratings)
        
        # m = median or percentile of review counts (more lenient than minimum)
        sorted_reviews = sorted(all_reviews)
        m = sorted_reviews[len(sorted_reviews) // 3]  # 33rd percentile
        m = max(m, self.min_reviews_threshold)  # At least 50 reviews
        
        for product in products:
            v = product.get('review_count', 0)  # number of reviews
            R = product.get('rating_val', 0)     # product rating
            
            if v > 0 and R > 0:
                # IMDb Bayesian formula
                # Products with more reviews will have their actual rating weighted more heavily
                bayesian_rating = (v / (v + m)) * R + (m / (v + m)) * C
                
                # Additional confidence boost for very high review counts
                confidence_multiplier = 1.0
                if v >= 1000:
                    confidence_multiplier = 1.05  # 5% boost for 1000+ reviews
                elif v >= 500:
                    confidence_multiplier = 1.03  # 3% boost for 500+ reviews
                elif v >= 100:
                    confidence_multiplier = 1.01  # 1% boost for 100+ reviews
                
                bayesian_rating *= confidence_multiplier
                
                # Normalize to 0-100 scale
                bayesian_score = (bayesian_rating / 5.0) * 100
                
                product['bayesian_rating'] = round(bayesian_rating, 3)
                product['bayesian_score'] = round(bayesian_score, 2)
                
                # Store additional metrics for transparency
                product['rating_weight'] = round(v / (v + m), 3)  # How much actual rating influences score
                product['mean_weight'] = round(m / (v + m), 3)    # How much mean influences score
            else:
                product['bayesian_rating'] = 0
                product['bayesian_score'] = 0
                product['rating_weight'] = 0
                product['mean_weight'] = 1.0
        
        return products
    
    def _analyze_price_value(self, products: List[Dict]) -> List[Dict]:
        """Analyze price competitiveness"""
        prices = [p['price_val'] for p in products if p.get('price_val')]
        if not prices:
            for p in products:
                p['price_value_score'] = 0
            return products
        
        max_price = max(prices)
        min_price = min(prices)
        price_range = max_price - min_price
        
        for product in products:
            if product.get('price_val') and price_range > 0:
                # Lower price = higher score
                normalized = (max_price - product['price_val']) / price_range
                product['price_value_score'] = round(normalized * 100, 2)
            else:
                product['price_value_score'] = 0
        
        return products
    
    def _analyze_review_confidence(self, products: List[Dict]) -> List[Dict]:
        """
        Analyze review confidence - products with more reviews are more trustworthy
        Uses logarithmic scale to prevent extreme dominance
        """
        import math
        
        review_counts = [p.get('review_count', 0) for p in products]
        max_reviews = max(review_counts) if review_counts else 0
        
        for product in products:
            reviews = product.get('review_count', 0)
            
            if reviews > 0 and max_reviews > 0:
                # Logarithmic scaling with confidence tiers
                if reviews >= 5000:
                    confidence_score = 100  # Maximum confidence
                elif reviews >= 1000:
                    confidence_score = 90 + (math.log(reviews / 1000) / math.log(5)) * 10
                elif reviews >= 100:
                    confidence_score = 70 + (math.log(reviews / 100) / math.log(10)) * 20
                elif reviews >= 50:
                    confidence_score = 50 + (reviews - 50) / 50 * 20
                else:
                    confidence_score = (reviews / 50) * 50
                
                product['review_confidence_score'] = round(min(confidence_score, 100), 2)
            else:
                product['review_confidence_score'] = 0
        
        return products
    
    def _analyze_rating_quality(self, products: List[Dict]) -> List[Dict]:
        """Analyze rating quality"""
        max_rating = 5.0
        
        for product in products:
            rating = product.get('rating_val', 0)
            # Exponential scoring favors higher ratings
            if rating:
                normalized = (rating / max_rating) ** 1.5
                product['rating_quality_score'] = round(normalized * 100, 2)
            else:
                product['rating_quality_score'] = 0
        
        return products
    
    def _analyze_review_volume(self, products: List[Dict]) -> List[Dict]:
        """Analyze review volume (confidence indicator)"""
        import math
        
        review_counts = [p.get('review_count', 0) for p in products]
        max_reviews = max(review_counts) if review_counts else 0
        
        for product in products:
            reviews = product.get('review_count', 0)
            if reviews > 0 and max_reviews > 0:
                # Logarithmic scaling for review count
                log_score = math.log(reviews + 1) / math.log(max_reviews + 1)
                product['review_volume_score'] = round(log_score * 100, 2)
            else:
                product['review_volume_score'] = 0
        
        return products
    
    def _analyze_sentiment(self, products: List[Dict]) -> List[Dict]:
        """Analyze customer sentiment from text"""
        
        positive_keywords = {
            'excellent': 10, 'amazing': 9, 'perfect': 8, 'outstanding': 9,
            'love': 7, 'great': 6, 'good': 5, 'best': 8,
            'comfortable': 6, 'durable': 7, 'quality': 6, 'sturdy': 7,
            'recommend': 8, 'satisfied': 6, 'happy': 5, 'pleased': 6,
            'beautiful': 6, 'worth': 6, 'impressive': 7, 'superb': 8
        }
        
        negative_keywords = {
            'terrible': -10, 'awful': -9, 'worst': -8, 'horrible': -9,
            'hate': -7, 'poor': -6, 'bad': -5, 'disappointing': -7,
            'cheap': -4, 'flimsy': -6, 'uncomfortable': -7, 'broken': -8,
            'difficult': -5, 'problem': -6, 'issue': -5, 'defective': -8,
            'complaint': -6, 'disappointed': -7, 'waste': -7, 'useless': -8
        }
        
        for product in products:
            sentiment_score = 50  # Neutral baseline
            
            # Get all text sources
            customers_say = product.get('customers_say', {})
            features = ' '.join(product.get('features', []))
            description = product.get('description', '')
            
            text = (
                customers_say.get('section', '') + ' ' +
                customers_say.get('positive', '') + ' ' +
                features + ' ' +
                description
            ).lower()
            
            negative_text = customers_say.get('negative', '').lower()
            
            # Calculate positive sentiment
            for word, weight in positive_keywords.items():
                count = text.count(word)
                sentiment_score += count * weight
            
            # Calculate negative sentiment
            for word, weight in negative_keywords.items():
                count = text.count(word) + negative_text.count(word) * 2
                sentiment_score += count * weight
            
            # Boost score based on rating if available
            if product.get('rating_val'):
                rating_boost = (product['rating_val'] - 3) * 10
                sentiment_score += rating_boost
            
            # Normalize to 0-100
            sentiment_score = max(0, min(100, sentiment_score))
            product['sentiment_score'] = round(sentiment_score, 2)
        
        return products
    
    def _calculate_final_scores(self, products: List[Dict]) -> List[Dict]:
        """Calculate weighted final scores with IMDb-style Bayesian rating as primary factor"""
        
        for product in products:
            final_score = (
                product.get('bayesian_score', 0) * self.weights['bayesian_rating'] +
                product.get('price_value_score', 0) * self.weights['price_value'] +
                product.get('review_confidence_score', 0) * self.weights['review_confidence'] +
                product.get('sentiment_score', 0) * self.weights['customer_sentiment']
            )
            
            product['final_score'] = round(final_score, 2)
        
        return products

# ------------------- Visualization Functions ------------------- #
def create_score_breakdown_chart(products: List[Dict]) -> go.Figure:
    """Create comprehensive score breakdown visualization"""
    
    best_product = products[0]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Overall AI Scores (IMDb-style Ranking)", "Best Product Analysis"),
        specs=[[{"type": "bar"}, {"type": "polar"}]]
    )
    
    # Bar chart: All products
    fig.add_trace(
        go.Bar(
            x=[f"#{p['final_rank']}" for p in products],
            y=[p['final_score'] for p in products],
            name="AI Score",
            marker=dict(
                color=[p['final_score'] for p in products],
                colorscale='RdYlGn',
                showscale=False,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{p['final_score']:.1f}" for p in products],
            textposition='outside',
            hovertemplate='<b>Rank %{x}</b><br>Score: %{y:.1f}/100<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Radar chart: Best product with new metrics
    categories = ['Bayesian Rating', 'Price Value', 'Review Trust', 'Sentiment']
    values = [
        best_product.get('bayesian_score', 0),
        best_product.get('price_value_score', 0),
        best_product.get('review_confidence_score', 0),
        best_product.get('sentiment_score', 0)
    ]
    
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Best Product',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=10, color='#ff6b6b')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=450,
        showlegend=True,
        title_text="Product Analysis Dashboard (IMDb-Style Bayesian Ranking)",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="Product Rank", row=1, col=1)
    fig.update_yaxes(title_text="AI Score (0-100)", range=[0, 105], row=1, col=1)
    fig.update_polars(radialaxis=dict(range=[0, 100], tickfont=dict(size=10)))
    
    return fig

def create_price_rating_chart(products: List[Dict]) -> go.Figure:
    """Create price vs rating scatter plot"""
    
    df = pd.DataFrame(products)
    
    fig = px.scatter(
        df,
        x='price_val',
        y='rating_val',
        size='review_count',
        color='final_score',
        hover_data=['final_rank', 'title'],
        title="Price vs Rating Analysis",
        labels={
            'price_val': 'Price (₹)',
            'rating_val': 'Rating (out of 5)',
            'review_count': 'Reviews',
            'final_score': 'AI Score'
        },
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='<b>Rank #%{customdata[0]}</b><br>' +
                     '%{customdata[1]}<br><br>' +
                     'Price: ₹%{x:,.0f}<br>' +
                     'Rating: %{y}/5<br>' +
                     'Reviews: %{marker.size:,}<br>' +
                     'AI Score: %{marker.color:.1f}/100<extra></extra>'
    )
    
    fig.update_layout(height=500)
    
    return fig

# ------------------- Main Streamlit App ------------------- #
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Amazon Product Finder AI</h1>
        <p>Find the best products with AI-powered analysis and ranking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'scraper' not in st.session_state:
        st.session_state.scraper = AmazonScraper()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ProductAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Search Parameters")
        
        # Search query
        query = st.text_input(
            "Product Search",
            placeholder="e.g., wireless headphones",
            help="Enter the product you want to search for"
        )
        
        # Price range
        st.subheader("💰 Price Range (₹)")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min", min_value=100, value=5000, step=500)
        with col2:
            max_price = st.number_input("Max", min_value=1000, value=50000, step=1000)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_pages = st.slider("Pages to Scan", 1, 5, 2, 
                                 help="More pages = more products but slower")
            max_products = st.slider("Products to Analyze", 5, 15, 8,
                                    help="Top products for detailed analysis")
            enable_deep_analysis = st.checkbox("Enable Deep Analysis", value=True,
                                              help="Scrape product pages for more details (slower)")
        
        # Search button
        search_button = st.button("🚀 Find Best Products", type="primary", use_container_width=True)
        
        # Info
        st.divider()
        st.info("💡 **Tip:** Start with 1-2 pages to test, then increase for more results.")
    
    # Main content
    if not query:
        show_welcome_screen()
        return
    
    if not search_button:
        return
    
    # Validate inputs
    if min_price >= max_price:
        st.error("❌ Minimum price must be less than maximum price!")
        return
    
    # Execute search
    execute_search(query, min_price, max_price, max_pages, max_products, enable_deep_analysis)

def show_welcome_screen():
    """Display welcome screen"""
    st.markdown("""
    ## 👋 Welcome to Amazon Product Finder AI!
    
    This tool helps you find the **best products** on Amazon using AI-powered analysis:
    
    ### 🎯 Analysis Factors:
    - **💰 Price Value** - Find products with the best value for money
    - **⭐ Rating Quality** - Higher customer ratings mean better products
    - **📊 Review Volume** - More reviews = higher confidence
    - **😊 Customer Sentiment** - Analyze what customers actually say
    
    ### 🚀 How to Use:
    1. Enter a product name in the sidebar
    2. Set your budget range
    3. Click "Find Best Products"
    4. Get AI-powered recommendations!
    
    ### ✨ Features:
    - Smart AI ranking algorithm
    - Real-time Amazon data
    - Visual analytics
    - Sentiment analysis
    - Export results
    
    **Ready? Enter a search term in the sidebar to get started! →**
    """)
    
    # Sample preview
    with st.expander("📊 See Sample Results"):
        sample_df = pd.DataFrame({
            'Rank': [1, 2, 3],
            'Product': ['Premium Headphones', 'Pro Headphones', 'Budget Headphones'],
            'Price': ['₹8,999', '₹12,499', '₹4,999'],
            'Rating': ['4.5 ⭐', '4.3 ⭐', '4.1 ⭐'],
            'AI Score': ['92.5', '87.3', '78.8']
        })
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

def execute_search(query: str, min_price: int, max_price: int, max_pages: int, 
                   max_products: int, enable_deep_analysis: bool):
    """Execute the search and analysis"""
    
    st.header(f"🔍 Searching: **{query}**")
    st.info(f"💰 Budget: ₹{min_price:,} - ₹{max_price:,}")
    
    scraper = st.session_state.scraper
    analyzer = st.session_state.analyzer
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_products = []
    
    # Phase 1: Scrape search results
    status_text.text("🔍 Phase 1: Searching Amazon pages...")
    
    for page in range(1, max_pages + 1):
        progress = (page / max_pages) * 0.4  # 40% for searching
        progress_bar.progress(progress)
        status_text.text(f"📖 Scanning page {page}/{max_pages}...")
        
        url = scraper.build_search_url(query, min_price, max_price, page)
        html = scraper.fetch_page(url)
        
        if not html:
            st.warning(f"⚠️ Could not fetch page {page}. Continuing with available data...")
            break
        
        products = scraper.parse_products(html)
        
        # Filter by price range
        filtered = [
            p for p in products
            if p.get('price_val') and min_price <= p['price_val'] <= max_price
        ]
        
        all_products.extend(filtered)
        st.success(f"✅ Page {page}: Found {len(filtered)} products")
        
        # Progress update
        if page < max_pages:
            time.sleep(random.uniform(2, 4))  # Respectful delay
    
    if not all_products:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ No products found matching your criteria. Try:")
        st.markdown("""
        - Broadening your price range
        - Using different search terms
        - Checking if the product category exists on Amazon.in
        """)
        return
    
    # Remove duplicates by ASIN
    seen_asins = set()
    unique_products = []
    for p in all_products:
        if p['asin'] not in seen_asins:
            seen_asins.add(p['asin'])
            unique_products.append(p)
    
    all_products = unique_products
    
    st.success(f"🎉 Found {len(all_products)} unique products!")
    
    # Sort by rating * reviews to get best candidates
    all_products.sort(
        key=lambda x: (x.get('rating_val', 0) * x.get('review_count', 0)), 
        reverse=True
    )
    
    # Select top products for detailed analysis
    top_products = all_products[:max_products]
    
    # Phase 2: Deep analysis (if enabled)
    if enable_deep_analysis:
        status_text.text("📝 Phase 2: Deep product analysis...")
        
        for i, product in enumerate(top_products):
            progress = 0.4 + ((i + 1) / len(top_products)) * 0.4  # 40% for analysis
            progress_bar.progress(progress)
            status_text.text(f"🔬 Analyzing product {i+1}/{len(top_products)}: {product['title'][:50]}...")
            
            details = scraper.scrape_product_details(product['link'])
            product.update(details)
            
            time.sleep(random.uniform(2, 4))  # Respectful delay
    
    # Phase 3: AI Analysis
    status_text.text("🤖 Phase 3: AI ranking and scoring...")
    progress_bar.progress(0.9)
    
    analyzed_products = analyzer.analyze_products(top_products)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    
    # Small delay then clear
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    display_results(analyzed_products, query, min_price, max_price)

def display_results(products: List[Dict], query: str, min_price: int, max_price: int):
    """Display comprehensive results"""
    
    if not products:
        st.error("No products to display")
        return
    
    best_product = products[0]
    
    # AI Recommendation Box
    st.markdown(f"""
    <div class="recommendation-box">
        <h2>🏆 AI RECOMMENDATION (IMDb-Style Ranking)</h2>
        <h3>{best_product['title'][:100]}</h3>
        <br>
        <p><strong>🎯 Why this is the best choice:</strong></p>
        <ul style="font-size: 16px;">
            <li><strong>Overall AI Score:</strong> {best_product['final_score']:.1f}/100</li>
            <li><strong>Bayesian Rating:</strong> {best_product.get('bayesian_rating', 0):.2f}/5.0 (Weighted with {best_product['review_count']:,} reviews)</li>
            <li><strong>Actual Rating:</strong> {best_product['rating_val']:.1f}/5.0</li>
            <li><strong>Price:</strong> ₹{best_product['price_val']:,.2f}</li>
            <li><strong>Review Confidence:</strong> {best_product.get('review_confidence_score', 0):.1f}/100</li>
            <li><strong>Customer Sentiment:</strong> {best_product.get('sentiment_score', 0):.1f}/100</li>
        </ul>
        <p><em>This product uses IMDb-style Bayesian ranking where products with more reviews are weighted more heavily. A product with 1000 reviews at 4.3★ can rank higher than one with 50 reviews at 4.7★ due to higher statistical confidence.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_price = sum(p['price_val'] for p in products) / len(products)
    price_diff = best_product['price_val'] - avg_price
    
    with col1:
        st.metric(
            "🏆 Best AI Score",
            f"{best_product['final_score']:.1f}/100",
            help="AI-calculated overall score"
        )
    
    with col2:
        st.metric(
            "💰 Best Price",
            f"₹{best_product['price_val']:,.0f}",
            delta=f"{price_diff:,.0f} vs avg",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "⭐ Rating",
            f"{best_product['rating_val']:.1f}/5.0",
            help="Customer rating"
        )
    
    with col4:
        st.metric(
            "👥 Reviews",
            f"{best_product['review_count']:,}",
            help="Number of customer reviews"
        )
    
    # Charts
    st.subheader("📈 Visual Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Score Breakdown", "💰 Price vs Rating", "📋 Comparison Table"])
    
    with tab1:
        fig_scores = create_score_breakdown_chart(products)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Additional insights
        st.markdown("### 🔍 Score Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Best Bayesian Rating:**  
            Product #{products[0]['final_rank']} has a Bayesian rating of {products[0].get('bayesian_rating', 0):.2f}/5.0 with {products[0]['review_count']:,} reviews (Rating weight: {products[0].get('rating_weight', 0)*100:.1f}%)
            """)
        
        with col2:
            highest_reviews = max(products, key=lambda x: x.get('review_count', 0))
            st.success(f"""
            **Most Reviewed:**  
            Product #{highest_reviews['final_rank']} has {highest_reviews['review_count']:,} reviews with {highest_reviews['rating_val']:.1f}/5.0 rating
            """)
    
    with tab2:
        fig_scatter = create_price_rating_chart(products)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Price distribution
        st.markdown("### 💵 Price Distribution")
        prices = [p['price_val'] for p in products]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lowest Price", f"₹{min(prices):,.0f}")
        with col2:
            st.metric("Average Price", f"₹{sum(prices)/len(prices):,.0f}")
        with col3:
            st.metric("Highest Price", f"₹{max(prices):,.0f}")
    
    with tab3:
        # Detailed comparison table
        comparison_df = pd.DataFrame([
            {
                'Rank': p['final_rank'],
                'Product': p['title'][:60] + '...',
                'Price': f"₹{p['price_val']:,.0f}",
                'Rating': f"{p['rating_val']:.1f} ⭐",
                'Reviews': f"{p['review_count']:,}",
                'Bayesian': f"{p.get('bayesian_rating', 0):.2f}",
                'AI Score': f"{p['final_score']:.1f}",
                'Confidence': f"{p.get('review_confidence_score', 0):.0f}%"
            }
            for p in products
        ])
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "AI Score": st.column_config.ProgressColumn(
                    "AI Score",
                    format="%.1f",
                    min_value=0,
                    max_value=100
                )
            }
        )
    
    # Detailed Product Cards
    st.subheader("🎯 Detailed Product Analysis")
    
    for i, product in enumerate(products):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        
        with st.expander(
            f"{rank_emoji} {product['title'][:70]}... (Score: {product['final_score']:.1f}/100)",
            expanded=(i == 0)
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {product['title']}")
                
                # Basic info
                st.markdown(f"""
                **💰 Price:** ₹{product['price_val']:,.2f}  
                **⭐ Rating:** {product['rating_val']:.1f}/5.0 ({product['review_count']:,} reviews)  
                **📊 Bayesian Rating:** {product.get('bayesian_rating', 0):.2f}/5.0  
                **🔗 ASIN:** {product['asin']}
                
                *Bayesian rating weights this product's actual rating ({product.get('rating_weight', 0)*100:.0f}%) vs global average ({product.get('mean_weight', 0)*100:.0f}%)*
                """)
                
                # AI Analysis
                st.markdown("---")
                st.markdown("### 🤖 AI Analysis Breakdown")
                
                scores_col1, scores_col2 = st.columns(2)
                
                with scores_col1:
                    st.metric("Overall Score", f"{product['final_score']:.1f}/100")
                    st.metric("Bayesian Rating", f"{product.get('bayesian_rating', 0):.2f}/5.0")
                    st.metric("Price Value", f"{product.get('price_value_score', 0):.1f}/100")
                
                with scores_col2:
                    st.metric("Review Confidence", f"{product.get('review_confidence_score', 0):.1f}/100")
                    st.metric("Customer Sentiment", f"{product.get('sentiment_score', 0):.1f}/100")
                    
                    # Show why this ranking makes sense
                    if product['review_count'] >= 1000:
                        st.success("🎯 High Trust (1000+ reviews)")
                    elif product['review_count'] >= 100:
                        st.info("✅ Good Trust (100+ reviews)")
                    else:
                        st.warning("⚠️ Limited data (<100 reviews)")
                
                # Features
                if product.get('features'):
                    st.markdown("### ✨ Key Features")
                    for feature in product['features'][:5]:
                        st.markdown(f"- {feature}")
                
                # Description
                if product.get('description'):
                    st.markdown("### 📝 Description")
                    st.text(product['description'][:400] + "...")
                
                # Customer insights
                customers_say = product.get('customers_say', {})
                
                if customers_say.get('positive'):
                    st.success(f"✅ **Positive Reviews:** {customers_say['positive'][:250]}...")
                
                if customers_say.get('negative'):
                    st.warning(f"⚠️ **Critical Reviews:** {customers_say['negative'][:250]}...")
                
                if customers_say.get('section'):
                    with st.expander("📊 Customer Insights"):
                        st.write(customers_say['section'][:500])
            
            with col2:
                # Image
                if product.get('image_url'):
                    st.image(product['image_url'], use_container_width=True)
                
                # Quick stats
                st.markdown("### 📊 Quick Stats")
                st.metric("Final Rank", f"#{product['final_rank']}")
                st.metric("AI Score", f"{product['final_score']:.1f}/100")
                st.metric("Bayesian Rating", f"{product.get('bayesian_rating', 0):.2f}/5.0")
                
                # Trust indicator
                confidence = product.get('review_confidence_score', 0)
                if confidence >= 90:
                    st.success(f"🛡️ Very High Trust\n{confidence:.0f}% confidence")
                elif confidence >= 70:
                    st.info(f"✅ High Trust\n{confidence:.0f}% confidence")
                elif confidence >= 50:
                    st.warning(f"⚠️ Medium Trust\n{confidence:.0f}% confidence")
                else:
                    st.error(f"❌ Low Trust\n{confidence:.0f}% confidence")
                
                # Action buttons
                st.link_button(
                    "🛒 View on Amazon",
                    product['link'],
                    type="primary",
                    use_container_width=True
                )
                
                # Price comparison
                if product['price_val'] < avg_price:
                    savings = avg_price - product['price_val']
                    st.success(f"💰 Save ₹{savings:,.0f} vs average")
                elif product['price_val'] > avg_price:
                    extra = product['price_val'] - avg_price
                    st.info(f"💵 ₹{extra:,.0f} above average")
    
    # Export Section
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_df = pd.DataFrame([
            {
                'Rank': p['final_rank'],
                'Title': p['title'],
                'ASIN': p['asin'],
                'Price': p['price_val'],
                'Rating': p['rating_val'],
                'Review_Count': p['review_count'],
                'Bayesian_Rating': p.get('bayesian_rating', 0),
                'AI_Score': p['final_score'],
                'Price_Value_Score': p.get('price_value_score', 0),
                'Review_Confidence_Score': p.get('review_confidence_score', 0),
                'Sentiment_Score': p.get('sentiment_score', 0),
                'Rating_Weight': p.get('rating_weight', 0),
                'Amazon_Link': p['link']
            }
            for p in products
        ])
        
        st.download_button(
            "📊 Download CSV",
            csv_df.to_csv(index=False),
            file_name=f"amazon_analysis_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON Export
        json_data = {
            'search_query': query,
            'price_range': {'min': min_price, 'max': max_price},
            'analysis_date': datetime.now().isoformat(),
            'total_products_analyzed': len(products),
            'best_product': {
                'rank': 1,
                'title': best_product['title'],
                'asin': best_product['asin'],
                'price': best_product['price_val'],
                'rating': best_product['rating_val'],
                'ai_score': best_product['final_score']
            },
            'all_products': products
        }
        
        st.download_button(
            "📄 Download JSON",
            json.dumps(json_data, indent=2, ensure_ascii=False),
            file_name=f"amazon_analysis_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Summary Report
        report = f"""# Amazon Product Analysis Report

## Search Details
- **Query:** {query}
- **Price Range:** ₹{min_price:,} - ₹{max_price:,}
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Products Analyzed:** {len(products)}

## Top Recommendation
**{best_product['title']}**

- Price: ₹{best_product['price_val']:,.2f}
- Rating: {best_product['rating_val']:.1f}/5.0
- Bayesian Rating: {best_product.get('bayesian_rating', 0):.2f}/5.0
- Reviews: {best_product['review_count']:,}
- AI Score: {best_product['final_score']:.1f}/100

**Why this product ranked #1:**
Uses IMDb-style Bayesian ranking where products with more reviews get weighted more heavily. This ensures statistical confidence in the rating.

## Ranking Methodology
- Bayesian Rating: 45% (IMDb-style weighted rating)
- Price Value: 25% (Best value for money)
- Review Confidence: 15% (Statistical confidence)
- Customer Sentiment: 15% (Text analysis)

## All Products Summary
"""
        for p in products:
            report += f"\n{p['final_rank']}. {p['title'][:60]}... (AI Score: {p['final_score']:.1f}/100)\n"
            report += f"   Price: ₹{p['price_val']:,.0f} | Rating: {p['rating_val']:.1f}/5.0 | Bayesian: {p.get('bayesian_rating', 0):.2f}/5.0 | Reviews: {p['review_count']:,}\n"
        
        st.download_button(
            "📝 Download Report",
            report,
            file_name=f"report_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Search again
    st.divider()
    if st.button("🔄 New Search", type="secondary", use_container_width=False):
        st.rerun()

if __name__ == "__main__":
    main()