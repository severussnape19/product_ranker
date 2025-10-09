import streamlit as st
import os
import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Amazon Shopping Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_params' not in st.session_state:
    st.session_state.search_params = {}

# Custom CSS for chatbot styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    
    .product-card-chat {
        background: white;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .positive-insight {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        color: #155724;
        font-weight: 500;
    }
    
    .negative-insight {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        color: #721c24;
        font-weight: 500;
    }
    
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-style: italic;
    }
    
    .recommendation-highlight {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Enhanced Customer Insights Extraction ------------------- #
class CustomerInsightsExtractor:
    def __init__(self):
        self.positive_keywords = [
            'positive', 'appreciate', 'good', 'great', 'excellent', 'love', 
            'smooth', 'solid', 'value', 'quality', 'comfortable', 'durable',
            'recommend', 'satisfied', 'happy', 'beautiful', 'sturdy', 'worth',
            'amazing', 'perfect', 'fantastic', 'outstanding', 'impressive'
        ]
        
        self.negative_keywords = [
            'negative', 'concern', 'problem', 'poor', 'bad', 'issue', 
            'slow', 'heat', 'battery', 'terrible', 'awful', 'worst', 'hate',
            'cheap', 'flimsy', 'uncomfortable', 'difficult', 'complaint',
            'disappointed', 'defective', 'broken', 'useless'
        ]
    
    def extract_comprehensive_insights(self, url):
        """Extract comprehensive customer insights from Amazon product page"""
        try:
            resp = requests.get(url, headers=random_headers(), timeout=30)
            if resp.status_code != 200:
                return {}

            soup = BeautifulSoup(resp.text, 'html.parser')
            page_text = resp.text.lower()
            
            # Check if we're blocked
            blocked_phrases = ['captcha', 'robot check', 'unusual traffic']
            if any(phrase in page_text for phrase in blocked_phrases):
                return {}
            
            insights = {}
            
            # Search for "Customers say" variants
            variants = [
                'customers say', 'customer say', 'customers find', 'customer find',
                'customers mention', 'customers report', 'review highlights', 
                'customer insights', 'what customers say', 'customers love',
                'customers appreciate', 'buyers say'
            ]
            
            found_variants = []
            customers_say_texts = []
            
            for variant in variants:
                if variant in page_text:
                    found_variants.append(variant)
                    elements = soup.find_all(string=re.compile(variant, re.IGNORECASE))
                    for elem in elements:
                        parent = elem.parent
                        if parent:
                            container = parent.find_parent(['div', 'section', 'span'])
                            if container:
                                text = container.get_text(separator=" ", strip=True)
                                if len(text) > 50 and text not in customers_say_texts:
                                    customers_say_texts.append(text[:1000])
            
            insights['raw_customer_texts'] = customers_say_texts
            insights['found_variants'] = found_variants
            insights.update(self.analyze_sentiment(customers_say_texts))
            
            return insights
            
        except Exception as e:
            return {}
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment from customer texts"""
        if not texts:
            return {}
        
        positive_mentions = []
        negative_mentions = []
        
        sentences = []
        for text in texts:
            sentences.extend(re.split(r'(?<=[.!?])\s+', text))
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
                
            sentence_lower = sentence.lower()
            positive_score = sum(1 for keyword in self.positive_keywords if keyword in sentence_lower)
            negative_score = sum(1 for keyword in self.negative_keywords if keyword in sentence_lower)
            
            if positive_score > negative_score and positive_score > 0:
                positive_mentions.append(sentence.strip())
            elif negative_score > positive_score and negative_score > 0:
                negative_mentions.append(sentence.strip())
        
        return {
            'positive_mentions': positive_mentions[:5],
            'negative_mentions': negative_mentions[:5],
            'sentiment_summary': {
                'total_positive': len(positive_mentions),
                'total_negative': len(negative_mentions),
                'overall_sentiment': 'positive' if len(positive_mentions) > len(negative_mentions) else 'negative' if len(negative_mentions) > 0 else 'neutral'
            }
        }

# ------------------- Product Analysis ------------------- #
class ProductAnalyzer:
    def __init__(self):
        self.weights = {
            'price_value': 0.20,
            'rating_quality': 0.30,
            'review_volume': 0.20,
            'customer_sentiment': 0.30
        }
    
    def calculate_final_rankings(self, products):
        """Calculate comprehensive final rankings"""
        if not products:
            return products
        
        # Analyze price value
        prices = [p['price_val'] for p in products if p['price_val']]
        if prices:
            max_price = max(prices)
            min_price = min(prices)
            
            for product in products:
                if product['price_val']:
                    if max_price == min_price:
                        price_score = 50
                    else:
                        price_score = ((max_price - product['price_val']) / (max_price - min_price)) * 100
                    product['price_value_score'] = round(price_score, 2)
                else:
                    product['price_value_score'] = 0
        
        # Calculate sentiment scores
        for product in products:
            product['sentiment_score'] = self.analyze_customer_sentiment(product)
        
        # Normalize scores
        max_rating = max(p['rating_val'] or 0 for p in products)
        max_reviews = max(p['review_count'] or 0 for p in products)
        
        for product in products:
            rating_score = ((product['rating_val'] or 0) / max_rating) * 100 if max_rating > 0 else 0
            
            import math
            if max_reviews > 0:
                review_score = (math.log((product['review_count'] or 0) + 1) / math.log(max_reviews + 1)) * 100
            else:
                review_score = 0
            
            final_score = (
                (product['price_value_score'] * self.weights['price_value']) +
                (rating_score * self.weights['rating_quality']) +
                (review_score * self.weights['review_volume']) +
                (product['sentiment_score'] * self.weights['customer_sentiment'])
            )
            
            product.update({
                'rating_quality_score': round(rating_score, 2),
                'review_volume_score': round(review_score, 2),
                'final_score': round(final_score, 2)
            })
        
        products.sort(key=lambda x: x['final_score'], reverse=True)
        
        for i, product in enumerate(products, 1):
            product['final_rank'] = i
        
        return products
    
    def analyze_customer_sentiment(self, product):
        """Enhanced customer sentiment analysis"""
        sentiment_score = 50
        
        customers_insights = product.get('customers_insights', {})
        positive_mentions = customers_insights.get('positive_mentions', [])
        negative_mentions = customers_insights.get('negative_mentions', [])
        
        total_positive = len(positive_mentions)
        total_negative = len(negative_mentions)
        
        if total_positive + total_negative > 0:
            positive_weight = total_positive * 8
            negative_weight = total_negative * 6
            sentiment_score = 50 + positive_weight - negative_weight
        
        if customers_insights.get('raw_customer_texts'):
            sentiment_score += 10
        
        if not customers_insights:
            sentiment_score -= 20
        
        sentiment_score = max(0, min(100, sentiment_score))
        return round(sentiment_score, 2)

# ------------------- Chatbot Engine ------------------- #
class AmazonShoppingBot:
    def __init__(self):
        self.analyzer = ProductAnalyzer()
        self.insights_extractor = CustomerInsightsExtractor()
        self.context = {}
    
    def parse_user_message(self, message):
        """Parse user message to extract intent and parameters"""
        message_lower = message.lower()
        
        # Extract product query
        product_indicators = ['find', 'search', 'looking for', 'want', 'need', 'show me', 'get me']
        if any(indicator in message_lower for indicator in product_indicators):
            # Extract the product name
            for indicator in product_indicators:
                if indicator in message_lower:
                    parts = message_lower.split(indicator)
                    if len(parts) > 1:
                        product_query = parts[1].strip()
                        # Clean up common words
                        product_query = re.sub(r'\b(under|below|within|around|for|a|an|the)\b', '', product_query).strip()
                        break
        else:
            product_query = message_lower
        
        # Extract price range
        price_matches = re.findall(r'₹?(\d+(?:,\d+)*(?:\.\d+)?)', message)
        prices = [float(p.replace(',', '')) for p in price_matches]
        
        min_price = None
        max_price = None
        
        if len(prices) == 1:
            if any(word in message_lower for word in ['under', 'below', 'less than', 'max']):
                max_price = prices[0]
                min_price = 1000
            else:
                min_price = max(1000, prices[0] - 10000)
                max_price = prices[0] + 10000
        elif len(prices) >= 2:
            min_price = min(prices)
            max_price = max(prices)
        else:
            # Default price range
            min_price = 5000
            max_price = 100000
        
        return {
            'intent': 'search',
            'product_query': product_query,
            'min_price': min_price,
            'max_price': max_price
        }
    
    def generate_response(self, user_message):
        """Generate chatbot response"""
        parsed = self.parse_user_message(user_message)
        
        if parsed['intent'] == 'search':
            return self.search_products(parsed)
        else:
            return "I'd be happy to help you find products on Amazon! Just tell me what you're looking for and your budget."
    
    def search_products(self, params):
        """Search for products and return formatted response"""
        try:
            # Store search parameters
            st.session_state.search_params = params
            
            # Search products
            all_products = []
            
            for page in range(1, 3):  # Search 2 pages
                html = fetch_amazon_html(params['product_query'], params['min_price'], params['max_price'], page=page)
                if html:
                    products = parse_amazon_html(html)
                    filtered_products = [
                        p for p in products 
                        if p['price_val'] and params['min_price'] <= p['price_val'] <= params['max_price']
                    ]
                    all_products.extend(filtered_products)
                time.sleep(2)
            
            if not all_products:
                return "I couldn't find any products matching your criteria. Try adjusting your search terms or price range."
            
            # Get top products
            all_products.sort(key=lambda x: (x['rating_val'] * x['review_count']), reverse=True)
            top_products = all_products[:4]  # Get top 4 products
            
            # Extract customer insights for top products
            for product in top_products:
                customers_insights = self.insights_extractor.extract_comprehensive_insights(product['link'])
                product['customers_insights'] = customers_insights
                time.sleep(2)  # Be respectful
            
            # Analyze and rank
            analyzed_products = self.analyzer.calculate_final_rankings(top_products)
            
            # Store results
            st.session_state.search_results = analyzed_products
            
            # Generate response
            return self.format_search_results(analyzed_products, params)
            
        except Exception as e:
            return f"Sorry, I encountered an error while searching: {str(e)}"
    
    def format_search_results(self, products, params):
        """Format search results for chat display"""
        if not products:
            return "No products found."
        
        best_product = products[0]
        
        response = f"🎉 **Great! I found {len(products)} amazing products for '{params['product_query']}' in your budget of ₹{params['min_price']:,} - ₹{params['max_price']:,}!**\n\n"
        
        response += "## 🏆 My Top Recommendation:\n\n"
        response += f"**{best_product['title'][:80]}...**\n\n"
        response += f"- 💰 **Price:** ₹{best_product['price_val']:,.2f}\n"
        response += f"- ⭐ **Rating:** {best_product['rating_val']}/5.0 ({best_product['review_count']:,} reviews)\n"
        response += f"- 🎯 **AI Score:** {best_product['final_score']}/100\n"
        response += f"- 😊 **Customer Sentiment:** {best_product['sentiment_score']}/100\n\n"
        
        # Add customer insights if available
        insights = best_product.get('customers_insights', {})
        if insights.get('positive_mentions'):
            response += "**✅ What customers love:**\n"
            for mention in insights['positive_mentions'][:2]:
                response += f"- {mention[:100]}...\n"
        
        if insights.get('negative_mentions'):
            response += "\n**⚠️ Common concerns:**\n"
            for mention in insights['negative_mentions'][:1]:
                response += f"- {mention[:100]}...\n"
        
        response += f"\n🔗 [View on Amazon]({best_product['link']})\n\n"
        response += "Would you like to see more options or get details about a specific product?"
        
        return response

# ------------------- Core Scraping Functions ------------------- #
def random_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def build_search_url(query, low, high, page=1, domain="amazon.in"):
    q = quote_plus(query)
    url = f"https://{domain}/s?k={q}&low-price={low}&high-price={high}"
    if page > 1:
        url += f"&page={page}"
    return url

def fetch_amazon_html(query, low, high, page=1, domain="amazon.in", retries=3):
    url = build_search_url(query, low, high, page, domain)
    headers = random_headers()
    
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            html = resp.text.lower()
            for phrase in ["captcha","robot check","unusual traffic"]:
                if phrase in html:
                    return None
            return resp.text
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
    return None

def parse_amazon_html(html):
    soup = BeautifulSoup(html, "html.parser")
    products = soup.find_all("div", {"data-asin": True, "data-component-type": "s-search-result"})
    results = []
    
    for product in products:
        if product.select_one('.puis-sponsored-label-text'):
            continue
            
        title_tag = product.select_one("h2 > span")
        title = title_tag.get_text(strip=True) if title_tag else "No title"
        
        link_tag = product.find("a", class_="a-link-normal", href=True)
        link = "https://www.amazon.in" + link_tag["href"] if link_tag else "No link"
        
        price_tag = product.select_one("span.a-price > span.a-offscreen")
        price_val = None
        if price_tag:
            try: 
                price_val = float(price_tag.get_text(strip=True).replace("₹","").replace(",",""))
            except: pass
        
        rating_tag = product.find("span", class_="a-icon-alt")
        rating_val = None
        if rating_tag:
            match = re.search(r"[\d.]+", rating_tag.get_text(strip=True))
            if match: rating_val = float(match.group())
        
        review_tag = product.select_one("span.a-size-base.s-underline-text")
        review_count = None
        if review_tag:
            try: review_count = int(review_tag.get_text(strip=True).replace(",", ""))
            except: pass
        
        asin = product.get("data-asin", "")
        
        if rating_val and review_count and price_val:
            results.append({
                "title": title,
                "link": link,
                "asin": asin,
                "price_val": price_val,
                "rating_val": rating_val,
                "review_count": review_count
            })
    
    return results

# ------------------- Main Chatbot App ------------------- #
def main():
    # Initialize bot
    if 'bot' not in st.session_state:
        st.session_state.bot = AmazonShoppingBot()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Amazon Shopping Assistant</h1>
        <p>Your AI-powered shopping companion for finding the best products!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hi! I'm your Amazon Shopping Assistant! I can help you find the best products based on ratings, prices, and customer reviews.\n\n**Just tell me:**\n- What product you're looking for\n- Your budget (optional)\n\n**Examples:**\n- \"Find me a gaming laptop under ₹80,000\"\n- \"I need wireless headphones between ₹2,000 and ₹5,000\"\n- \"Show me the best office chairs\""
        })
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong><br><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Show search results if available
        if st.session_state.search_results:
            st.markdown("### 📦 Product Details")
            display_products_in_chat(st.session_state.search_results)
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Searches")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_searches = [
        ("💻 Laptops", "gaming laptop under ₹100000"),
        ("🎧 Headphones", "wireless headphones under ₹5000"),
        ("📱 Smartphones", "smartphone between ₹15000 and ₹30000"),
        ("🪑 Chairs", "office chair under ₹20000")
    ]
    
    for i, (display, query) in enumerate(quick_searches):
        with [col1, col2, col3, col4][i]:
            if st.button(display, key=f"quick_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🔍 Searching for products..."):
                    response = st.session_state.bot.generate_response(query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Chat input
    user_input = st.chat_input("Type your message here... (e.g., 'Find me a gaming laptop under ₹80,000')")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator
        with st.spinner("🤖 Let me search for the best products for you..."):
            response = st.session_state.bot.generate_response(user_input)
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Sidebar with additional options
    with st.sidebar:
        st.header("🛠️ Chat Options")
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.search_results = None
            st.rerun()
        
        if st.session_state.search_results:
            st.subheader("📊 Current Results")
            st.write(f"Found {len(st.session_state.search_results)} products")
            
            if st.button("📈 View Analytics", type="primary"):
                show_analytics_popup()
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips:
        - Be specific about what you want
        - Mention your budget for better results
        - Ask for comparisons between products
        - Request specific features you need
        
        ### 🔍 I can help you find:
        - Electronics & Gadgets
        - Home & Kitchen items
        - Fashion & Accessories  
        - Books & Stationery
        - Sports & Fitness gear
        - And much more!
        """)

def display_products_in_chat(products):
    """Display products in chat-friendly format"""
    for i, product in enumerate(products):
        with st.expander(f"#{i+1} - {product['title'][:60]}... (AI Score: {product['final_score']}/100)", 
                        expanded=(i == 0)):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **🏷️ {product['title']}**
                
                **💰 Price:** ₹{product['price_val']:,.2f}  
                **⭐ Rating:** {product['rating_val']}/5.0 ({product['review_count']:,} reviews)  
                **🎯 AI Score:** {product['final_score']}/100
                """)
                
                # Customer insights
                insights = product.get('customers_insights', {})
                if insights.get('positive_mentions'):
                    st.markdown("**✅ What customers love:**")
                    for mention in insights['positive_mentions'][:2]:
                        st.markdown(f"<div class='positive-insight'>{mention}</div>", unsafe_allow_html=True)
                
                if insights.get('negative_mentions'):
                    st.markdown("**⚠️ Common concerns:**")
                    for mention in insights['negative_mentions'][:1]:
                        st.markdown(f"<div class='negative-insight'>{mention}</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Rank", f"#{product['final_rank']}")
                st.metric("Sentiment", f"{product.get('sentiment_score', 0)}/100")
                st.link_button("🛒 Buy Now", product['link'], use_container_width=True)

def show_analytics_popup():
    """Show analytics in a popup"""
    if st.session_state.search_results:
        st.markdown("### 📊 Search Analytics")
        
        df = pd.DataFrame(st.session_state.search_results)
        
        # Price vs Rating chart
        fig = px.scatter(
            df, 
            x='price_val', 
            y='rating_val',
            size='review_count',
            color='final_score',
            title="Price vs Rating Analysis",
            labels={'price_val': 'Price (₹)', 'rating_val': 'Rating'}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
