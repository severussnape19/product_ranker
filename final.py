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

# ------------------- Enhanced Ranking & Analysis System ------------------- #
class ProductAnalyzer:
    def __init__(self):
        self.weights = {
            'price_value': 0.25,      # Price competitiveness
            'rating_quality': 0.30,   # Rating score
            'review_volume': 0.20,    # Review count influence
            'customer_sentiment': 0.25 # Positive vs negative sentiment
        }
    
    def analyze_price_value(self, products):
        """Analyze price competitiveness (lower price = better value)"""
        if not products:
            return products
        
        prices = [p['price_val'] for p in products if p['price_val']]
        if not prices:
            return products
            
        max_price = max(prices)
        min_price = min(prices)
        
        for product in products:
            if product['price_val']:
                # Invert price score (lower price = higher score)
                price_score = ((max_price - product['price_val']) / (max_price - min_price)) * 100
                product['price_value_score'] = round(price_score, 2)
            else:
                product['price_value_score'] = 0
        
        return products
    
    def analyze_customer_sentiment(self, product):
        """Analyze customer sentiment from scraped insights"""
        sentiment_score = 50  # Neutral baseline
        
        customers_say = product.get('customers_say', {})
        
        # Positive indicators
        positive_text = (
            customers_say.get('positive_mentions', '') + ' ' +
            customers_say.get('customers_say_section', '') + ' ' +
            customers_say.get('key_features', '')
        ).lower()
        
        # Negative indicators
        negative_text = customers_say.get('negative_mentions', '').lower()
        
        # Positive keywords with weights
        positive_keywords = {
            'excellent': 10, 'amazing': 9, 'perfect': 8, 'love': 7,
            'great': 6, 'good': 5, 'comfortable': 6, 'durable': 7,
            'quality': 5, 'recommend': 8, 'satisfied': 6, 'happy': 5,
            'beautiful': 6, 'sturdy': 7, 'value': 5, 'worth': 6
        }
        
        # Negative keywords with weights
        negative_keywords = {
            'terrible': -10, 'awful': -9, 'worst': -8, 'hate': -7,
            'poor': -6, 'bad': -5, 'cheap': -4, 'flimsy': -6,
            'uncomfortable': -7, 'difficult': -5, 'problem': -6,
            'issue': -5, 'complaint': -6, 'disappointed': -7
        }
        
        # Calculate positive sentiment
        for word, weight in positive_keywords.items():
            count = positive_text.count(word)
            sentiment_score += count * weight
        
        # Calculate negative sentiment
        for word, weight in negative_keywords.items():
            count = negative_text.count(word)
            sentiment_score += count * weight
        
        # Normalize to 0-100 scale
        sentiment_score = max(0, min(100, sentiment_score))
        
        return round(sentiment_score, 2)
    
    def calculate_final_rankings(self, products):
        """Calculate comprehensive final rankings"""
        if not products:
            return products
        
        # Analyze price value
        products = self.analyze_price_value(products)
        
        # Calculate sentiment scores
        for product in products:
            product['sentiment_score'] = self.analyze_customer_sentiment(product)
        
        # Normalize scores
        max_rating = max(p['rating_val'] or 0 for p in products)
        max_reviews = max(p['review_count'] or 0 for p in products)
        
        for product in products:
            # Rating quality score
            rating_score = ((product['rating_val'] or 0) / max_rating) * 100 if max_rating > 0 else 0
            
            # Review volume score (logarithmic)
            import math
            review_score = (math.log((product['review_count'] or 0) + 1) / math.log(max_reviews + 1)) * 100 if max_reviews > 0 else 0
            
            # Calculate weighted final score
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
        
        # Sort by final score
        products.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking
        for i, product in enumerate(products, 1):
            product['final_rank'] = i
        
        return products

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return ProductAnalyzer()

# ------------------- Core Scraping Functions ------------------- #
def random_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
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
                    st.warning(f"⚠️ Bot detection detected: '{phrase}'")
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

def scrape_customers_say_section(url):
    """
    Fetch the 'What Customers Say' or 'Customer Insights' section directly from an Amazon product page.
    Returns a dictionary with the extracted text.
    """
    try:
        resp = requests.get(url, headers=random_headers(), timeout=30)
        if resp.status_code != 200:
            return {}

        soup = BeautifulSoup(resp.text, 'html.parser')
        customers_say_data = {}

        # Look for the section heading (h2, span, etc.) containing 'customer insights' or 'customers say'
        section = None
        for header in soup.find_all(["h2", "span"], string=re.compile(r'customer insights|what customers say', re.IGNORECASE)):
            parent_div = header.find_parent("div")
            if parent_div:
                section = parent_div
                break

        if section:
            customers_say_data['customers_say_section'] = section.get_text(separator=" ", strip=True)[:800]

        return customers_say_data

    except Exception as e:
        return {}

# ------------------- Main Streamlit App ------------------- #
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Amazon Product Finder AI</h1>
        <p>Find the best products with AI-powered analysis and ranking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for search parameters
    with st.sidebar:
        st.header("🔍 Search Parameters")
        
        # Search query
        query = st.text_input(
            "Product Search Query",
            placeholder="e.g., gaming laptop, wireless headphones, office chair",
            help="Enter the product you want to search for"
        )
        
        # Price range
        st.subheader("💰 Price Range (₹)")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum", min_value=1000, value=10000, step=1000)
        with col2:
            max_price = st.number_input("Maximum", min_value=5000, value=100000, step=5000)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_pages = st.slider("Pages to Scan", 1, 10, 3, help="More pages = more products but slower")
            max_products = st.slider("Products to Analyze", 3, 10, 5, help="Number of top products to analyze in detail")
        
        # Search button
        search_button = st.button("🚀 Find Best Products", type="primary", use_container_width=True)
        
        # Quick search options
        st.subheader("⚡ Quick Searches")
        quick_searches = [
            ("💻 Gaming Laptop", "gaming laptop"),
            ("🎧 Wireless Headphones", "wireless headphones"),
            ("📱 Smartphone", "smartphone"),
            ("🪑 Office Chair", "office chair"),
            ("🛋️ Sofa Set", "sofa set"),
            ("⌨️ Mechanical Keyboard", "mechanical keyboard")
        ]
        
        for display, search_term in quick_searches:
            if st.button(display, key=f"quick_{search_term}", use_container_width=True):
                st.session_state.query = search_term
                st.rerun()
    
    # Handle quick search
    if 'query' in st.session_state:
        query = st.session_state.query
        del st.session_state.query
    
    # Main content area
    if not query:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Amazon Product Finder AI!
        
        This tool helps you find the **best products** on Amazon by analyzing:
        
        - 🌟 **Rating Quality** - Higher ratings from customers
        - 💰 **Price Value** - Best value for your money  
        - 📊 **Review Volume** - Products with more reviews (higher confidence)
        - 😊 **Customer Sentiment** - What customers actually say about the product
        
        ### How to use:
        1. Enter a product name in the sidebar (e.g., "gaming laptop")
        2. Set your price range
        3. Click "Find Best Products"
        4. Get AI-powered recommendations with detailed analysis!
        
        ### Features:
        - 🏆 **Smart Ranking** - AI combines multiple factors for best recommendations
        - 📈 **Visual Analytics** - Interactive charts and graphs
        - 🔍 **Deep Analysis** - Customer sentiment analysis from reviews
        - 📱 **Real-time Data** - Fresh data directly from Amazon
        
        **Ready to find your perfect product? Start by entering a search term in the sidebar! →**
        """)
        
        # Sample results preview
        st.subheader("🎯 Sample Results Preview")
        sample_data = {
            'Product': ['Gaming Laptop A', 'Gaming Laptop B', 'Gaming Laptop C'],
            'Price (₹)': [75000, 85000, 65000],
            'Rating': [4.5, 4.3, 4.1],
            'Reviews': [2847, 1523, 3241],
            'AI Score': [89.2, 84.7, 78.3]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        return
    
    # Validate inputs
    if min_price >= max_price:
        st.error("❌ Minimum price must be less than maximum price!")
        return
    
    # Search execution
    if search_button or query:
        st.header(f"🔍 Searching for: **{query}**")
        st.info(f"💰 Price Range: ₹{min_price:,} - ₹{max_price:,}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analyzer = get_analyzer()
        all_products = []
        
        # Phase 1: Search products
        status_text.text("🔍 Searching Amazon pages...")
        for page in range(1, max_pages + 1):
            progress_bar.progress((page / max_pages) * 0.4)  # 40% for searching
            status_text.text(f"📖 Scanning page {page}/{max_pages}...")
            
            html = fetch_amazon_html(query, min_price, max_price, page=page)
            if not html:
                st.warning(f"⚠️ Failed to fetch page {page}")
                break
                
            products = parse_amazon_html(html)
            filtered_products = [
                p for p in products 
                if p['price_val'] and min_price <= p['price_val'] <= max_price
            ]
            all_products.extend(filtered_products)
            
            # Add delay to be respectful
            time.sleep(1)
        
        if not all_products:
            st.error("❌ No products found matching your criteria. Try adjusting your search terms or price range.")
            return
        
        # Sort and get top products
        all_products.sort(key=lambda x: (x['rating_val'] * x['review_count']), reverse=True)
        top_products = all_products[:max_products]
        
        st.success(f"✅ Found {len(all_products)} total products, analyzing top {len(top_products)}")
        
        # Phase 2: Analyze customer insights
        status_text.text("📝 Analyzing customer insights...")
        for i, product in enumerate(top_products):
            progress_bar.progress(0.4 + ((i + 1) / len(top_products)) * 0.5)  # 50% for analysis
            status_text.text(f"🔄 Analyzing product {i+1}/{len(top_products)}...")
            
            customers_say = scrape_customers_say_section(product['link'])
            product['customers_say'] = customers_say
            
            # Add delay
            time.sleep(2)
        
        # Phase 3: Final ranking
        status_text.text("🏆 Generating final rankings...")
        progress_bar.progress(0.9)
        
        analyzed_products = analyzer.calculate_final_rankings(top_products)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Clear progress
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(analyzed_products, query)

def display_results(products, query):
    """Display the analysis results"""
    if not products:
        st.error("No products to display")
        return
    
    best_product = products[0]
    
    # AI Recommendation
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>🏆 AI RECOMMENDATION</h3>
        <h4>{best_product['title'][:80]}...</h4>
        <p><strong>Why this is the best choice:</strong></p>
        <ul>
            <li>🎯 <strong>Overall AI Score:</strong> {best_product['final_score']}/100</li>
            <li>💰 <strong>Price:</strong> ₹{best_product['price_val']:,.2f}</li>
            <li>⭐ <strong>Rating:</strong> {best_product['rating_val']}/5.0 ({best_product['review_count']:,} reviews)</li>
            <li>😊 <strong>Customer Sentiment:</strong> {best_product['sentiment_score']}/100</li>
        </ul>
        <p>This product offers the best combination of price, quality, and customer satisfaction based on our AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏆 Best Score", 
            f"{best_product['final_score']}/100",
            help="AI-calculated overall score"
        )
    
    with col2:
        avg_price = sum(p['price_val'] for p in products) / len(products)
        st.metric(
            "💰 Best Price", 
            f"₹{best_product['price_val']:,.0f}",
            delta=f"₹{best_product['price_val'] - avg_price:,.0f} vs avg",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "⭐ Best Rating", 
            f"{best_product['rating_val']}/5.0",
            help="Customer rating"
        )
    
    with col4:
        st.metric(
            "👥 Most Trusted", 
            f"{best_product['review_count']:,}",
            help="Number of customer reviews"
        )
    
    # Interactive charts
    st.subheader("📊 Product Analysis Charts")
    
    # Create dataframe for charts
    df = pd.DataFrame(products)
    
    # Chart tabs
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Score Breakdown", "💰 Price vs Rating", "📊 Detailed Metrics"])
    
    with chart_tab1:
        # Score breakdown chart
        # Create subplot with polar support
        fig_scores = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Overall AI Scores", "Score Components"),
            specs=[[{"type": "bar"}, {"type": "polar"}]]   # <-- use polar
        )

        # Example bar
        fig_scores.add_trace(
            go.Bar(x=["Prod A", "Prod B"], y=[80, 90], name="Overall Score"),
            row=1, col=1
        )

        # Example radar (scatterpolar)
        fig_scores.add_trace(
            go.Scatterpolar(
                r=[80, 70, 90, 85, 75],
                theta=["Quality", "Price", "Durability", "Features", "Support"],
                fill="toself",
                name="Product A"
            ),
            row=1, col=2
        )

        fig_scores.update_layout(height=600, width=1000, showlegend=True)
        
        # Radar chart for best product
        categories = ['Price Value', 'Rating Quality', 'Review Volume', 'Customer Sentiment']
        values = [
            best_product.get('price_value_score', 0),
            best_product.get('rating_quality_score', 0), 
            best_product.get('review_volume_score', 0),
            best_product.get('sentiment_score', 0)
        ]
        
        fig_scores.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"#{best_product['final_rank']} Best Product",
                line=dict(color='#ff6b6b')
            ),
            row=1, col=2
        )
        
        fig_scores.update_layout(height=400, showlegend=True)
        fig_scores.update_polars(radialaxis=dict(range=[0, 100]))
        
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with chart_tab2:
        # Price vs Rating scatter plot
        fig_scatter = px.scatter(
            df, 
            x='price_val', 
            y='rating_val',
            size='review_count',
            color='final_score',
            hover_data=['final_rank'],
            title="Price vs Rating Analysis",
            labels={
                'price_val': 'Price (₹)',
                'rating_val': 'Rating (out of 5)',
                'review_count': 'Review Count',
                'final_score': 'AI Score'
            },
            color_continuous_scale='RdYlBu_r'
        )
        
        fig_scatter.update_traces(
            hovertemplate='<b>Rank #%{customdata[0]}</b><br>' +
                         'Price: ₹%{x:,.0f}<br>' +
                         'Rating: %{y}/5<br>' +
                         'Reviews: %{marker.size}<br>' +
                         'AI Score: %{marker.color}/100<extra></extra>'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with chart_tab3:
        # Detailed metrics table
        metrics_df = pd.DataFrame([
            {
                'Rank': p['final_rank'],
                'Product': p['title'][:50] + '...',
                'Price (₹)': f"₹{p['price_val']:,.0f}",
                'Rating': f"{p['rating_val']}/5",
                'Reviews': f"{p['review_count']:,}",
                'AI Score': f"{p['final_score']}/100",
                'Price Value': f"{p.get('price_value_score', 0)}/100",
                'Sentiment': f"{p.get('sentiment_score', 0)}/100"
            } for p in products
        ])
        
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Detailed product cards
    st.subheader("🎯 Detailed Product Analysis")
    
    for i, product in enumerate(products):
        with st.expander(f"#{i+1} - {product['title'][:60]}... (Score: {product['final_score']}/100)", 
                        expanded=(i == 0)):  # Expand first product by default
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **🏷️ Product Title:** {product['title']}
                
                **💰 Price:** ₹{product['price_val']:,.2f}
                
                **⭐ Rating:** {product['rating_val']}/5.0 ({product['review_count']:,} reviews)
                
                **🎯 AI Analysis:**
                - Overall Score: {product['final_score']}/100
                - Price Value: {product.get('price_value_score', 0)}/100
                - Rating Quality: {product.get('rating_quality_score', 0)}/100
                - Review Confidence: {product.get('review_volume_score', 0)}/100
                - Customer Sentiment: {product.get('sentiment_score', 0)}/100
                """)
                
                # Customer insights
                customers_say = product.get('customers_say', {})
                if customers_say.get('positive_mentions'):
                    st.success(f"✅ **Customers Love:** {customers_say['positive_mentions'][:200]}...")
                
                if customers_say.get('negative_mentions'):
                    st.warning(f"⚠️ **Common Concerns:** {customers_say['negative_mentions'][:200]}...")
            
            with col2:
                # Quick metrics
                st.metric("Final Rank", f"#{product['final_rank']}")
                st.metric("AI Score", f"{product['final_score']}/100")
                
                # Amazon link
                st.link_button(
                    "🛒 View on Amazon",
                    product['link'],
                    type="primary",
                    use_container_width=True
                )
    
    # Export options
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download CSV
        csv_data = pd.DataFrame([
            {
                'Rank': p['final_rank'],
                'Title': p['title'],
                'Price': p['price_val'],
                'Rating': p['rating_val'],
                'Review_Count': p['review_count'],
                'AI_Score': p['final_score'],
                'Price_Value_Score': p.get('price_value_score', 0),
                'Sentiment_Score': p.get('sentiment_score', 0),
                'Amazon_Link': p['link'],
                'Search_Query': query,
                'Analysis_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            } for p in products
        ])
        
        st.download_button(
            "📊 Download CSV",
            csv_data.to_csv(index=False),
            file_name=f"amazon_analysis_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download JSON
        json_data = {
            'search_query': query,
            'analysis_date': datetime.now().isoformat(),
            'products': products,
            'summary': {
                'total_analyzed': len(products),
                'best_product': products[0]['title'],
                'best_score': products[0]['final_score'],
                'price_range': {
                    'min': min(p['price_val'] for p in products),
                    'max': max(p['price_val'] for p in products)
                }
            }
        }
        
        st.download_button(
            "📄 Download JSON",
            json.dumps(json_data, indent=2, ensure_ascii=False),
            file_name=f"amazon_analysis_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()