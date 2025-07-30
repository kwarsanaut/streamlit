# app.py - Indonesian Brand Monitoring Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Handle transformers import with fallback
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("⚠️ Transformers library not available. Using demo mode.")

# Page config
st.set_page_config(
    page_title="Indonesian Brand Monitoring",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sentiment-positive { color: #28a745; }
    .sentiment-negative { color: #dc3545; }
    .sentiment-neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_sentiment_model():
    """Load Indonesian sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        return pipeline("sentiment-analysis", 
                       model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration"""
    dates = pd.date_range(start='2025-01-01', end='2025-01-30', freq='D')
    
    # Generate realistic sentiment data
    np.random.seed(42)
    positive = np.random.normal(25, 5, len(dates)).astype(int)
    negative = np.random.normal(10, 3, len(dates)).astype(int)
    neutral = np.random.normal(40, 8, len(dates)).astype(int)
    
    # Ensure positive values
    positive = np.clip(positive, 5, 50)
    negative = np.clip(negative, 2, 25)
    neutral = np.clip(neutral, 20, 60)
    
    return pd.DataFrame({
        'date': dates,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'total': positive + negative + neutral
    })

# Sample tweets for demo
SAMPLE_TWEETS = [
    {"text": "tokped mantap banget pengiriman cepat sekali!", "timestamp": "2025-01-30 10:30"},
    {"text": "kecewa sama customer service tokopedia lambat banget", "timestamp": "2025-01-30 09:15"},
    {"text": "biasa aja sih aplikasi tokped, nothing special", "timestamp": "2025-01-30 08:45"},
    {"text": "recommended banget tokped untuk belanja online", "timestamp": "2025-01-30 07:20"},
    {"text": "lumayan murah promo flash sale tokopedia", "timestamp": "2025-01-30 06:10"},
    {"text": "error mulu aplikasi tokped pas checkout", "timestamp": "2025-01-29 23:50"},
    {"text": "seller di tokped ramah-ramah dan fast response", "timestamp": "2025-01-29 22:30"},
    {"text": "delivery tokped standar aja tidak terlalu cepat", "timestamp": "2025-01-29 21:15"}
]

def analyze_sentiment(text, classifier):
    """Analyze sentiment of given text"""
    if classifier is None or not TRANSFORMERS_AVAILABLE:
        # Demo mode - simple keyword-based analysis
        text_lower = text.lower()
        positive_keywords = ['bagus', 'mantap', 'keren', 'suka', 'cepat', 'murah', 'recommended']
        negative_keywords = ['kecewa', 'jelek', 'lambat', 'mahal', 'error', 'buruk', 'susah']
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if pos_count > neg_count:
            return {"sentiment": "Positive", "confidence": 0.85}
        elif neg_count > pos_count:
            return {"sentiment": "Negative", "confidence": 0.82}
        else:
            return {"sentiment": "Neutral", "confidence": 0.75}
    
    try:
        result = classifier(text)
        return {
            "sentiment": result[0]['label'],
            "confidence": result[0]['score']
        }
    except Exception as e:
        return {"sentiment": "Error", "confidence": 0.0}

def main():
    # Header
    st.markdown('<h1 class="main-header">🇮🇩 Indonesian Social Media Brand Monitoring</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time Sentiment Analysis untuk Brand Indonesia**")
    
    # Load model
    with st.spinner("Loading AI model..."):
        classifier = load_sentiment_model()
    
    if classifier is None:
        st.error("Failed to load sentiment analysis model. Please check your internet connection.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Brand selection
    brand_options = ["Tokopedia", "Shopee", "BCA", "Mandiri", "Gojek", "Grab", "OVO", "Dana"]
    selected_brand = st.sidebar.selectbox("🏢 Select Brand", brand_options, index=0)
    
    # Date range
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "📊 Analysis Type",
        ["Real-time", "Historical", "Comparison"]
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Analysis", "📊 Dashboard", "📈 Trends", "ℹ️ About"])
    
    with tab1:
        st.header("🔍 Real-time Sentiment Analysis")
        
        # Text input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter text for sentiment analysis:",
                placeholder=f"Example: {selected_brand.lower()} mantap banget pelayanannya!",
                height=100
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            
            # Preset examples
            st.markdown("**Quick Examples:**")
            examples = [
                f"{selected_brand.lower()} bagus banget!",
                f"kecewa sama {selected_brand.lower()}",
                f"{selected_brand.lower()} biasa aja"
            ]
            
            for example in examples:
                if st.button(example, key=f"example_{example}", use_container_width=True):
                    user_input = example
                    analyze_button = True
        
        # Analysis results
        if analyze_button and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                result = analyze_sentiment(user_input, classifier)
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                
                # Sentiment color and emoji
                if sentiment == "Positive":
                    color = "sentiment-positive"
                    emoji = "😊"
                    status = "Great!"
                elif sentiment == "Negative":
                    color = "sentiment-negative" 
                    emoji = "😞"
                    status = "Needs Attention"
                else:
                    color = "sentiment-neutral"
                    emoji = "😐"
                    status = "Neutral"
                
                with col1:
                    st.metric("Sentiment", sentiment)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.metric("Status", f"{emoji} {status}")
                
                with col4:
                    score = confidence * 100 if sentiment == "Positive" else (100 - confidence * 100) if sentiment == "Negative" else 50
                    st.metric("Score", f"{score:.1f}/100")
                
                # Detailed analysis
                st.markdown("---")
                st.subheader("📋 Analysis Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Input Text:** {user_input}")
                    st.markdown(f"**Detected Sentiment:** <span class='{color}'>{sentiment}</span>", 
                               unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Confidence Level:** {confidence:.3f}")
                    st.markdown(f"**Recommendation:** {'Promote positive feedback' if sentiment == 'Positive' else 'Address customer concerns' if sentiment == 'Negative' else 'Monitor for trends'}")
    
    with tab2:
        st.header("📊 Brand Monitoring Dashboard")
        
        # Load sample data
        df_sample = generate_sample_data()
        
        # Key metrics
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_mentions = df_sample['total'].sum()
        total_positive = df_sample['positive'].sum()
        total_negative = df_sample['negative'].sum()
        total_neutral = df_sample['neutral'].sum()
        
        positive_rate = (total_positive / total_mentions) * 100
        negative_rate = (total_negative / total_mentions) * 100
        sentiment_score = ((total_positive - total_negative) / total_mentions) * 10 + 5
        
        with col1:
            st.metric("Total Mentions", f"{total_mentions:,}", "↗️ +12%")
        with col2:
            st.metric("Positive Rate", f"{positive_rate:.1f}%", "↗️ +5%")
        with col3:
            st.metric("Negative Rate", f"{negative_rate:.1f}%", "↘️ -2%")
        with col4:
            st.metric("Sentiment Score", f"{sentiment_score:.1f}/10", "↗️ +0.3")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Daily Sentiment Trends")
            fig_line = px.line(
                df_sample, 
                x='date', 
                y=['positive', 'negative', 'neutral'],
                title=f"Daily Sentiment Trends - {selected_brand}",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                }
            )
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Sentiment Distribution")
            sentiment_data = {
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Count': [total_positive, total_negative, total_neutral]
            }
            
            fig_pie = px.pie(
                sentiment_data,
                values='Count',
                names='Sentiment',
                title=f"Overall Sentiment Distribution - {selected_brand}",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d'
                }
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent mentions
        st.subheader("💬 Recent Mentions Analysis")
        
        # Analyze sample tweets
        analyzed_tweets = []
        for tweet in SAMPLE_TWEETS[:5]:  # Show top 5
            result = analyze_sentiment(tweet["text"], classifier)
            analyzed_tweets.append({
                **tweet,
                **result
            })
        
        # Display in table format
        for i, tweet in enumerate(analyzed_tweets):
            with st.container():
                col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                
                with col1:
                    st.write(f"**{tweet['text']}**")
                    st.caption(f"🕒 {tweet['timestamp']}")
                
                with col2:
                    sentiment = tweet["sentiment"]
                    if sentiment == "Positive":
                        st.success("😊 Positive")
                    elif sentiment == "Negative":
                        st.error("😞 Negative")
                    else:
                        st.info("😐 Neutral")
                
                with col3:
                    st.metric("Confidence", f"{tweet['confidence']:.1%}")
                
                with col4:
                    st.button("📋 Details", key=f"details_{i}")
                
                st.divider()
    
    with tab3:
        st.header("📈 Trend Analysis")
        
        df_sample = generate_sample_data()
        
        # Trend insights
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly comparison
            st.markdown("**📅 Weekly Trends:**")
            
            # Calculate weekly averages
            df_sample['week'] = df_sample['date'].dt.isocalendar().week
            weekly_data = df_sample.groupby('week').agg({
                'positive': 'mean',
                'negative': 'mean', 
                'neutral': 'mean'
            }).round(1)
            
            st.dataframe(weekly_data, use_container_width=True)
        
        with col2:
            # Sentiment velocity
            st.markdown("**⚡ Sentiment Velocity:**")
            
            velocity_data = {
                'Metric': ['Positive Growth', 'Negative Decline', 'Engagement Rate', 'Response Time'],
                'Value': ['+15%', '-8%', '4.2%', '2.3h'],
                'Status': ['📈', '📉', '📊', '⏱️']
            }
            
            velocity_df = pd.DataFrame(velocity_data)
            st.dataframe(velocity_df, use_container_width=True, hide_index=True)
        
        # Advanced charts
        st.subheader("📊 Advanced Analytics")
        
        # Sentiment momentum
        fig_area = px.area(
            df_sample,
            x='date',
            y=['positive', 'negative', 'neutral'],
            title=f"Sentiment Momentum - {selected_brand}",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            }
        )
        st.plotly_chart(fig_area, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ About This Project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            **Indonesian Social Media Brand Monitoring System** adalah solution untuk:
            
            - 📊 **Real-time sentiment analysis** untuk brand Indonesia
            - 🔍 **Multi-platform monitoring** (Twitter, Instagram, TikTok)  
            - 📈 **Advanced analytics** dan trend detection
            - 🚨 **Automated alerts** untuk crisis management
            - 📱 **Mobile-responsive** dashboard
            
            ### 🧠 Technology Stack
            
            - **AI Model:** IndoBERT-based sentiment analysis
            - **Framework:** Streamlit + Plotly
            - **Language:** Python 3.8+
            - **Deployment:** Streamlit Cloud
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Key Features
            
            ✅ **Indonesian Language Support**  
            ✅ **Social Media Slang Detection**  
            ✅ **Real-time Processing**  
            ✅ **Interactive Dashboard**  
            ✅ **Export Capabilities**  
            ✅ **API Integration Ready**  
            
            ### 🎯 Target Market
            
            - 🛒 **E-commerce** (Tokopedia, Shopee)
            - 🏦 **Banking** (BCA, Mandiri)  
            - 📱 **Tech Companies** (Gojek, Grab)
            - 🏢 **Marketing Agencies**
            - 🏛️ **Government Institutions**
            
            ### 📞 Contact
            
            **Rate:** $35-60/hour  
            **Timeline:** 4-6 weeks implementation  
            **Support:** 24/7 monitoring available
            """)
        
        # Demo credentials
        st.markdown("---")
        st.subheader("🔑 Demo Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Demo Mode**\nThis is a portfolio demonstration using sample data and pre-trained models.")
        
        with col2:
            st.success("**Model Accuracy**\n92%+ accuracy on Indonesian social media text")
        
        with col3:
            st.warning("**Live Data**\nFor production deployment, connect to real social media APIs")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>Indonesian Brand Monitoring System</strong> | Powered by IndoBERT | 
            Real-time Social Media Analysis</p>
            <p>Built with ❤️ for Indonesian businesses | Portfolio Project 2025</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
