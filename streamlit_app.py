import streamlit as st
import requests
import pandas as pd
import altair as alt

st.set_page_config(page_title="Pump & Dump Detector (Demo)", layout="wide", page_icon="📈")

st.title("📈 Advanced Pump & Dump Detection Dashboard")
st.markdown("Analyze extreme stock movements and separate genuine growth from artificial manipulation.")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., NAVA, SUZLON)", "NAVA")

mode = st.radio("Select View Mode:", ["Beginner Mode", "Advanced Mode"], horizontal=True)

if st.button("Analyze Stock"):
    with st.spinner(f"Analyzing {stock_symbol}..."):
        try:
            response = requests.get(f"http://localhost:8000/analyze?stock={stock_symbol}")
            if response.status_code == 200:
                data = response.json()
                
                col1, col2 = st.columns([1, 1])
                
                risk_score = data.get("risk_score", 0)
                decision = data.get("finalDecision", "Unknown")
                
                if risk_score <= 40:
                    color = "green"
                elif risk_score <= 70:
                    color = "orange"
                else:
                    color = "red"
                
                with col1:
                    st.subheader(f"Results for {stock_symbol}")
                    st.markdown(f"### Overall Risk Score: <span style='color:{color}'>{risk_score} / 100</span>", unsafe_allow_html=True)
                    st.progress(risk_score / 100)
                    st.markdown(f"**System Decision:** {decision}")
                    
                    st.subheader("Why Flagged?")
                    
                    if mode == "Beginner Mode":
                        st.write(f"- The stock's price increased by **{data['price_change_pct']}%** recently.")
                        st.write(f"- Trading volume is **{data['volume_ratio']}x** higher than normal.")
                        
                        if data['momentum_detected']:
                            st.write("- Strong positive momentum was detected over the past few days.")
                        else:
                            st.write("- No sustained momentum over the past few days.")
                            
                        st.write(f"- RSI indicator value is **{data['rsi_value']}** (Values over 50 add to pump likelihood).")
                        
                        sentiment_score = data['sentimentScore']
                        sentiment_label = data['sentimentLabel']
                        st.write(f"- News Sentiment: **{sentiment_label}**")
                        
                        if sentiment_score > 0.3:
                            st.info("This rise is supported by strong positive news, hence risk level is lower.")
                        elif sentiment_score >= 0:
                            st.warning("This rise is NOT strongly supported by news, keeping risk elevated.")
                        else:
                            st.error("This rise happens amidst negative news, significantly increasing manipulation risk.")
                            
                    else: # Advanced Mode
                        st.write("Detailed Technical Breakdown:")
                        st.write(f"- Price Change: {data['price_change_pct']}%  → **Score: +{data['price_score']}**")
                        st.write(f"- Volume Ratio: {data['volume_ratio']}x  → **Score: +{data['volume_score']}**")
                        st.write(f"- Momentum Detected: {data['momentum_detected']} → **Score: +{data['momentum_score']}**")
                        st.write(f"- RSI Value: {data['rsi_value']} → **Score: +{data['rsi_score']}**")
                        
                        base_pump = data['price_score'] + data['volume_score'] + data['momentum_score'] + data['rsi_score']
                        st.write(f"**Base Pump Score = {min(base_pump, 100)}**")
                        
                        sentiment_score = data['sentimentScore']
                        if sentiment_score > 0.3:
                            sentiment_adj = -20
                        elif sentiment_score >= 0:
                            sentiment_adj = -10
                        else:
                            sentiment_adj = 10
                            
                        st.write(f"- News Sentiment Adjustment ({sentiment_score}): **{sentiment_adj}**")
                        st.write(f"**Final Risk Score (Base + Adj) = {risk_score}**")
                        
                with col2:
                    st.subheader("Price Overview")
                    chart_data = data.get("chart_data", [])
                    if chart_data:
                        df = pd.DataFrame(chart_data)
                        line_chart = alt.Chart(df).mark_line().encode(
                            x=alt.X('date:T', title='Date'),
                            y=alt.Y('close:Q', scale=alt.Scale(zero=False), title='Closing Price')
                        ).properties(
                            height=300
                        )
                        st.altair_chart(line_chart, use_container_width=True)
                        
                    st.subheader("Filtered Relevant News")
                    top_articles = data.get("topArticles", [])
                    if len(top_articles) > 0:
                        for idx, art in enumerate(top_articles):
                            st.write(f"{idx+1}. [{art['title']}]({art['url']})")
                    else:
                        st.write("No relevant highly correlated news found.")
                        
            else:
                st.error(f"Error fetching data: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
