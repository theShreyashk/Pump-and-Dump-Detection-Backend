from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from email_utils import send_alert_email
import os
import requests
from textblob import TextBlob
from dotenv import load_dotenv
import nltk

load_dotenv()

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def fetch_news_sentiment(stock_symbol: str) -> dict:
    if not NEWSAPI_KEY:
        return {
            "sentimentScore": 0.0,
            "sentimentLabel": "No Data",
            "articleCount": 0,
            "finalDecision": "Uncertain ⚠️",
            "topArticles": []
        }
        
    query = stock_symbol.replace(".NS", "").replace(".BO", "")
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={NEWSAPI_KEY}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return {"sentimentScore": 0.0, "sentimentLabel": "No Data", "articleCount": 0, "finalDecision": "Uncertain ⚠️", "topArticles": []}
            
        data = response.json()
        articles = data.get("articles", [])
        
        keywords = [query.lower(), "steel", "mining", "ferro", "power", "metal"]
        filtered_articles = []
        for article in articles:
            title = (article.get("title") or "").lower()
            description = (article.get("description") or "").lower()
            
            if any(kw in title or kw in description for kw in keywords):
                filtered_articles.append(article)
                
        article_count = len(filtered_articles)
        
        if article_count == 0:
            return {"sentimentScore": 0.0, "sentimentLabel": "No Data", "articleCount": 0, "finalDecision": "Uncertain ⚠️", "topArticles": []}
            
        top_articles = []
        for i, article in enumerate(filtered_articles):
            if i < 3:
                top_articles.append({
                    "title": article.get("title") or "Unknown Title",
                    "url": article.get("url") or "#"
                })
            
        total_polarity = 0
        for article in filtered_articles:
            title = article.get("title") or ""
            blob = TextBlob(title)
            total_polarity += blob.sentiment.polarity
            
        avg_polarity = total_polarity / article_count
        
        if avg_polarity > 0.3:
            sentimentLabel = "Strong Positive"
        elif avg_polarity > 0:
            sentimentLabel = "Weak Positive"
        elif avg_polarity == 0:
            sentimentLabel = "Neutral"
        elif avg_polarity >= -0.3:
            sentimentLabel = "Weak Negative"
        else:
            sentimentLabel = "Strong Negative"
            
        return {
            "sentimentScore": round(avg_polarity, 2),
            "sentimentLabel": sentimentLabel,
            "articleCount": article_count,
            "topArticles": top_articles,
            "finalDecision": "" 
        }
    except Exception:
        return {"sentimentScore": 0.0, "sentimentLabel": "No Data", "articleCount": 0, "finalDecision": "Uncertain ⚠️", "topArticles": []}


app = FastAPI(title="Pump and Dump Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

watchlist_db = []
alerted_set = set()
users_db = {}

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class StockEmailRequest(BaseModel):
    stock: str
    email: str

class ChartDataPoint(BaseModel):
    date: str
    close: float
    volume: int
    is_pump: bool

class AnalyzeResponse(BaseModel):
    stock: str
    pump_probability: int
    dump_risk: str
    reason: str
    chart_data: List[ChartDataPoint]
    sentimentLabel: Optional[str] = None
    finalDecision: Optional[str] = None
    sentimentScore: Optional[float] = None
    articleCount: Optional[int] = None
    topArticles: Optional[List[dict]] = None
    price_score: Optional[int] = 0
    volume_score: Optional[int] = 0
    momentum_score: Optional[int] = 0
    rsi_score: Optional[int] = 0
    rsi_value: Optional[float] = 0.0
    price_change_pct: Optional[float] = 0.0
    volume_ratio: Optional[float] = 0.0
    momentum_detected: Optional[bool] = False
    risk_score: Optional[int] = 0


def calculate_base_pump_score(df, recent_day):
    price_change = float(recent_day['Price_Change_Pct'])
    volume_ratio = float(recent_day['Volume_Ratio'])
    rsi_value = float(recent_day['RSI']) if pd.notna(recent_day['RSI']) else 50.0

    price_score = 0
    if price_change >= 10.0: price_score = 40
    elif price_change >= 5.0: price_score = 20
        
    volume_score = 0
    if volume_ratio >= 4.0: volume_score = 40
    elif volume_ratio >= 2.0: volume_score = 20
        
    last_3 = df.loc[:recent_day.name].tail(3)
    green_candles = sum((last_3['Close'] > last_3['Open']).tolist())
    momentum_detected = (green_candles >= 2)
    momentum_score = 20 if momentum_detected else 0
        
    rsi_score = 10 if rsi_value > 50 else 0
    
    pump_score = price_score + volume_score + momentum_score + rsi_score
    pump_score = min(pump_score, 100)
    
    return {
        "price_change": price_change,
        "volume_ratio": volume_ratio,
        "rsi_value": rsi_value,
        "price_score": price_score,
        "volume_score": volume_score,
        "momentum_score": momentum_score,
        "momentum_detected": momentum_detected,
        "rsi_score": rsi_score,
        "pump_score": pump_score
    }

def calculate_risk_decision(sentiment_val, pump_score):
    if sentiment_val > 0.30:
        sentiment_adj = -20
    elif sentiment_val >= 0.0:
        sentiment_adj = -10
    else:
        sentiment_adj = 10
        
    risk_score = pump_score + sentiment_adj
    risk_score = max(0, min(risk_score, 100))
    
    if risk_score <= 40:
        finalDecision = "Normal ✅"
        dump_risk = "Low"
    elif risk_score <= 70:
        finalDecision = "Suspicious ⚠️"
        dump_risk = "Medium"
    else:
        finalDecision = "High Pump / Manipulation Risk 🚨"
        dump_risk = "High"
        
    return risk_score, dump_risk, finalDecision

def prepare_dataframe(df):
    df = df.copy()
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    avg_volume_10d = df['Volume'].rolling(window=10).mean()
    df['Avg_Volume'] = avg_volume_10d.shift(1).fillna(df['Volume'].mean()) 
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def get_stock_analysis(stock: str) -> AnalyzeResponse:
    if not stock.endswith(".NS") and not stock.endswith(".BO"):
        symbol = stock + ".NS"
    else:
        symbol = stock

    ticker = yf.Ticker(symbol)
    df = ticker.history(period="60d")
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for stock: {stock}")
    if len(df) < 15:
        raise HTTPException(status_code=400, detail="Not enough historical data for analysis.")

    df = prepare_dataframe(df)
    df['Is_Pump'] = (df['Price_Change_Pct'] > 10.0) & (df['Volume_Ratio'] > 2.0)
    recent_day = df.iloc[-1]
    
    metrics = calculate_base_pump_score(df, recent_day)
    sentiment_data = fetch_news_sentiment(stock)
    sentiment_val = sentiment_data.get("sentimentScore", 0.0)
    
    risk_score, dump_risk, finalDecision = calculate_risk_decision(sentiment_val, metrics['pump_score'])

    reason_text = "Gradual scoring applied. Explanations available in UI details."

    chart_df = df.tail(30)
    chart_data = []
    for date, row in chart_df.iterrows():
        chart_data.append(ChartDataPoint(
            date=date.strftime("%Y-%m-%d"),
            close=round(row['Close'], 2),
            volume=int(row['Volume']),
            is_pump=bool(row['Is_Pump'])
        ))

    return AnalyzeResponse(
        stock=stock,
        pump_probability=metrics['pump_score'],
        dump_risk=dump_risk,
        reason=reason_text,
        chart_data=chart_data,
        sentimentLabel=sentiment_data.get("sentimentLabel"),
        finalDecision=finalDecision,
        sentimentScore=sentiment_val,
        articleCount=sentiment_data.get("articleCount"),
        topArticles=sentiment_data.get("topArticles"),
        price_score=metrics['price_score'],
        volume_score=metrics['volume_score'],
        momentum_score=metrics['momentum_score'],
        rsi_score=metrics['rsi_score'],
        rsi_value=round(float(metrics['rsi_value']), 2),
        price_change_pct=round(metrics['price_change'], 2),
        volume_ratio=round(metrics['volume_ratio'], 2),
        momentum_detected=metrics['momentum_detected'],
        risk_score=risk_score
    )

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze_stock(stock: str):
    try:
        return get_stock_analysis(stock)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signup")
def signup(user: UserCreate):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    users_db[user.email] = { "username": user.username, "email": user.email, "password": user.password }
    return {"message": "User registered successfully"}

@app.post("/login")
def login(user: UserLogin):
    if user.email not in users_db:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    stored_user = users_db[user.email]
    if stored_user["password"] != user.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return { "message": "Login successful", "username": stored_user["username"], "email": stored_user["email"], "token": "mock-jwt-token" }

@app.post("/add-stock")
def add_stock(request: StockEmailRequest):
    stock = request.stock.upper()
    if not stock.endswith(".NS") and not stock.endswith(".BO"):
        stock += ".NS"
    entry = {"stock": stock, "email": request.email}
    if entry not in watchlist_db:
        watchlist_db.append(entry)
    return {"message": f"Stock {stock} added to watchlist successfully for {request.email}."}

@app.get("/watchlist")
def get_watchlist():
    response = []
    for entry in watchlist_db:
        alerted = (entry['stock'], entry['email']) in alerted_set
        response.append({
            "stock": entry["stock"],
            "email": entry["email"],
            "status": "Alert Triggered" if alerted else "Monitoring"
        })
    return response

@app.delete("/remove-stock")
def remove_stock(request: StockEmailRequest):
    stock = request.stock.upper()
    if not stock.endswith(".NS") and not stock.endswith(".BO"):
        stock += ".NS"
    entry = {"stock": stock, "email": request.email}
    if entry in watchlist_db:
        watchlist_db.remove(entry)
    alert_key = (stock, request.email)
    if alert_key in alerted_set:
        alerted_set.remove(alert_key)
    return {"message": "Alert removed successfully"}

async def monitor_stocks():
    while True:
        await asyncio.sleep(300)
        for entry in watchlist_db:
            stock = entry["stock"]
            email = entry["email"]
            alert_key = (stock, email)
            if alert_key in alerted_set:
                continue
            try:
                analysis = get_stock_analysis(stock)
                # Using risk_score >= 71 for High Risk alerts
                if analysis.risk_score >= 71:
                    send_alert_email(email, stock, int(analysis.risk_score), analysis.dump_risk, analysis.reason)
                    alerted_set.add(alert_key)
            except Exception as e:
                print(f"Error analyzing {stock} in background task: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_stocks())

PENNY_STOCK_POOL = [
    "SUZLON.NS", "YESBANK.NS", "IDEA.NS", "GTLINFRA.NS", "JPPOWER.NS",
    "RPOWER.NS", "VIKASLIFE.NS", "HCC.NS", "RELINFRA.NS", "DISHTV.NS",
    "TRIDENT.NS", "UCOBANK.NS", "IOB.NS", "SOUTHBANK.NS", "URJA.NS"
]
SMALL_CAP_POOL = [
    "NBCC.NS", "BHEL.NS", "PNB.NS", "NMDC.NS", "IRFC.NS",
    "RVNL.NS", "BANKIOM.NS", "ZOMATO.NS", "SUBCG.NS", "BAJAJHIND.NS",
    "RENUKA.NS", "MRPL.NS", "NHPC.NS", "SJVN.NS", "HUDCO.NS",
    "AWL.NS", "IRCTC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HAL.NS"
]
LARGE_CAP_POOL = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS"
]
ALL_POOLS = PENNY_STOCK_POOL + SMALL_CAP_POOL + LARGE_CAP_POOL

@app.get("/top-pump-stocks")
def get_top_pump_stocks():
    try:
        data = yf.download(ALL_POOLS, period="2mo", group_by="ticker", progress=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

    results = []
    for stock in ALL_POOLS:
        try:
            if hasattr(data.columns, 'levels') and stock in data.columns.levels[0]:
                df = data[stock].dropna()
            elif not hasattr(data.columns, 'levels') and stock == ALL_POOLS[0]:
                df = data.dropna()
            else: continue
            if len(df) < 15: continue
            
            df = prepare_dataframe(df)
            recent_day = df.iloc[-1]
            metrics = calculate_base_pump_score(df, recent_day)
            
            sentiment_data = fetch_news_sentiment(stock)
            sentiment_val = sentiment_data.get("sentimentScore", 0.0)
            
            risk_score, dump_risk, finalDecision = calculate_risk_decision(sentiment_val, metrics["pump_score"])
            
            # High Risk Only according to dashboard chart condition (10% surge & 2x volume)
            if metrics["price_change"] > 10.0 and metrics["volume_ratio"] > 2.0:
                results.append({
                    "id": stock, "symbol": stock, 
                    "price": round(float(recent_day['Close']), 2),
                    "marketCap": "--", 
                    "change": round(metrics["price_change"], 2),
                    "volumeRatio": round(metrics["volume_ratio"], 1), 
                    "pumpProb": metrics["pump_score"], # Keeping field name but representing Base Score
                    "risk_score": risk_score,
                    "risk": dump_risk, "isFallback": False,
                    "sentimentLabel": sentiment_data["sentimentLabel"],
                    "finalDecision": finalDecision,
                    "sentimentScore": sentiment_val,
                    "articleCount": sentiment_data["articleCount"],
                    "topArticles": sentiment_data["topArticles"]
                })
        except Exception: 
            pass
        
    results.sort(key=lambda x: x['risk_score'], reverse=True)
    return {"stocks": results[:5], "fallback": False}

@app.get("/historical-pump-stocks")
def get_historical_pump_stocks(period: str = Query("1mo", regex="^(1mo|3mo|6mo)$")):
    try:
        data = yf.download(ALL_POOLS, period=period, group_by="ticker", progress=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")
        
    historical_results = []
    
    for stock in ALL_POOLS:
        try:
            if hasattr(data.columns, 'levels') and stock in data.columns.levels[0]:
                df = data[stock].dropna()
            elif not hasattr(data.columns, 'levels') and stock == ALL_POOLS[0]:
                df = data.dropna()
            else: continue
            
            if len(df) < 15: continue
            
            df = prepare_dataframe(df)
            
            # Iterate through history, skip the first 14 days (RSI warmup)
            for i in range(14, len(df)):
                recent_day = df.iloc[i]
                metrics = calculate_base_pump_score(df.iloc[:i+1], recent_day)
                base_pump_score = metrics['pump_score']
                
                # High Risk Only (Dashboard conditions: 10% surge & 2x volume)
                if metrics["price_change"] > 10.0 and metrics["volume_ratio"] > 2.0:
                    date_of_pump = df.index[i].strftime('%b %d, %Y')
                    historical_results.append({
                        "id": f"{stock}_{date_of_pump}",
                        "symbol": stock,
                        "date": date_of_pump,
                        "raw_date": df.index[i],
                        "change": round(metrics["price_change"], 2),
                        "volumeRatio": round(metrics["volume_ratio"], 1),
                        "pumpProb": base_pump_score, # Historical Base Score
                        "risk": "High"
                    })
        except Exception:
            continue
            
    # Sort by recent
    historical_results.sort(key=lambda x: x['raw_date'], reverse=True)
    
    return {"stocks": historical_results}
