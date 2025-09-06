from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np

app = Flask(__name__)

API_KEY = "4b1c09afd0ed4d89aa78370485ca2356"
BASE_URL = "https://api.twelvedata.com/time_series"

# -------------------- INDICATOR FUNCTIONS --------------------

def ema(series, period=9):
    return series.ewm(span=period, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k.iloc[-1], d.iloc[-1]

def bollinger(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper.iloc[-1], lower.iloc[-1], series.iloc[-1]

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    return ((tp - sma) / (0.015 * mad)).iloc[-1]

def atr(high, low, close, period=14):
    data = pd.DataFrame({"high": high, "low": low, "close": close})
    data["prev_close"] = data["close"].shift(1)
    data["tr"] = data[["high", "low", "close", "prev_close"]].apply(
        lambda x: max(x["high"] - x["low"], abs(x["high"] - x["prev_close"]), abs(x["low"] - x["prev_close"])), axis=1
    )
    return data["tr"].rolling(window=period).mean().iloc[-1]

# -------------------- SIGNAL LOGIC --------------------

def generate_signal(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Indicators
    ema9 = ema(close, 9).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    macd_line, signal_line = macd(close)
    rsi_val = rsi(close)
    k, d = stochastic(high, low, close)
    upper, lower, last_price = bollinger(close)
    cci_val = cci(high, low, close)
    atr_val = atr(high, low, close)

    # Conditions
    bullish = 0
    bearish = 0

    # EMA crossover
    if ema9 > ema21: bullish += 1
    else: bearish += 1

    # MACD
    if macd_line > signal_line: bullish += 1
    else: bearish += 1

    # RSI
    if rsi_val < 30: bullish += 1
    elif rsi_val > 70: bearish += 1

    # Stochastic
    if k < 20 and k > d: bullish += 1
    elif k > 80 and k < d: bearish += 1

    # Bollinger
    if last_price <= lower: bullish += 1
    elif last_price >= upper: bearish += 1

    # CCI
    if cci_val < -100: bullish += 1
    elif cci_val > 100: bearish += 1

    # ATR (just check volatility > 0)
    if atr_val > 0: 
        if bullish > bearish: bullish += 1
        else: bearish += 1

    # Final Signal
    if bullish > bearish:
        return "CALL", bullish, bearish
    elif bearish > bullish:
        return "PUT", bullish, bearish
    else:
        return "NEUTRAL", bullish, bearish

# -------------------- API ENDPOINT --------------------

@app.route("/signal", methods=["POST"])
def signal():
    data = request.json
    pair = data.get("pair", "USD/INR")
    timeframe = data.get("timeframe", "5min")

    url = f"{BASE_URL}?symbol={pair}&interval={timeframe}&apikey={API_KEY}&outputsize=100"
    response = requests.get(url).json()

    if "values" not in response:
        return jsonify({"error": "Failed to fetch data", "details": response}), 400

    df = pd.DataFrame(response["values"])
    df = df.astype(float).iloc[::-1]  # Reverse order
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]

    signal, bullish, bearish = generate_signal(df)

    return jsonify({
        "pair": pair,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": f"{bullish}/7 bullish vs {bearish}/7 bearish"
    })

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
