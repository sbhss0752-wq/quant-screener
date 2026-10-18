import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

def calculate_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def send_telegram_alert(token: str, chat_id: str, message: str) -> None:
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(endpoint, json=payload, timeout=10)
    if not response.ok:
        raise RuntimeError(f"텔레그램 메시지 전송 실패: {response.text}")

def run_quant_screen(universe: list) -> dict:
    candidates = []
    data = yf.download(universe, period="6mo", interval="1d", group_by="ticker", auto_adjust=True, threads=True)
    
    for ticker in universe:
        try:
            df = data[ticker].dropna()
            if len(df) < 30:
                continue
            
            close = df["Close"]
            current_price = float(close.iloc[-1])
            ret_1m = float((current_price - close.iloc[-21]) / close.iloc[-21])
            
            rsi_series = calculate_wilder_rsi(close, period=14)
            current_rsi = float(rsi_series.iloc[-1])
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            target_price = info.get("targetMeanPrice")
            forward_pe = info.get("forwardPE")
            roe = info.get("returnOnEquity")
            market_cap = info.get("marketCap", 0)
            
            if market_cap < 10_000_000_000:
                continue
            if roe is None or roe <= 0:
                continue
            if target_price is None or target_price <= current_price:
                continue
            
            upside = (target_price - current_price) / current_price
            
            score = (-ret_1m * 0.35) + ((50.0 - current_rsi) * 0.35) + (upside * 0.30)
            
            candidates.append({
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "price": current_price,
                "ret_1m": ret_1m,
                "rsi": current_rsi,
                "target_price": target_price,
                "upside": upside,
                "roe": roe,
                "forward_pe": forward_pe,
                "score": score
            })
        except Exception:
            continue
            
    if not candidates:
        return None
        
    candidates_df = pd.DataFrame(candidates)
    best_candidate = candidates_df.sort_values(by="score", ascending=False).iloc[0]
    return best_candidate.to_dict()

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        raise ValueError("환경 변수 TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
    
    universe_pool = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "PG", "HD", "DIS", "ADBE", "NFLX", "CRM", "INTC",
        "AMD", "PYPL", "NKE", "QCOM", "TXN", "AMAT", "COST", "PEP", "AVGO"
    ]
    
    result = run_quant_screen(universe_pool)
    
    if result is None:
        message = "⚠️ *퀀트 스크리닝 알림*: 현재 기준 조건을 충족하는 저평가 우량주가 유니버스 내에 없습니다."
    else:
        message = (
            f"📊 *[단기 역전 퀀트 모델 선별 종목]*\n\n"
            f"• *티커 (종목명)*: `{result['ticker']}` ({result['name']})\n"
            f"• *현재가*: `${result['price']:.2f}`\n"
            f"• *1개월 수익률*: `{result['ret_1m']*100:.2f}%` (과매도 영역)\n"
            f"• *Wilder RSI(14)*: `{result['rsi']:.1f}`\n"
            f"• *월가 컨센서스 목표가*: `${result['target_price']:.2f}`\n"
            f"• *상승 여력(괴리율)*: `+{result['upside']*100:.2f}%`\n"
            f"• *자기자본이익률(ROE)*: `{result['roe']*100:.2f}%`\n"
            f"• *Forward P/E*: `{result['forward_pe']}`\n\n"
            f"📌 *계량적 상태*: 우수한 기업 기초체력에도 불구하고 단기 수급 이탈로 인해 통계적 평균회귀 압력이 집중된 구간입니다."
        )
        
    send_telegram_alert(token, chat_id, message)

if __name__ == "__main__":
    main()
