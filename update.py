
"""
USD/JPY カオス理論アンサンブル予測モデル
毎日自動実行 → docs/index.html を生成

依存: numpy scipy yfinance
"""

import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# ============================================================
# 1. データ取得
# ============================================================
def fetch_usdjpy():
    """yfinance でドル円日次データ取得（最大10年分）"""
    try:
        import yfinance as yf
        df = yf.download("USDJPY=X", period="10y", interval="1d", progress=False, auto_adjust=True)
        df = df.dropna()
        if len(df) < 100:
            raise ValueError("データ不足")
        closes = df["Close"].values.flatten()
        dates  = [d.strftime("%Y/%m/%d") for d in df.index]
        print(f"[OK] yfinance: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
        return dates, closes.astype(float)
    except Exception as e:
        print(f"[WARN] yfinance 失敗: {e}")
        return fetch_usdjpy_stooq()

def fetch_usdjpy_stooq():
    """stooq.com をフォールバックとして使用"""
    import urllib.request, csv, io
    url = "https://stooq.com/q/d/l/?s=usdjpy&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    rows = [r for r in rows[1:] if len(r) >= 5 and r[4] not in ("", "null")]
    rows.sort(key=lambda r: r[0])
    dates  = [r[0].replace("-", "/") for r in rows]
    closes = [float(r[4]) for r in rows]
    print(f"[OK] stooq: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
    return dates, np.array(closes)

# ============================================================
# 2. 特徴量計算
# ============================================================
def calc_features(prices):
    N = len(prices)
    # ローリングボラティリティ（20日）
    returns = np.diff(np.log(prices))
    roll_vol = np.array([
        np.std(returns[max(0,i-20):i]) * np.sqrt(252)
        for i in range(1, len(returns)+1)
    ])
    roll_vol = np.concatenate([[roll_vol[0]], roll_vol])  # N個に揃える

    # モメンタム（MA20/MA60乖離）
    ma20 = np.array([np.mean(prices[max(0,i-20):i+1]) for i in range(N)])
    ma60 = np.array([np.mean(prices[max(0,i-60):i+1]) for i in range(N)])
    momentum = (ma20 - ma60) / (ma60 + 1e-8) * 100

    # レジーム分類（33/67パーセンタイル）
    vq33 = np.percentile(roll_vol, 33)
    vq67 = np.percentile(roll_vol, 67)
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        v = roll_vol[i]
        regimes[i] = 0 if v < vq33 else (1 if v < vq67 else 2)

    return roll_vol, momentum, regimes, vq33, vq67

# ============================================================
# 3. 位相空間再構成
# ============================================================
TAU = 20
DIM = 3

def embed(prices_n, mom_n, tau=TAU, dim=DIM):
    n = len(prices_n) - (dim-1)*tau
    out = []
    for i in range(n):
        vec = [prices_n[i + j*tau] for j in range(dim)]
        vec.append(mom_n[i + (dim-1)*tau])
        out.append(vec)
    return np.array(out)

# ============================================================
# 4. 各モデル予測
# ============================================================
def predict_chaos(prices, momentum, regimes, steps=5, k=10):
    """位相空間 k-NN（レジーム重み付き）"""
    N = len(prices)
    offset = (DIM-1)*TAU
    p_mean, p_std = prices.mean(), prices.std()
    m_mean, m_std = momentum.mean(), momentum.std() + 1e-8
