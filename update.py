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

    pn = (prices  - p_mean) / p_std
    mn = (momentum - m_mean) / m_std
    Xe = embed(pn, mn)
    Xo = prices[offset:]
    Xr = regimes[offset:]
    cur_r = int(regimes[-1])

    dists = np.linalg.norm(Xe[:-1] - Xe[-1], axis=1)
    penalty = np.where(Xr[:-1] == cur_r, 1.0, 2.0)
    adj = dists * penalty
    adj[-20:] = np.inf
    nn_all = np.argsort(adj)
    valid_nn = [j for j in nn_all if j + steps < len(Xo)][:k]

    preds = []
    for s in range(1, steps+1):
        vals = [Xo[j+s] for j in valid_nn if j+s < len(Xo)]
        if not vals:
            vals = [prices[-1]]
        ws = np.array([1.0 / (adj[j] + 1e-8) for j in valid_nn if j+s < len(Xo)])
        ws /= ws.sum()
        mean = float(np.average(vals, weights=ws))
        std  = float(np.sqrt(np.average((np.array(vals)-mean)**2, weights=ws)))
        p10  = float(np.percentile(vals, 10))
        p90  = float(np.percentile(vals, 90))
        preds.append({"mean": mean, "std": std, "p10": p10, "p90": p90})
    return preds

def predict_ar(prices, p=5, steps=5):
    """AR(p) 最小二乗"""
    y = prices[-300:]
    Xar = np.array([y[i:i+p] for i in range(len(y)-p)])
    yar = y[p:]
    A   = np.column_stack([Xar, np.ones(len(Xar))])
    coef, _, _, _ = np.linalg.lstsq(A, yar, rcond=None)
    hist = list(prices[-p:])
    out  = []
    for _ in range(steps):
        v = float(np.dot(coef, hist[-p:] + [1.0]))
        out.append(v)
        hist.append(v)
    return out

def predict_momentum(prices, steps=5):
    """短期モメンタム + 平均回帰"""
    st = (prices[-1] - prices[-4]) / 3
    lt = (prices[-1] - prices[-11]) / 10
    drift = 0.5*st + 0.5*lt
    mean200 = np.mean(prices[-200:])
    reversion = (mean200 - prices[-1]) * 0.01
    drift += reversion
    out = []
    last = prices[-1]
    for s in range(1, steps+1):
        last = last + drift * (0.9**s)
        out.append(float(last))
    return out

# ============================================================
# 5. バックテスト最適化（毎回自動実行）
# ============================================================
def backtest_optimize(prices, momentum, regimes, n_bt=150):
    """
    直近 n_bt 日でモデルごとの誤差を計測し、
    グリッドサーチでRMSEを最小化するアンサンブル重みを返す
    """
    N = len(prices)
    offset = (DIM-1)*TAU
    p_mean, p_std = prices.mean(), prices.std()
    m_mean, m_std = momentum.mean(), momentum.std() + 1e-8

    ec, ea, em = [], [], []

    for i in range(n_bt, 0, -1):
        idx = N - i
        if idx < offset + 50:
            continue

        # カオス (修正箇所: predict_chaos から辞書ではなく数値を正しく取得)
        cp_dict = predict_chaos(prices[:idx], momentum[:idx], regimes[:idx], steps=1)
        cp = cp_dict[0]["mean"] if cp_dict else prices[idx-1]

        # AR
        ap = predict_ar(prices[:idx], steps=1)[0]

        # モメンタム
        mp = predict_momentum(prices[:idx], steps=1)[0]

        actual = prices[idx]
        ec.append(actual - cp)
        ea.append(actual - ap)
        em.append(actual - mp)

    ec, ea, em = np.array(ec), np.array(ea), np.array(em)

    # グリッドサーチ（0.1刻み）
    best_w    = (0.2, 0.6, 0.2)
    if len(ec) > 0:
        best_rmse = 1e9
        for wc in range(0, 11):
            for wa in range(0, 11 - wc):
                wm = 10 - wc - wa
                if wm < 0:
                    continue
                wc_f, wa_f, wm_f = wc/10, wa/10, wm/10
                rmse = float(np.sqrt(np.mean((wc_f*ec + wa_f*ea + wm_f*em)**2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_w    = (wc_f, wa_f, wm_f)
    
    best_rmse = float(np.sqrt(np.mean((best_w[0]*ec + best_w[1]*ea + best_w[2]*em)**2))) if len(ec) > 0 else 0.0

    weights = {"chaos": best_w[0], "ar": best_w[1], "momentum": best_w[2]}
    bt_stats = {
        "chaos":    {"rmse": float(np.sqrt(np.mean(ec**2))) if len(ec) > 0 else 0.0, "mae": float(np.mean(np.abs(ec))) if len(ec) > 0 else 0.0,
                     "da": float(np.mean(np.sign(ec) == np.sign(ea))) if len(ec) > 0 else 0.5},  # 相対
        "ar":       {"rmse": float(np.sqrt(np.mean(ea**2))) if len(ea) > 0 else 0.0, "mae": float(np.mean(np.abs(ea))) if len(ea) > 0 else 0.0,
                     "da":   _direction_accuracy(ea, prices, n_bt)},
        "momentum": {"rmse": float(np.sqrt(np.mean(em**2))) if len(em) > 0 else 0.0, "mae": float(np.mean(np.abs(em))) if len(em) > 0 else 0.0,
                     "da":   _direction_accuracy(em, prices, n_bt)},
        "ensemble": {"rmse": best_rmse}
    }
    return weights, bt_stats

def _direction_accuracy(errors, prices, n_bt):
    N = len(prices)
    correct = 0
    total   = 0
    for i, err in enumerate(errors):
        idx = N - n_bt + i
        if idx >= N - 1 or idx < 0:
            continue
        actual_dir = prices[idx+1] - prices[idx]
        pred_dir   = (prices[idx+1] - err) - prices[idx]
        if actual_dir * pred_dir > 0:
            correct += 1
        total += 1
    return float(correct / total) if total > 0 else 0.5

# ============================================================
# 6. アンサンブル統合
# ============================================================
def ensemble_predict(chaos_preds, ar_preds, mom_preds, weights, current_regime):
    regime_vol_mult = {0: 1.0, 1: 1.3, 2: 1.8}
    vm = regime_vol_mult[current_regime]
    wc, wa, wm = weights["chaos"], weights["ar"], weights["momentum"]
    out = []
    for s in range(5):
        cp = chaos_preds[s]["mean"]
        ap = ar_preds[s]
        mp = mom_preds[s]
        mean = wc*cp + wa*ap + wm*mp
        std  = chaos_preds[s]["std"] * vm
        out.append({
            "mean":     round(mean, 4),
            "std":      round(std,  4),
            "p10":      round(mean - 1.282*std, 4),
            "p90":      round(mean + 1.282*std, 4),
            "chaos":    round(cp, 4),
            "ar":       round(ap, 4),
            "momentum": round(
