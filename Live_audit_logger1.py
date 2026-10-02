"""
V75 (1s) 5M Live Auditor - PRODUCTION FINAL
Fixes: WS endpoint, symbol, collision, drift
"""
import json, os, ssl, time
import pandas as pd
import websocket
import xgboost as xgb

# --- CONFIG - FINAL ---
SYMBOL = "1HZ75V" # Must match ws.derivws.com
GRANULARITY = 300
MODEL_FILE = "v75_1s_5m_model.json"
LOG_FILE = "v75_live_audit.csv"
WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
LOOKAHEAD_BARS = 3

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)
print(f"[INIT] Loaded {MODEL_FILE}")

if not os.path.exists(LOG_FILE):
    cols = ["timestamp","entry_price","atr_14","tp_target","sl_target","prob_long","signal","status","outcome","bars_elapsed"]
    pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

def compute_features(df):
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_z"] = (df["close"] - df["bb_mid"]) / (df["bb_std"] + 1e-8)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["mom_5"] = df["close"].pct_change(5) * 1000
    df["vol_range"] = df["high"] - df["low"]
    df["high_20_past"] = df["high"].shift(1).rolling(20).max()
    df["low_20_past"] = df["low"].shift(1).rolling(20).min()
    df["dist_20"] = df["close"] - df["ema_20"]
    df["dist_50"] = df["close"] - df["ema_50"]
    df["dist_low"] = df["close"] - df["low_20_past"]
    df["dist_high"] = df["high_20_past"] - df["close"]
    return df

def update_pending_audits(candle):
    if not os.path.exists(LOG_FILE): return
    audit_df = pd.read_csv(LOG_FILE)
    pending = audit_df["status"] == "PENDING"
    if not pending.any(): return
    high, low, close = candle["high"], candle["low"], candle["close"]
    for idx, row in audit_df[pending].iterrows():
        tp, sl, signal = row["tp_target"], row["sl_target"], row["signal"]
        bars = int(row["bars_elapsed"]) + 1
        hit_tp = high >= tp if signal == "BUY" else low <= tp
        hit_sl = low <= sl if signal == "BUY" else high >= sl
        if hit_tp and hit_sl:
            audit_df.at[idx, "status"] = "CLOSED"; audit_df.at[idx, "outcome"] = "LOSS"
            print(f"[LOSS-COLLISION] #{idx}")
        elif hit_tp:
            audit_df.at[idx, "status"] = "CLOSED"; audit_df.at[idx, "outcome"] = "WIN"
            print(f"[WIN] #{idx} TP {tp:.2f}")
        elif hit_sl:
            audit_df.at[idx, "status"] = "CLOSED"; audit_df.at[idx, "outcome"] = "LOSS"
            print(f"[LOSS] #{idx} SL {sl:.2f}")
        elif bars >= LOOKAHEAD_BARS:
            audit_df.at[idx, "status"] = "CLOSED"
            entry = float(row["entry_price"])
            audit_df.at[idx, "outcome"] = "WIN" if (close > entry if signal=="BUY" else close < entry) else "LOSS"
            print(f"[TIME EXIT] #{idx}")
        audit_df.at[idx, "bars_elapsed"] = bars
    audit_df.to_csv(LOG_FILE, index=False)

def fetch_and_predict():
    ws = websocket.WebSocket()
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    try:
        ws.connect(WS_URL, sslopt={"context": ctx})
        ws.send(json.dumps({"ticks_history": SYMBOL, "adjust_start_time":1, "count":100, "end":"latest", "granularity":GRANULARITY, "style":"candles"}))
        res = json.loads(ws.recv()); ws.close()
        if "candles" not in res: print(f"[WARN] {res}"); return
        df = pd.DataFrame(res["candles"]); df["time"] = pd.to_datetime(df["epoch"], unit="s")
        update_pending_audits(df.iloc[-2])
        df_feat = compute_features(df)
        last = df_feat.iloc[-2]
        X = pd.DataFrame([last[["dist_20","dist_50","atr_14","bb_z","rsi","mom_5","vol_range","dist_low","dist_high","bb_std"]]])
        prob = float(model.predict_proba(X)[0][1]); entry, atr = float(last["close"]), float(last["atr_14"])
        signal = "BUY" if prob>0.60 else "SELL" if prob<0.40 else "NO_TRADE"
        tp = entry + 1.5*atr if signal=="BUY" else entry - 1.5*atr if signal=="SELL" else 0
        sl = entry - 1.0*atr if signal=="BUY" else entry + 1.0*atr if signal=="SELL" else 0
        ts = last["time"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {entry:.2f} | Prob {prob*100:.1f}% -> {signal}")
        if signal in ["BUY","SELL"]:
            row = {"timestamp":ts,"entry_price":entry,"atr_14":round(atr,2),"tp_target":round(tp,2),"sl_target":round(sl,2),"prob_long":round(prob,4),"signal":signal,"status":"PENDING","outcome":"PENDING","bars_elapsed":0}
            audit = pd.read_csv(LOG_FILE)
            pd.concat([audit, pd.DataFrame([row])]).to_csv(LOG_FILE, index=False)
            print(f"[LOGGED] {signal} TP:{tp:.2f} SL:{sl:.2f}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print("="*60 + "\n V75 5M AUDITOR FINAL - LIVE\n" + "="*60)
    while True:
        fetch_and_predict()
        now = time.time()
        next_boundary = ((now // GRANULARITY) + 1) * GRANULARITY + 5
        sleep = max(0, next_boundary - now)
        print(f"[WAIT] {int(sleep)}s\n")
        time.sleep(sleep)