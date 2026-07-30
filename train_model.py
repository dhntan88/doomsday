import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import yfinance as yf
import ta
import joblib

print("Mengunduh data historis untuk training model Neural Network (MLP)...")
# Mengambil data historis emas (GC=F) selama 60 hari terakhir dengan interval 1 jam
df = yf.download("GC=F", period="60d", interval="60m")

if df.empty:
    print("Gagal mengunduh data latihan dari Yahoo Finance.")
else:
    # Merapikan format kolom jika berbentuk MultiIndex dari yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # Hitung Indikator Teknikal sebagai Fitur Pembelajaran
    df['ema_fast'] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    
    # Fitur turunan matematis untuk analisis tren
    df['ema_diff'] = df['ema_fast'] - df['ema_slow']
    df['price_vs_ema'] = close - df['ema_fast']

    # Buat Target Label (1 jika periode berikutnya harga naik, 0 jika turun)
    df['future_close'] = close.shift(-1)
    df['target'] = (df['future_close'] > close).astype(int)

    # Hapus baris kosong (NaN)
    df = df.dropna()

    # Tentukan Kolom Fitur (Features) yang akan digunakan model
    feature_cols = ['ema_diff', 'rsi', 'atr', 'price_vs_ema']
    X = df[feature_cols]
    y = df['target']

    print(f"Melatih model Neural Network (MLP) dengan {len(X)} baris data...")
    # Menggunakan Neural Network melalui pipeline dengan Standard Scaler
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16), 
            activation='relu', 
            solver='adam', 
            max_iter=500, 
            random_state=42
        )
    )
    model.fit(X, y)

    # Simpan model baru ke file model.pkl
    joblib.dump(model, "model.pkl")
    print("Sukses! Model Neural Network baru berhasil disimpan ke file 'model.pkl'.")
