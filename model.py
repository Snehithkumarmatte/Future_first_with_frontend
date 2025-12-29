import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# ======================================================
# Utility: create LSTM sequences
# ======================================================
def create_sequences(X, y, window=3):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


# ======================================================
# MAIN PIPELINE
# ======================================================
def run_pipeline(data_df, price_df):
    # ----------------------------
    # Data preprocessing
    # ----------------------------
    data_df["Date"] = pd.to_datetime(data_df["Date"], errors="coerce")
    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")

    data_df = data_df.dropna().sort_values("Date")
    price_df = price_df.dropna().sort_values("Date")

    df = pd.merge(data_df, price_df, on="Date", how="inner")
    df.rename(columns={"Data": "data_value", "Price": "stock_price"}, inplace=True)

    # ----------------------------
    # Feature engineering (DATA ONLY)
    # ----------------------------
    df["data_change"] = df["data_value"].diff()
    df["data_pct_change"] = df["data_value"].pct_change()
    df["data_change_lag1"] = df["data_change"].shift(1)
    df["data_change_lag2"] = df["data_change"].shift(2)
    df["data_ma3"] = df["data_value"].rolling(3).mean()
    df["data_momentum3"] = df["data_value"] - df["data_value"].shift(3)

    # Target: PRICE MOVEMENT
    df["price_change"] = df["stock_price"].diff()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    features = [
        "data_change",
        "data_change_lag1",
        "data_change_lag2",
        "data_pct_change",
        "data_ma3",
        "data_momentum3",
    ]

    X = df[features]
    y = df["price_change"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    predictions = {}

    # ======================================================
    # MODEL 1: Elastic Net (Recruiter loves this)
    # ======================================================
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    enet.fit(X_train_scaled, y_train)
    enet_pred = enet.predict(X_test_scaled)

    results.append({
        "Model": "Elastic Net",
        "RMSE": np.sqrt(mean_squared_error(y_test, enet_pred)),
        "MAE": mean_absolute_error(y_test, enet_pred),
        "R2": r2_score(y_test, enet_pred),
        "Directional Accuracy (%)":
            np.mean(np.sign(y_test.values) == np.sign(enet_pred)) * 100
    })
    predictions["Elastic Net"] = enet_pred

    # ======================================================
    # MODEL 2: Gradient Boosting
    # ======================================================
    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=2, random_state=42
    )
    gbr.fit(X_train_scaled, y_train)
    gbr_pred = gbr.predict(X_test_scaled)

    results.append({
        "Model": "Gradient Boosting",
        "RMSE": np.sqrt(mean_squared_error(y_test, gbr_pred)),
        "MAE": mean_absolute_error(y_test, gbr_pred),
        "R2": r2_score(y_test, gbr_pred),
        "Directional Accuracy (%)":
            np.mean(np.sign(y_test.values) == np.sign(gbr_pred)) * 100
    })
    predictions["Gradient Boosting"] = gbr_pred

    # ======================================================
    # MODEL 3: LSTM (Deep Learning)
    # ======================================================
    X_lstm, y_lstm = create_sequences(X_train_scaled, y_train.values, window=3)
    X_lstm_test, y_lstm_test = create_sequences(X_test_scaled, y_test.values, window=3)

    lstm = Sequential([
        LSTM(32, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dense(1)
    ])
    lstm.compile(optimizer=Adam(0.01), loss="mse")
    lstm.fit(X_lstm, y_lstm, epochs=40, batch_size=8, verbose=0)

    lstm_pred = lstm.predict(X_lstm_test).flatten()

    results.append({
        "Model": "LSTM",
        "RMSE": np.sqrt(mean_squared_error(y_lstm_test, lstm_pred)),
        "MAE": mean_absolute_error(y_lstm_test, lstm_pred),
        "R2": r2_score(y_lstm_test, lstm_pred),
        "Directional Accuracy (%)":
            np.mean(np.sign(y_lstm_test) == np.sign(lstm_pred)) * 100
    })
    predictions["LSTM"] = lstm_pred

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    return results_df, y_test.values, predictions
