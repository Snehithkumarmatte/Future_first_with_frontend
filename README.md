
# 📈 Stock Price Prediction Using Machine Learning

A production-style machine learning application that predicts **stock price movement**
using **previous day’s data**. The project combines a **Streamlit frontend** with a
**modular Python backend**, allowing users to upload CSV files, run multiple ML models,
visualize results, and download outputs.

---

## 🚀 Key Features

- 📂 Upload **Data.csv** and **StockPrice.csv**
- 🧠 Backend ML pipeline with:
  - **Elastic Net Regression**
  - **Gradient Boosting Regressor**
  - **LSTM (Deep Learning)**
- 📊 Automatic **model comparison**
- 📈 Interactive charts (Actual vs Predicted)
- ⬇️ Downloadable results
- 🔁 Same backend code reusable as `.py` or `.ipynb`

---

## 🧱 Project Architecture

Frontend (Streamlit UI)
- Upload Data.csv
- Upload StockPrice.csv
- Run ML Pipeline
- View Results & Charts
- Download Results

Backend (Python ML Pipeline)
- Data Processing
- Feature Engineering
- Model Training
- Model Comparison
- Output Metrics & Predictions

---

## 📁 Project Structure

```
stock_price_app/
│
├── app.py           # Streamlit frontend + orchestration
├── model.py         # Reusable ML pipeline (backend logic)
├── requirements.txt
└── README.md
```

---

## 📊 Models Used

| Model | Purpose |
|-----|--------|
| Elastic Net | Regularized linear model |
| Gradient Boosting | Strong tabular ML model |
| LSTM | Deep learning for time-series |

---

## 🎯 Prediction Target

The model predicts **stock price movement**, not absolute price:

```
Price_Change = Price(today) − Price(yesterday)
```

This avoids non-stationarity and ensures realistic financial modeling.

---

## 📂 Input File Format

### Data.csv
```
Date,Data
2025-03-26,2.369
2025-03-25,2.365
```

### StockPrice.csv
```
Date,Price
2025-03-26,5759.5
2025-03-25,5826.5
```

Dates must overlap between both files.

---

## ▶️ How to Run

### 1. Create virtual environment
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the app
```
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 🖥️ How to Use the App

1. Upload **Data.csv**
2. Upload **StockPrice.csv**
3. Click **Run ML Pipeline**
4. View model comparison & plots
5. Download results as CSV

---

## 🔬 Evaluation Metrics

- R² Score
- RMSE
- MAE
- Directional Accuracy

---

## 📓 Convert Backend to Jupyter Notebook

```
pip install jupytext
jupytext --to notebook model.py
```

---

## 🧠 Why This Project Is Strong

- Frontend–backend separation
- Reusable ML pipeline
- No data leakage
- Multiple model comparison
- Deep learning included (LSTM)
- Production-ready structure

---

## 👨‍💻 Author

snehith kumar matte

---

## 📜 License

Educational and demonstration purposes.
