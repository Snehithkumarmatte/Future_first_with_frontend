import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import run_pipeline

st.set_page_config(page_title="Stock Price ML App", layout="wide")

st.title("📈 Stock Price Prediction – Advanced ML Pipeline")
st.write("Upload your datasets and run Elastic Net, Gradient Boosting, and LSTM.")

data_file = st.file_uploader("Upload Data.csv", type="csv")
price_file = st.file_uploader("Upload StockPrice.csv", type="csv")

if data_file and price_file:
    data_df = pd.read_csv(data_file)
    price_df = pd.read_csv(price_file)

    st.success("Files uploaded successfully.")

    if st.button("🚀 Run ML Pipeline"):
        with st.spinner("Training models..."):
            results_df, y_test, predictions = run_pipeline(data_df, price_df)

        st.subheader("📊 Model Comparison")
        st.dataframe(results_df, use_container_width=True)

        best_model = results_df.iloc[0]["Model"]
        st.subheader(f"🏆 Best Model: {best_model}")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test[-len(predictions[best_model]):], label="Actual")
        ax.plot(predictions[best_model], label="Predicted")
        ax.axhline(0, color="black", linestyle="--")
        ax.legend()
        ax.set_title("Actual vs Predicted Price Movement")
        st.pyplot(fig)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "model_results.csv",
            "text/csv"
        )
else:
    st.info("Please upload both CSV files.")
