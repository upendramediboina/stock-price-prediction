import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Stock Predictor", page_icon="📈")

st.title("📈 Stock Price Predictor")
st.caption("Upload historical stock data and predict the next closing price")

# Currency
currency = st.radio(
    "💱 Display Currency",
    ["USD ($)", "INR (₹)"],
    horizontal=True
)

rate = 1 if currency == "USD ($)" else 83
symbol = "$" if currency == "USD ($)" else "₹"

# Upload dataset
file = st.file_uploader("📁 Upload Stock CSV", type=["csv"])

if file:

    data = pd.read_csv(file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(), width="stretch")

    # Select price column
    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        st.error("❌ No numeric columns found.")
        st.stop()

    default_column = next(
        (c for c in numeric_columns if c.lower() == "close"),
        numeric_columns[0]
    )

    price_column = st.selectbox(
        "📌 Select Price Column",
        numeric_columns,
        index=numeric_columns.index(default_column)
    )

    # Prepare data
    data["Previous_Price"] = data[price_column].shift(1)
    data = data.dropna()

    X = data[["Previous_Price"]]
    y = data[price_column]

    # Train-test split
    split = int(len(data) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    test_prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_prediction)
    r2 = r2_score(y_test, test_prediction)

    # Next-day prediction
    latest_price = data[price_column].iloc[-1]

    prediction = model.predict(
        pd.DataFrame({"Previous_Price": [latest_price]})
    )[0]

    # Currency conversion
    latest = latest_price * rate
    predicted = prediction * rate
    error = mae * rate

    change = predicted - latest
    percentage = (change / latest) * 100

    # Prediction
    st.subheader("🔮 Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Price",
        f"{symbol}{latest:,.2f}"
    )

    col2.metric(
        "Predicted Price",
        f"{symbol}{predicted:,.2f}"
    )

    col3.metric(
        "Expected Change",
        f"{percentage:+.2f}%"
    )

    # Model performance
    st.subheader("🤖 Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{symbol}{error:,.2f}"
    )

    col2.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    # Price chart
    st.subheader("📈 Price History")

    if "date" in data.columns:
        chart = data.set_index("date")[price_column] * rate
    else:
        chart = data[price_column] * rate

    st.line_chart(chart)

    # Download result
    result = pd.DataFrame({
        "Latest Price": [latest],
        "Predicted Price": [predicted],
        "Expected Change (%)": [percentage],
        "MAE": [error],
        "R2 Score": [r2]
    })

    st.download_button(
        "⬇️ Download Prediction",
        result.to_csv(index=False),
        "stock_prediction.csv",
        "text/csv"
    )

    st.success("✅ Prediction generated successfully!")

else:
    st.info("👆 Upload a CSV file to start the prediction.")

st.caption("⚠️ Educational project only — not financial advice.")