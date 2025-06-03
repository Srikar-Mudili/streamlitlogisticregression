import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Stock Direction Predictor")

st.title("📈 Logistic Regression - Stock Price Direction Predictor")

st.markdown("This app uses logistic regression to predict whether the stock price will go **up (1)** or **down (0)** the next day based on historical trends.")

# Simulate stock data
np.random.seed(0)
n = 200
returns = np.random.normal(0, 1, n).cumsum() + 100
dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

df = pd.DataFrame({'Date': dates, 'Close': returns})
df['Prev_Return'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

X = df[['Prev_Return', 'MA_5', 'MA_10']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.write(f"**Accuracy:** {acc:.2f}")

# Display confusion matrix
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title("Confusion Matrix")
st.pyplot(fig)

# Show raw data
with st.expander("📂 Show Raw Data"):
    st.dataframe(df.tail(10))
