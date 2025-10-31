# Saham
prediksi_saham_app/
│
├─ app.py              # Kode utama Streamlit
├─ requirements.txt    # Library yang dibutuhkan
└─ README.md           # Penjelasan singkat aplikasi & referensi
streamlit
yfinance
pandas
numpy
matplotlib
scikit-learn
tensorflow
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("Prediksi Saham + Sinyal Beli/Jual dengan Referensi 📊")

symbol = st.text_input("Masukkan simbol saham (misal: AAPL)").upper()
if symbol:
    try:
        df = yf.download(symbol, period="2y")[['Close']]
        st.subheader("Harga Penutupan Historis")
        st.line_chart(df)

        # Hitung indikator teknikal
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(window=14).mean()
        loss = (-delta.where(delta<0,0)).rolling(window=14).mean()
        RS = gain / loss
        df['RSI'] = 100 - (100 / (1 + RS))

        # LSTM preprocessing
        data = df['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        time_step = 60
        def create_dataset(dataset, time_step):
            X, y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i-time_step:i,0])
                y.append(dataset[i,0])
            return np.array(X), np.array(y)
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1],1)

        # Build LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        st.text("Melatih model LSTM, tunggu beberapa saat...")
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        st.success("Model selesai dilatih!")

        # Prediksi 7 hari
        temp_input = list(scaled_data[-time_step:])
        lst_output = []
        for _ in range(7):
            x_input = np.array(temp_input[-time_step:]).reshape(1, time_step,1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
        lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))

        pred_dates = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=7)
        pred_df = pd.DataFrame(lst_output, index=pred_dates, columns=['Predicted Close'])
        st.subheader("Prediksi Harga 7 Hari ke Depan")
        st.line_chart(pred_df)
        st.table(pred_df)

        # Sinyal beli/jual
        latest = df.iloc[-1]
        signal = "HOLD"
        if (latest['MA20'] > latest['MA50']) and (latest['RSI'] < 30):
            signal = "BUY"
        elif (latest['MA20'] < latest['MA50']) and (latest['RSI'] > 70):
            signal = "SELL"
        st.subheader("Sinyal Saat Ini")
        st.write(f"Indikasi berdasarkan MA20/MA50 & RSI: **{signal}**")

        # Link referensi
        st.subheader("Referensi Indikator & Prediksi")
        st.markdown("- MACD: [Investopedia](https://www.investopedia.com/terms/m/macd.asp)")
        st.markdown("- RSI: [Penjelasan RSI PDF](https://jhss.scholasticahq.com/article/88915.pdf)")
        st.markdown("- Strategi Moving Average: [Medium](https://medium.com/@tim.po.developer/building-your-first-algorithmic-trading-strategy-a-python-guide-from-data-to-execution-12af0ba2737a)")
        st.markdown("- Sinyal Beli/Jual Bot: [Freshworks](https://community.freshworks.com/ai-bots-11376/how-do-signal-trading-bots-generate-and-interpret-buy-sell-signals-42710)")

    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
