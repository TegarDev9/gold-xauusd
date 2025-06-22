import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Emas (XAU/USD)",
    page_icon="⚜️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Judul dan Deskripsi Aplikasi
# -----------------------------------------------------------------------------
st.title("⚜️ Aplikasi Prediksi Harga Emas (XAU/USD)")
st.markdown("""
Aplikasi ini melakukan forecasting harga emas menggunakan model **LightGBM** dengan mempertimbangkan berbagai faktor makroekonomi sebagai variabel eksogen. 
Data diambil secara *real-time* dari Yahoo Finance.
""")

# -----------------------------------------------------------------------------
# Fungsi untuk Memuat Data
# -----------------------------------------------------------------------------
# Menggunakan cache untuk mempercepat pemuatan ulang data
@st.cache_data
def load_data(start_date, end_date):
    """
    Memuat data historis untuk Emas, Dolar AS, Volatilitas, dan Suku Bunga.
    """
    # Simbol ticker di Yahoo Finance
    symbols = {
        'Gold': 'GC=F',      # Emas
        'DXY': 'DX-Y.NYB',   # Indeks Dolar AS
        'VIX': '^VIX',       # Indeks Volatilitas (Fear Index)
        'TNX': '^TNX'        # Imbal Hasil Obligasi 10 Tahun AS (proxy suku bunga/inflasi)
    }
    
    data_frames = []
    for key, ticker in symbols.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Hanya gunakan kolom 'Close' dan ganti namanya sesuai simbol
            df = df[['Close']].rename(columns={'Close': key})
            data_frames.append(df)
        except Exception as e:
            st.error(f"Gagal memuat data untuk {key} ({ticker}): {e}")
            return pd.DataFrame()

    # Gabungkan semua data frame menjadi satu
    if not data_frames:
        return pd.DataFrame()

    full_df = pd.concat(data_frames, axis=1)
    
    # Isi nilai yang hilang dengan metode forward fill
    full_df = full_df.ffill()
    # Hapus baris yang masih memiliki nilai NaN (biasanya di awal periode)
    full_df = full_df.dropna()
    
    return full_df

# -----------------------------------------------------------------------------
# Sidebar untuk Input Pengguna
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")
start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("Tanggal Selesai", pd.to_datetime('today'))
prediction_days = st.sidebar.slider("Hari Prediksi ke Depan", 1, 30, 7)

# -----------------------------------------------------------------------------
# Proses Utama
# -----------------------------------------------------------------------------
# Memuat data berdasarkan input pengguna
data = load_data(start_date, end_date)

if not data.empty:
    st.header("📈 Data Historis Gabungan")
    st.markdown("Data harga penutupan harian untuk Emas dan variabel-variabel makroekonomi terkait.")
    st.dataframe(data.tail(), use_container_width=True)

    # Menyiapkan data untuk model
    df_model = data.copy()
    # Buat target prediksi: harga emas N hari ke depan
    df_model['Gold_Target'] = df_model['Gold'].shift(-prediction_days)
    
    # Hapus baris terakhir yang tidak memiliki target
    df_model.dropna(inplace=True)

    # Definisikan fitur (X) dan target (y)
    features = ['Gold', 'DXY', 'VIX', 'TNX']
    X = df_model[features]
    y = df_model['Gold_Target']

    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Inisialisasi dan latih model LightGBM
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              eval_metric='rmse', 
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # -------------------------------------------------------------------------
    # Prediksi dan Visualisasi
    # -------------------------------------------------------------------------
    st.header("🔮 Hasil Prediksi")
    
    # Buat prediksi untuk N hari ke depan
    last_known_data = data[features].tail(1)
    predicted_price = model.predict(last_known_data)[0]
    
    last_close_price = data['Gold'].iloc[-1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Prediksi Harga Emas dalam {prediction_days} hari",
            value=f"${predicted_price:,.2f}",
            delta=f"${predicted_price - last_close_price:,.2f}"
        )
    with col2:
        st.metric(
            label="Harga Penutupan Terakhir",
            value=f"${last_close_price:,.2f}"
        )
    
    # Buat DataFrame untuk visualisasi
    future_date = data.index[-1] + pd.Timedelta(days=prediction_days)
    forecast_df = pd.DataFrame({
        'Tanggal': [data.index[-1], future_date],
        'Harga': [last_close_price, predicted_price],
        'Tipe': ['Aktual', 'Prediksi']
    })

    # Visualisasi data historis dan prediksi
    fig = go.Figure()

    # Tambahkan data historis
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Gold'], 
        mode='lines', 
        name='Harga Historis Emas',
        line=dict(color='gold')
    ))

    # Tambahkan garis prediksi
    fig.add_trace(go.Scatter(
        x=forecast_df['Tanggal'], 
        y=forecast_df['Harga'], 
        mode='lines+markers', 
        name='Prediksi',
        line=dict(color='red', dash='dot', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f'Sejarah Harga Emas dan Prediksi untuk {prediction_days} Hari ke Depan',
        xaxis_title='Tanggal',
        yaxis_title='Harga Emas (USD)',
        template='plotly_dark',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Analisis Pengaruh Fitur
    # -------------------------------------------------------------------------
    st.header("📊 Analisis Faktor Penggerak Harga (Feature Importance)")
    st.markdown("""
    Grafik ini menunjukkan seberapa besar pengaruh setiap variabel terhadap model prediksi. 
    Semakin tinggi nilainya, semakin penting variabel tersebut dalam menentukan prediksi harga emas.
    """)
    
    feature_importance = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_imp = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    
    fig_imp.update_layout(
        title='Peringkat Pentingnya Fitur dalam Model Prediksi',
        xaxis_title='Tingkat Kepentingan',
        yaxis_title='Fitur/Variabel',
        template='plotly_dark',
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("Gagal memuat data. Silakan periksa kembali pengaturan tanggal atau koneksi internet Anda.")

