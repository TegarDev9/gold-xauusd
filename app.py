import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re

# -----------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis & Prediksi Harga Emas (XAU/USD)",
    page_icon="⚜️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Judul dan Deskripsi Aplikasi
# -----------------------------------------------------------------------------
st.title("⚜️ Analisis & Prediksi Harga Emas (XAU/USD)")
st.markdown("Sebuah Dasbor Analitik untuk Memproyeksikan Pergerakan Harga Emas Menggunakan Model *Ensemble Machine Learning* dan Variabel Makroekonomi.")

# -----------------------------------------------------------------------------
# Fungsi-fungsi Inti
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(start_date, end_date):
    """
    Memuat dan menggabungkan data historis untuk Emas dan variabel eksogen.
    """
    symbols = {
        'Gold': 'GC=F',      # Emas Futures
        'DXY': 'DX-Y.NYB',   # Indeks Dolar AS
        'VIX': '^VIX',       # Indeks Volatilitas CBOE
        'TNX': '^TNX'        # Imbal Hasil Obligasi Pemerintah AS 10-Tahun
    }
    
    data_frames = {}
    for key, ticker in symbols.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            data_frames[key] = df[['Close']].rename(columns={'Close': key})
        except Exception as e:
            st.error(f"Gagal memuat data untuk {key} ({ticker}): {e}")
            return pd.DataFrame()

    if not data_frames:
        return pd.DataFrame()

    full_df = pd.concat(data_frames.values(), axis=1)
    # Menggunakan metode interpolasi linear untuk mengisi data yang lebih baik daripada ffill
    full_df = full_df.interpolate(method='linear').dropna()
    
    return full_df

@st.cache_data
def create_features(_data):
    """
    Melakukan rekayasa fitur (feature engineering) untuk meningkatkan performa model.
    """
    df = _data.copy()
    # Fitur Lag
    df['Gold_Lag_1'] = df['Gold'].shift(1)
    
    # Fitur Moving Average
    df['Gold_SMA_20'] = df['Gold'].rolling(window=20).mean()
    df['Gold_SMA_50'] = df['Gold'].rolling(window=50).mean()
    
    # Fitur Momentum (Relative Strength Index - RSI)
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Fitur Volatilitas (Rolling Standard Deviation)
    df['Volatility_20D'] = df['Gold'].pct_change().rolling(window=20).std() * np.sqrt(20)

    return df.dropna()

# -----------------------------------------------------------------------------
# Sidebar untuk Input Pengguna
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameter Model & Periode Data")
    start_date = st.date_input("Tanggal Mulai Analisis", pd.to_datetime('2010-01-01'))
    end_date = st.date_input("Tanggal Selesai Analisis", pd.to_datetime('today'))
    prediction_days = st.slider("Horizon Prediksi (Hari)", 1, 90, 14, 
                                help="Jangka waktu prediksi ke depan. Nilai yang lebih pendek cenderung lebih akurat.")
    
    st.info("""
    **Variabel yang Digunakan:**
    - **Emas (GC=F):** Harga target.
    - **Indeks Dolar (DXY):** Korelasi negatif; Dolar kuat cenderung menekan harga emas.
    - **VIX (Indeks Ketakutan):** Korelasi positif; Ketidakpastian pasar global meningkatkan permintaan emas sebagai *safe haven*.
    - **Suku Bunga 10-Thn (TNX):** Korelasi negatif; Suku bunga tinggi meningkatkan *opportunity cost* memegang emas.
    """)

# -----------------------------------------------------------------------------
# Proses Utama dan Pembuatan Model
# -----------------------------------------------------------------------------
raw_data = load_data(start_date, end_date)

if not raw_data.empty:
    featured_data = create_features(raw_data)
    
    # Persiapan data untuk model
    df_model = featured_data.copy()
    df_model['Gold_Target'] = df_model['Gold'].shift(-prediction_days)
    df_model.dropna(inplace=True)

    features = [col for col in df_model.columns if col not in ['Gold_Target']]
    X = df_model[features]
    y = df_model['Gold_Target']
    
    # PERBAIKAN: Mengonversi setiap nama kolom ke string sebelum dibersihkan
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in X.columns]
    
    # Pembagian data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Pelatihan Model Quantile Regression ---
    models = {}
    quantiles = {'lower': 0.1, 'median': 0.5, 'upper': 0.9}

    with st.spinner("Melatih model prediktif... Ini mungkin memerlukan beberapa saat."):
        for name, q in quantiles.items():
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                metric='quantile',
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1,
                n_jobs=-1,
                seed=42
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric='quantile',
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            models[name] = model

    # --- Generate Prediksi ---
    last_known_features = X.tail(1)
    predictions = {name: model.predict(last_known_features)[0] for name, model in models.items()}
    
    last_close_price = featured_data['Gold'].iloc[-1]
    
    # --- Membuat DataFrame untuk Visualisasi ---
    future_dates = pd.to_datetime(pd.date_range(start=featured_data.index[-1], periods=prediction_days + 1, freq='B'))
    plot_data = featured_data.tail(60).copy()

    # Gabungkan data historis dengan proyeksi
    fig = go.Figure()
    
    # Plot Interval Prediksi (area abu-abu)
    fig.add_trace(go.Scatter(
        x=[plot_data.index[-1], future_dates[-1]],
        y=[last_close_price, predictions['upper']],
        fill=None, mode='lines', line_color='rgba(211,211,211,0.5)', name='Batas Atas'
    ))
    fig.add_trace(go.Scatter(
        x=[plot_data.index[-1], future_dates[-1]],
        y=[last_close_price, predictions['lower']],
        fill='tonexty', mode='lines', line_color='rgba(211,211,211,0.5)', name='Batas Bawah'
    ))

    # Plot data historis
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Gold'], mode='lines', name='Harga Historis', line=dict(color='gold', width=3)))
    
    # Plot garis prediksi median
    fig.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['median']],
                           mode='lines+markers', name='Prediksi Median', line=dict(color='red', dash='dot', width=2),
                           marker=dict(size=8, symbol='x')))

    fig.update_layout(
        title=f'Proyeksi Harga Emas untuk {prediction_days} Hari ke Depan',
        xaxis_title='Tanggal', yaxis_title='Harga Emas (USD)', template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # =========================================================================
    # Tampilan Aplikasi dengan TAB
    # =========================================================================
    
    tab1, tab2, tab3 = st.tabs(["**Ringkasan Prediksi**", "**Analisis Model & Data**", "**Metodologi & Risiko**"])

    with tab1:
        st.header("🎯 Ringkasan Eksekutif Prediksi")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Proyeksi Harga (Median)",
                value=f"${predictions['median']:,.2f}",
                delta=f"${predictions['median'] - last_close_price:,.2f}"
            )
        with col2:
            st.metric(label="Harga Penutupan Terakhir", value=f"${last_close_price:,.2f}")
        with col3:
            st.metric(label="Horizon Waktu", value=f"{prediction_days} Hari")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Interpretasi Analitik")
        st.markdown(f"""
        Berdasarkan analisis model *quantile regression*, proyeksi median untuk harga emas dalam **{prediction_days} hari** ke depan adalah **${predictions['median']:,.2f}**.
        
        Namun, yang lebih penting adalah **interval keyakinan 80%** yang dihasilkan, yang berkisar antara **${predictions['lower']:,.2f} (batas bawah)** dan **${predictions['upper']:,.2f} (batas atas)**.
        
        - **Rentang yang Lebar** pada interval ini mengindikasikan tingginya volatilitas yang diantisipasi atau ketidakpastian model.
        - **Rentang yang Sempit** menandakan keyakinan model yang lebih tinggi terhadap prediksinya.
        
        Perubahan yang diproyeksikan sebesar **${predictions['median'] - last_close_price:,.2f}** dari harga penutupan terakhir menunjukkan sentimen pasar yang diinterpretasikan oleh model, baik itu *bullish* (positif) maupun *bearish* (negatif).
        """)

    with tab2:
        st.header("🔍 Analisis Data & Kinerja Model")
        
        # --- Kinerja Model ---
        st.subheader("Evaluasi Kinerja Model pada Data Tes")
        y_pred = models['median'].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")
        c2.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
        c3.metric("R² Score", f"{r2:.2%}")
        st.info("""
        - **RMSE & MAE**: Mengukur rata-rata kesalahan prediksi dalam Dolar AS. Semakin rendah, semakin baik.
        - **R² Score**: Menunjukkan seberapa besar persentase variasi harga emas yang dapat dijelaskan oleh model. Nilai 100% berarti prediksi sempurna.
        """, icon="ℹ️")
        
        # --- Pentingnya Fitur ---
        st.subheader("Faktor Penggerak Utama (Feature Importance)")
        st.markdown("Grafik ini menunjukkan variabel mana yang memiliki dampak paling signifikan terhadap prediksi model. Ini adalah jantung dari pemahaman 'mengapa' di balik sebuah prediksi.")
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': models['median'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_imp = go.Figure(go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], orientation='h'))
        fig_imp.update_layout(title='Peringkat Kepentingan Fitur', xaxis_title='Tingkat Kepentingan', template='plotly_dark', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("""
        **Interpretasi:**
        - **Gold_Lag_1 & SMA:** Menunjukkan bahwa harga emas di masa lalu (*momentum*) adalah prediktor terkuat untuk harga di masa depan, sebuah konsep inti dalam analisis time series.
        - **DXY, VIX, TNX:** Kepentingan variabel makroekonomi ini mengkonfirmasi bahwa model tidak hanya melihat harga emas itu sendiri, tetapi juga mempertimbangkan kesehatan ekonomi AS, sentimen risiko pasar, dan kebijakan moneter.
        - **RSI & Volatility:** Fitur teknikal ini membantu model memahami kondisi *overbought/oversold* dan tingkat gejolak pasar saat ini.
        """)

    with tab3:
        st.header("📜 Metodologi & Risiko")
        st.subheader("Arsitektur Model")
        st.markdown("""
        Aplikasi ini menggunakan **Light Gradient Boosting Machine (LightGBM)**, sebuah implementasi *gradient boosting* yang sangat efisien dan akurat. Keunggulannya terletak pada kecepatan dan kemampuannya menangani data dalam jumlah besar.

        Pendekatan inti yang digunakan adalah **Quantile Regression**. Berbeda dengan regresi standar yang hanya memprediksi nilai rata-rata (mean), *quantile regression* memungkinkan kita untuk memprediksi persentil tertentu dari distribusi target. Dalam kasus ini, kita memprediksi:
        1.  **Quantile 0.5 (Median):** Sebagai prediksi titik utama (best guess).
        2.  **Quantile 0.1 (Batas Bawah):** Skenario pesimis, di mana ada 10% kemungkinan harga akan berada di bawah nilai ini.
        3.  **Quantile 0.9 (Batas Atas):** Skenario optimis, di mana ada 10% kemungkinan harga akan berada di atas nilai ini.
        
        Gabungan dari batas bawah dan atas membentuk **interval keyakinan 80%**, memberikan gambaran rentang pergerakan harga yang paling mungkin terjadi.
        """)
        
        st.subheader("Limitasi & Asumsi Model")
        st.warning("""
        **PENTING:** Model ini adalah alat bantu analitis, bukan bola kristal. Pengguna harus memahami asumsi dan risikonya:
        
        1.  **Kinerja Masa Lalu Bukan Jaminan:** Akurasi model pada data historis tidak menjamin akurasi yang sama di masa depan.
        2.  **Tidak Memprediksi "Black Swan":** Model ini dilatih pada data historis dan tidak dapat memprediksi peristiwa geopolitik atau krisis finansial mendadak yang belum pernah terjadi sebelumnya.
        3.  **Ketergantungan pada Korelasi:** Model mengasumsikan bahwa hubungan historis antara harga emas dan variabel makroekonomi (seperti DXY dan VIX) akan tetap berlanjut. Perubahan struktural dalam ekonomi global dapat mematahkan asumsi ini.
        4.  **Hanya Data Kuantitatif:** Model ini tidak dapat memproses sentimen berita kualitatif, rumor pasar, atau perubahan kebijakan mendadak yang belum tercermin dalam data harga.
        
        Gunakan informasi dari aplikasi ini sebagai salah satu dari banyak faktor dalam pengambilan keputusan investasi Anda.
        """, icon="⚠️")

else:
    st.error("Gagal memuat data. Mohon periksa kembali rentang tanggal yang Anda pilih atau coba lagi nanti.")
