# ================================================================
# 🧠 Analisis Forecast + Berita PT ANTAM (Streamlit & Cortex Arctic)
# VERSI FINAL: KORELASI KUANTITATIF DENGAN KEDALAMAN ANALISIS
# ================================================================

import streamlit as st
import pandas as pd
import altair as alt
from snowflake.snowpark.exceptions import SnowparkSessionException
from snowflake.snowpark.context import get_active_session

# --- Variable Configuration ---
MODEL_TO_USE = 'snowflake-arctic'
SNOWFLAKE_DATABASE = 'ANTAM'
SNOWFLAKE_SCHEMA = 'DATA'
FORECAST_TABLE = 'FORECAST_RESULT_ALL'
NEWS_TABLE = 'DATA_BERITA'

# Batasan baris untuk LLM (hanya untuk data forecast)
MAX_ROWS_FOR_LLM = 50

# Batasan Karakter Keras untuk input prompt
MAX_CHARS_FORECAST = 4000 
MAX_CHARS_NEWS = 4000
MAX_CHARS_SUMMARY = 4000

FULL_FORECAST_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{FORECAST_TABLE}"
FULL_NEWS_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{NEWS_TABLE}"

# --- Snowflake Session Setup and Caching ---

@st.cache_data(show_spinner="Memuat data unik untuk filter...")
def load_unique_values(_session):
    """Mengambil item unik dan nama lokasi untuk filter. Tidak ada fallback simulasi."""
    try:
        unique_query = f"""
            SELECT
                LISTAGG(DISTINCT NAMABARANG, ', ') WITHIN GROUP (ORDER BY NAMABARANG) AS UNIQUE_BARANG,
                LISTAGG(DISTINCT LOCATION, ', ') WITHIN GROUP (ORDER BY LOCATION) AS UNIQUE_LOKASI
            FROM {FULL_FORECAST_TABLE};
        """
        unique_res = _session.sql(unique_query).collect()[0]

        unique_barang_list = unique_res['UNIQUE_BARANG'].split(', ') if unique_res['UNIQUE_BARANG'] else []
        unique_lokasi_list = unique_res['UNIQUE_LOKASI'].split(', ') if unique_res['UNIQUE_LOKASI'] else []

        return unique_barang_list, unique_lokasi_list, unique_res['UNIQUE_BARANG'], unique_res['UNIQUE_LOKASI']
    except Exception as e:
        st.error(f"Error memuat nilai unik dari Snowflake: {e}")
        return [], [], "", ""

@st.cache_data(show_spinner="Memuat semua data forecast (sejak 2022)...")
def load_all_forecast_data(_session):
    """Mengambil semua data forecast. Tidak ada fallback simulasi."""
    try:
        query = f"""
            SELECT INVOICEDATE, LOCATION, NAMABARANG, ACTUAL, TRAIN_PRED, TEST_PRED, FORECAST_PRED
            FROM {FULL_FORECAST_TABLE}
            ORDER BY INVOICEDATE ASC;
        """
        df_snow = _session.sql(query).to_pandas()
        df_snow['INVOICEDATE'] = pd.to_datetime(df_snow['INVOICEDATE'])
        return df_snow
    except Exception as e:
        st.error(f"Error memuat data forecast dari Snowflake: {e}")
        return pd.DataFrame()


# --- Cortex Analysis Logic (Runs on Button Click) ---

def run_cortex_analysis(_session, df_filtered, selected_barang, selected_lokasi):
    """Mengeksekusi langkah-langkah AI Agent hanya dengan data Snowflake."""

    st.info("⏳ Mempersiapkan data dan menjalankan alur kerja tiga Agen. Menggunakan data AKTUAL dari Snowflake.")

    # 1. Batasi df_filtered menjadi 50 baris data terbaru saja untuk LLM
    if not df_filtered.empty:
        df_llm = df_filtered.sort_values(by='INVOICEDATE', ascending=False).head(MAX_ROWS_FOR_LLM)
        df_llm = df_llm.sort_values(by='INVOICEDATE', ascending=True)
    else:
        df_llm = pd.DataFrame()

    # 2. Format Filtered Forecast Data
    forecast_data_list = [
        f"Tgl: {row['INVOICEDATE'].strftime('%Y-%m-%d')} | Lokasi: {row['LOCATION']} | Brg: {row['NAMABARANG']} | Forecast: {row['FORECAST_PRED']}"
        for index, row in df_llm.iterrows()
    ]
    forecast_data = '\n'.join(forecast_data_list)

    # 3. Fetch News Data (MENGGUNAKAN QUERY TERAKHIR YANG PALING KETAT)
    try:
        news_query = f"""
            SELECT
                LISTAGG(
                    'Tanggal: ' || COALESCE(C6, '-') || 
                    ' | Ringkasan: ' || COALESCE(C2, '-') || 
                    ' | Tautan: ' || COALESCE(C5, '-'), 
                    '\n- - - BERITA - - - \n'
                ) WITHIN GROUP (ORDER BY C6 DESC)
            FROM {FULL_NEWS_TABLE}
            WHERE C6 IS NOT NULL 
            ;
        """
        news_data = _session.sql(news_query).collect()[0][0] or ""
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal mengambil data berita dari Snowflake. Error: {e}")
        return # Hentikan eksekusi

    # 4. IMPLEMENTASI SOLUSI KRITIS: BATASI STRING DENGAN SLICING
    forecast_data_sliced = forecast_data[:MAX_CHARS_FORECAST]
    news_data_sliced = news_data[:MAX_CHARS_NEWS]

    if len(forecast_data) > MAX_CHARS_FORECAST:
        st.warning(f"Data forecast dipotong menjadi {MAX_CHARS_FORECAST} char.")
    if len(news_data) > MAX_CHARS_NEWS:
        st.warning(f"Data berita dipotong menjadi {MAX_CHARS_NEWS} char.")

    # ================================================================
    # 📌 VERIFIKASI LOG DATA BERITA UNTUK DEBUGGING AGENT 3
    # ================================================================
    st.markdown("---")
    st.subheader("🛠️ DEBUG LOG: Input Data Berita untuk Agent 3")
    st.code(f"Jumlah karakter total data berita (setelah slicing): {len(news_data_sliced)}")
    if len(news_data_sliced) > 0:
        st.text("Preview data berita yang akan dikirim ke LLM (Agent 3):")
        st.code(news_data_sliced)
    else:
        st.error("Data berita kosong (0 karakter). Agent 3 TIDAK AKAN dapat melakukan korelasi.")
    st.markdown("---")
    # ================================================================


    # --- AI Agent 1: Forecast Saja (Fokus FORECAST_PRED & Kuantitatif) ---
    with st.spinner("⏳ [AGENT 1] Menganalisis data forecast saja..."):
        
        forecast_only_prompt = f"""
            Anda adalah analis senior PT ANTAM (Agent 1). Tugas Anda adalah **MENGANALISIS HANYA KOLOM FORECAST (FORECAST_PRED)** dari data penjualan emas batangan di bawah ini. Anda WAJIB MENGGUNAKAN ANGKA SPESIFIK dari data untuk mendukung analisis.
            
            Fakta Filter: **Nama Barang: {selected_barang}** dan **Lokasi: {selected_lokasi}**.

            Tugas Analisis Forecast (Wajib Detail dan Terstruktur):
            1. **VERIFIKASI FILTER**: Sebutkan ulang Nama Barang dan Lokasi.
            2. **PERBANDINGAN ANGKA KRITIS**:
               a. Identifikasi **Nilai Forecast Tertinggi** dan **Tanggal**-nya.
               b. Identifikasi **Nilai Forecast Terendah** dan **Tanggal**-nya.
            3. **ANALISIS TREN (KUANTITATIF)**:
               a. Jelaskan tren **jangka panjang** secara kuantitatif.
               b. Analisis tren **jangka pendek** (5 baris data terbaru). Sebutkan nilai awal dan nilai akhirnya, dan tentukan apakah terjadi kenaikan/penurunan tajam.
            4. **KESIMPULAN FOKUS**: Berikan 3 poin kunci hasil analisis yang **hanya berisi fakta numerik** dari kolom Forecast.

            === DATA FORECAST YANG DIFILTER (FOKUS: Tgl | Lokasi | Brg | Forecast) ===
            {forecast_data_sliced}
        """
        cortex_query_forecast = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{MODEL_TO_USE}', $${forecast_only_prompt}$$) AS ANALISIS_ARCTIC;"
        analisis_forecast_only = _session.sql(cortex_query_forecast).collect()[0][0]
        st.session_state.analisis_1 = analisis_forecast_only
        analisis_1_output = analisis_forecast_only

    # --- AI Agent S: Summarization (Mengkompres Analisis 1) ---
    with st.spinner("⏳ [AGENT KOMPRESI] Meringkas hasil Agent 1..."):
        summarization_prompt = f"""
            Rangkum secara padat (maksimal 4-5 poin kunci, JANGAN LEBIH dari 1000 karakter) hasil analisis forecast berikut. Fokus **HANYA** pada tren kuantitatif dan perbandingan angka yang spesifik:
            === ANALISIS LENGKAP DARI AGENT 1 ===
            {analisis_1_output}
        """
        cortex_query_summary = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{MODEL_TO_USE}', $${summarization_prompt}$$) AS RINGKASAN_ARCTIC;"
        analisis_1_summary = _session.sql(cortex_query_summary).collect()[0][0]
        analisis_1_summary_sliced = analisis_1_summary[:MAX_CHARS_SUMMARY]
        st.session_state.analisis_1_summary = analisis_1_summary_sliced 

    # --- AI Agent 3: Korelasi Berita dan Kesimpulan Akhir ---
    with st.spinner("⏳ [AGENT 3] Menganalisis korelasi berita dengan forecast dan memberikan kesimpulan akhir..."):
        
        combined_analysis_prompt = f"""
            Lakukan Analisis Gabungan di bawah ini sebagai analis senior PT ANTAM (Agent 3).

            **WAJIB INNTEGRITAS DATA:** Data berita yang disediakan di bagian `=== DATA BERITA ===` adalah fakta tunggal yang boleh Anda gunakan. JANGAN MENGARANG TANGGAL ATAU KONTEN BERITA.

            Tugas Anda adalah:
            1. **VERIFIKASI DATA BERITA**: Tentukan dan Nyatakan tanggal berita **TERTUA** dan **TERBARU** yang *benar-benar* Anda temukan di `=== DATA BERITA ===`.
            2. **ANALISIS BERITA RELEVAN**: Cari 2-3 berita yang paling relevan dengan Tren/Kesimpulan kuantitatif dari RINGKASAN Agent 1.
            3. **PENYAJIAN BERITA KRITIS (TAMPILKAN DATA ASLI)**: Untuk setiap berita yang relevan, **TAMPILKAN DATA ASLI** dari input berita tersebut dengan format:
                * **Berita Relevan 1:** [Tanggal ASLI] | [Ringkasan ASLI] | [Tautan ASLI]
                * **Berita Relevan 2:** [Tanggal ASLI] | [Ringkasan ASLI] | [Tautan ASLI]
                * [Dan seterusnya, salin data yang ada di input]
            4. **KORELASI KRITIS (WAJIB KUANTITATIF DAN MENDALAM)**: 
               a. Jelaskan korelasi sebab-akibat yang kuat. Korelasi **WAJIB MENGGABUNGKAN TEMUAN KUANTITATIF DARI RINGKASAN FORECAST** dengan fakta-fakta berita yang telah disajikan.
               b. **Kembangkan analisis ini menjadi paragraf yang komprehensif (minimal 3-4 kalimat).** Jelaskan **mekanisme pasar** (misalnya, bagaimana pelemahan data makro global memicu pergerakan harga komoditas dan memengaruhi forecast ANTAM).
            5. **KESIMPULAN AKHIR & REKOMENDASI**: Berikan KESIMPULAN BISNIS FINAL dan **3-5 REKOMENDASI STRATEGIS** yang spesifik dan padat.

            === RINGKASAN HASIL ANALISIS FORECAST (DARI AGENT 1/KOMPRESI) ===
            {analisis_1_summary_sliced}

            === DATA BERITA (DIBATASI {MAX_CHARS_NEWS} KARAKTER) ===
            {news_data_sliced}
        """
        cortex_query_combined = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{MODEL_TO_USE}', $${combined_analysis_prompt}$$) AS ANALISIS_ARCTIC;"
        analisis_combined = _session.sql(cortex_query_combined).collect()[0][0]
        st.session_state.analisis_2 = analisis_combined 

    return


# --- Streamlit Main App ---

def main():
    st.set_page_config(layout="wide")
    st.title("💰 Analisis Gabungan Forecast Penjualan Emas & Berita")
    st.markdown("Aplikasi ini menggunakan **3 AI Agent Cortex Arctic** dan **HANYA** mengambil data dari Snowflake. *Log Debug* diaktifkan untuk memverifikasi input Agent 3.")

    # Get Snowpark Session
    try:
        session = get_active_session()
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal mendapatkan Snowpark Session. Pastikan Anda menjalankan aplikasi ini di lingkungan Snowsight/Streamlit-in-Snowflake. Error: {e}")
        st.stop()
        return

    # Initialize session state
    if 'analisis_1' not in st.session_state: st.session_state.analisis_1 = None
    if 'analisis_1_summary' not in st.session_state: st.session_state.analisis_1_summary = None
    if 'analisis_2' not in st.session_state: st.session_state.analisis_2 = None

    # 1. Load Data
    unique_barang_list, unique_lokasi_list, full_unique_barang, full_unique_lokasi = load_unique_values(session)
    df_forecast = load_all_forecast_data(session)

    if df_forecast.empty or not unique_barang_list:
        if not df_forecast.empty and unique_barang_list:
             # Menghindari crash jika data forecast/filter valid, tetapi ada logic error
             pass
        else:
             st.error("Tidak dapat melanjutkan karena data filter atau forecast kosong dari Snowflake.")
             return

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data Forecast")
    lokasi_semua = "Semua Lokasi"
    lokasi_options = [lokasi_semua] + unique_lokasi_list
    selected_lokasi = st.sidebar.selectbox("Pilih Lokasi:", lokasi_options, index=0)
    barang_semua = "Semua Barang"
    barang_options = [barang_semua] + unique_barang_list
    selected_barang = st.sidebar.selectbox("Pilih Nama Barang:", barang_options, index=0)

    # --- Apply Filters ---
    df_filtered = df_forecast.copy()
    if selected_lokasi != lokasi_semua:
        df_filtered = df_filtered[df_filtered['LOCATION'] == selected_lokasi]
    if selected_barang != barang_semua:
        df_filtered = df_filtered[df_filtered['NAMABARANG'] == selected_barang]

    # --- 2. Data Visualization ---
    is_selection_made = (selected_lokasi != lokasi_semua) or (selected_barang != barang_semua)

    if not df_filtered.empty and is_selection_made:
        st.subheader(f"Data Forecast Terpilih (Total {len(df_filtered)} Baris)")
        df_plot = df_filtered.melt(
            id_vars=['INVOICEDATE', 'LOCATION', 'NAMABARANG'],
            value_vars=['ACTUAL', 'TRAIN_PRED', 'TEST_PRED', 'FORECAST_PRED'],
            var_name='Tipe_Nilai', value_name='Nilai_Penjualan'
        )
        chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X('INVOICEDATE', title='Tanggal'),
            y=alt.Y('Nilai_Penjualan', title='Penjualan (Aktual & Prediksi)'),
            color=alt.Color('Tipe_Nilai', title='Tipe Nilai'),
            tooltip=['INVOICEDATE', 'Tipe_Nilai', 'Nilai_Penjualan', 'LOCATION', 'NAMABARANG']
        ).properties(
            title=f'Tren Penjualan dan Forecast: {selected_barang} di {selected_lokasi}'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_filtered, use_container_width=True)
    elif not df_filtered.empty and not is_selection_made:
        st.info("Pilih filter di sidebar untuk visualisasi.")
    else:
        st.warning("Filter yang dipilih tidak menghasilkan data.")

    st.markdown("---")

    # --- 3. Analysis Execution Button ---
    if st.button("Jalankan Analisis Cortex Arctic (3 AI Agents)", type="primary"):
        if df_filtered.empty:
            st.error("Tidak dapat menjalankan analisis: Data yang difilter kosong.")
            return
        run_cortex_analysis(session, df_filtered, selected_barang, selected_lokasi)

    # --- 4. Display Results ---
    st.header("🔬 Hasil Analisis AI Agent Cortex Arctic")

    st.subheader("1. [AGENT 1] ANALISIS FORECAST SAJA")
    if st.session_state.analisis_1: st.markdown(st.session_state.analisis_1)
    else: st.info("Hasil Analisis 1 akan muncul di sini.")

    st.subheader("1A. [AGENT KOMPRESI] RINGKASAN POIN KUNCI")
    if st.session_state.analisis_1_summary: st.markdown(st.session_state.analisis_1_summary)
    else: st.info("Ringkasan poin kunci akan dibuat setelah Analisis 1 selesai.")

    st.subheader("2. [AGENT 3] ANALISIS KORELASI BERITA DAN KESIMPULAN AKHIR")
    if st.session_state.analisis_2: st.markdown(st.session_state.analisis_2)
    else: st.info("Hasil Analisis 3 (Korelasi & Kesimpulan) akan muncul di sini.")


if __name__ == "__main__":
    main()