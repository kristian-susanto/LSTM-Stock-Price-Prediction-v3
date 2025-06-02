import os
import io
import time
import math
import random
import platform
import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import yfinance as yf
import sklearn
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title="Prediksi Harga Saham",
    page_icon="assets/favicon.ico",
    layout="wide"
)

st.header("Analisis Prediksi Harga Saham Menggunakan Metode LSTM", divider="gray")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
    
if st.session_state.logged_in:
    st.sidebar.header("Autentikasi")
    st.sidebar.success(f"Nama akun: {st.session_state.username}")
    is_admin = True
    if st.sidebar.button("Keluar"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Keluar berhasil!")

        st.markdown("""
            <meta http-equiv="refresh" content="3; url=/">
        """, unsafe_allow_html=True)

        st.stop()
else:
    is_admin = False

st.sidebar.header("Pengaturan Model")
ticker = st.sidebar.text_input(
    "Masukkan Ticker Saham",
    "HLAG.DE",
    help="Ticker adalah kode unik yang digunakan untuk mewakili nama perusahaan pada saham."
)
start_date = st.sidebar.date_input("Tanggal Awal", datetime.date(1980, 1, 1))
end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())
freq = st.sidebar.selectbox(
    "Frekuensi Data",
    options=["Harian", "Mingguan", "Bulanan"],
    help="Frekuensi data merupakan frekuensi pengambilan data harga saham."
)
model_option = st.sidebar.radio(
    "Metode Pemodelan",
    ["Latih model baru", "Gunakan model dari database"],
    help="Pemilihan metode `Latih model baru` disarankan untuk prediksi terkini."
)

predict_button = st.sidebar.button("Mulai Prediksi")

def save_info_model(model, freq, model_type="baseline", history=None, metadata=None):
    os.makedirs("datas/models", exist_ok=True)
    os.makedirs("datas/histories", exist_ok=True)
    os.makedirs("datas/metadatas", exist_ok=True)

    model_name = f"{freq.lower()}_{model_type}.keras"
    model_path = os.path.join("datas/models", model_name)
    model.save(model_path)

    if history is not None and hasattr(history, "history"):
        history_name = f"{freq.lower()}_{model_type}_history.json"
        history_path = os.path.join("datas/histories", history_name)
        with open(history_path, "w") as f:
            json.dump(history.history, f, indent=4)

    if metadata is not None:
        metadata_name = f"{freq.lower()}_{model_type}_params.json"
        metadata_path = os.path.join("datas/metadatas", metadata_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

def delete_old_model(freq, model_type="baseline"):
    model_file = f"datas/models/{freq}_{model_type}.keras"
    history_file = f"datas/histories/{freq}_{model_type}_history.json"
    metadata_file = f"datas/metadatas/{freq}_{model_type}_params.json"

    files_deleted = False
    for file in [model_file, history_file, metadata_file]:
        if os.path.exists(file):
            os.remove(file)
            files_deleted = True
    return files_deleted

def find_model_file(freq, model_type="baseline"):
    model_name = f"{freq.lower()}_{model_type}.keras"
    model_path = os.path.join("datas/models", model_name)
    if os.path.exists(model_path):
        return model_path
    return None

def load_info_model(model_path=None, freq=None, model_type=None, info_type="model"):
    if info_type == "model":
        if model_path and os.path.exists(model_path):
            return load_model(model_path)
        else:
            return None

    elif info_type == "history":
        if model_path:
            history_path = model_path.replace("datas/models", "datas/histories").replace(".keras", "_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    return json.load(f)
        return None

    elif info_type == "metadata":
        if freq and model_type:
            metadata_path = f"datas/metadatas/{freq.lower()}_{model_type}_params.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    return json.load(f)
        return None

    else:
        raise ValueError("info_type harus 'model', 'history', atau 'metadata'.")

def get_frequency_code(freq):
    return {"Harian": "1d", "Mingguan": "1wk", "Bulanan": "1mo"}[freq]

def get_time_step(freq):
    return {"Harian": 30, "Mingguan": 4, "Bulanan": 12}[freq]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs=50, batch_size=32):
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    start_time = time.time()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    duration = time.time() - start_time

    return model, history, duration, epochs, batch_size

def dataset_information_summary(df, dataset_name="Dataset", expanded=True):
    with st.expander(dataset_name, expanded=expanded):
        st.dataframe(df)
        st.write("Ringkasan informasi")
        non_null_counts = df.notnull().sum()
        info_table = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [f"{count} non-null" for count in non_null_counts],
            "Dtype": df.dtypes.astype(str).values
        })
        st.dataframe(info_table)

        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False)
        full_info_str = buffer.getvalue()
        st.code(full_info_str, language="text")

def generate_lstm_model_config(
    model, 
    time_step, 
    epochs, 
    batch_size, 
    duration=None, 
    title="Model Configuration"
):
    lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTM)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]
    output_layer = model.layers[-1]

    config = {
        "Nama Metode": "LSTM",
        "Nama Model LSTM": type(model).__name__,
        "Time Step": time_step,
        "Jumlah Neuron per LSTM Layer": ", ".join(str(layer.units) for layer in lstm_layers),
        "Jumlah LSTM Layer": len(lstm_layers),
        "Learning Rate": round(float(model.optimizer.learning_rate.numpy()), 5),
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Dropout Rate": ", ".join(str(layer.rate) for layer in dropout_layers) if dropout_layers else "0",
        "Optimizer": type(model.optimizer).__name__,
        "Loss": model.loss if isinstance(model.loss, str) else model.loss.__name__,
        "Output Layer": type(output_layer).__name__ + f"({output_layer.units})" if hasattr(output_layer, "units") else str(output_layer)
    }

    if duration is not None:
        config["Training Duration (s)"] = round(duration, 2)

    df = pd.DataFrame(list(config.items()), columns=["Parameter", "Nilai"])
    st.markdown(f"##### {title}")
    st.dataframe(df)

def model_architecture_summary(model):
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + "\n"))
    model_summary_str = summary_buffer.getvalue()
    st.code(model_summary_str, language="text")

    summary_lines = model_summary_str.strip().split("\n")
    table_start = next(i for i, line in enumerate(summary_lines) if "Layer (type)" in line)
    table_end = next(i for i, line in enumerate(summary_lines) if "Total params" in line)
    table_data = summary_lines[table_start + 2 : table_end - 1]

    parsed_rows = []
    for row in table_data:
        clean_row = row.replace("│", "").replace("─", "").replace("└", "").replace("┌", "").replace("┐", "").replace("┘", "")
        parts = [part.strip() for part in clean_row.strip().split("  ") if part.strip()]
        
        if len(parts) >= 3:
            layer_type = parts[0]
            output_shape = parts[1]
            param_count = parts[2]
            parsed_rows.append((layer_type, output_shape, param_count))
    
    df_model_summary = pd.DataFrame(parsed_rows, columns=["Layer (type)", "Output Shape", "Param #"])
    st.dataframe(df_model_summary, use_container_width=True)

def show_prediction_results(
    y_true_rescaled,
    y_pred_rescaled,
    plot_dates,
    title="Hasil Prediksi",
    show_table=True
):
    df_result = pd.DataFrame({
        "Sebenarnya": y_true_rescaled.flatten(),
        "Prediksi": y_pred_rescaled.flatten(),
        "Galat": y_true_rescaled.flatten() - y_pred_rescaled.flatten(),
    })
    df_result["Galat (%)"] = np.where(
        df_result["Sebenarnya"] == 0, 0,
        df_result["Galat"] / df_result["Sebenarnya"] * 100
    )
    df_result["Tanggal"] = plot_dates.reset_index(drop=True)

    cols = ["Tanggal"] + [col for col in df_result.columns if col != "Tanggal"]
    df_result = df_result[cols]

    if show_table:
        st.write(df_result.head())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_result["Tanggal"], df_result["Sebenarnya"], label="Sebenarnya")
    ax.plot(df_result["Tanggal"], df_result["Prediksi"], label="Prediksi")
    ax.set_title(title)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(df_result["Galat"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribusi Galat")
    ax.set_xlabel("Galat")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    rmse = math.sqrt(mean_squared_error(df_result["Sebenarnya"], df_result["Prediksi"]))
    mape = mean_absolute_percentage_error(df_result["Sebenarnya"], df_result["Prediksi"])

    return df_result, rmse, mape

if predict_button:
    st.subheader("0. System and Library Information")
    st.markdown("#### 0.1 Sistem")
    device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    gpus = tf.config.list_physical_devices('GPU')
    cpu_info = platform.processor()
    ram_info = psutil.virtual_memory()

    components = [
        "Sistem Operasi",
        "Versi Python",
        "Perangkat Komputasi",
        "Total RAM (GB)"
    ]
    info_values = [
        f"{platform.system()} {platform.release()}",
        platform.python_version(),
        device_name,
        round(ram_info.total / (1024 ** 3), 2)
    ]

    if cpu_info and cpu_info.strip():
        components.insert(3, "Informasi CPU")
        info_values.insert(3, cpu_info)

    if gpus:
        components.insert(3, "Jumlah GPU")
        info_values.insert(3, str(len(gpus)))
        components.insert(4, "Nama GPU")
        info_values.insert(4, ", ".join([gpu.name for gpu in gpus]))

    system_info = {
        "Komponen": components,
        "Informasi": info_values
    }

    st.dataframe(pd.DataFrame(system_info))

    st.markdown("#### 0.2 Library")
    env_info = {
        "Streamlit": st.__version__,
        "TensorFlow": tf.__version__,
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Sklearn": sklearn.__version__,
        "Matplotlib": matplotlib.__version__,
        "Seaborn": sns.__version__
    }

    env_df = pd.DataFrame(list(env_info.items()), columns=["Library", "Versi"])
    st.dataframe(env_df)

    st.subheader("1. Business Understanding")
    st.markdown("""
        Dalam dunia investasi dan pasar modal, kemampuan untuk memprediksi harga saham secara akurat 
        sangat penting bagi pengambilan keputusan yang tepat dan strategis. Oleh karena itu, dibutuhkan 
        metode yang mampu menangkap pola waktu (time series) secara efektif. Long Short-Term Memory (LSTM) 
        dirancang untuk mengenali pola dalam data berurutan dan memiliki keunggulan dalam mengatasi masalah 
        long-term dependencies. Dengan menerapkan LSTM, perusahaan atau investor dapat memperoleh prediksi 
        harga saham yang mendukung perencanaan dan manajemen risiko investasi.
    """)

    st.subheader("2. Data Understanding")
    with st.spinner("Mengambil data dari Yahoo Finance..."):
        if start_date >= end_date:
            st.sidebar.error("Tanggal awal harus lebih kecil dari tanggal akhir.")
            st.stop()

        freq_code = get_frequency_code(freq)
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), interval=freq_code, auto_adjust=True)

        if data.empty:
            st.error(f"Data dengan ticker {ticker} tidak ditemukan atau gagal diunduh.")
        else:
            actual_start_date = data.index.min().to_pydatetime().date()
            actual_end_date = data.index.max().to_pydatetime().date()

            st.write(
                f"Data dengan ticker **{ticker}** dari tanggal "
                f"**{actual_start_date.strftime('%d %b, %Y')}** sampai "
                f"**{actual_end_date.strftime('%d %b, %Y')}** berhasil diunduh dari Yahoo Finance."
                f"Dataset dapat dilihat pada tautan [ini](https://finance.yahoo.com/quote/{ticker}/history)."
            )
    dataset_information_summary(data, "Data mentah", expanded=True)

    st.subheader("3. Data Preparation")
    st.markdown("#### 3.1 Data Transformation")
    with st.expander("Mengganti multiIndex dengan indeks biasa."):
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            data.columns.name = None
        st.write(data.head())
    
    with st.expander("Mengubah indeks Date menjadi feature dan mengonversi tipe datanya ke datetime."):
        data.reset_index(inplace=True)
        data["Date"] = pd.to_datetime(data["Date"])
        st.write(data.head())
    
    st.markdown("#### 3.2 Data Filtering")
    st.write("Mengambil feature Date dan Close.")
    df = data[["Date", "Close"]].copy()
    df["Close"] = df["Close"].round(2)
    dataset_information_summary(df, "Data yang difilter", expanded=True)

    st.subheader("3.3 Data Cleaning")
    st.markdown("##### 3.3.1 Missing Values")
    with st.expander("Memeriksa dan menangani nilai yang hilang pada feature Close."):
        missing_before = df[["Close"]].isnull().sum()

        if df["Close"].isnull().sum() > 0:
            df["Close"] = df["Close"].fillna(method="ffill")

        missing_after = df[["Close"]].isnull().sum()
        missing_data = {
            "Feature": ["Close"],
            "Missing Values": missing_before.values,
            "Keterangan": [
                "Tidak ada missing value" if missing_before["Close"] == 0 else "Terdapat missing value"
            ]
        }

        if missing_before["Close"] > 0:
            missing_data["Missing sesudah Ditangani"] = missing_after.values
        missing_table = pd.DataFrame(missing_data)
        st.write(missing_table)
    
    st.markdown("##### 3.3.2 Negative Values")
    with st.expander("Memeriksa nilai yang negatif pada feature Close."):
        negative_table = pd.DataFrame({
            "Feature": ["Close"],
            "Negative Values": [(df["Close"] < 0).sum()],
            "Keterangan": [
                "Tidak ada nilai negatif" if (df["Close"] < 0).sum() == 0 else "Terdapat nilai negatif"
            ]
        })
        st.write(negative_table)

    st.markdown("#### 3.4 Exploratory Data Analysis (EDA)")
    st.write("Ringkasan statistik untuk feature Close.")
    st.write(df["Close"].describe())
    df["Year"] = df["Date"].dt.year
    year_counts = df["Year"].value_counts().sort_index()

    st.write("Distribusi jumlah data per tahun.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_counts.index.astype(str), year_counts.values, color="skyblue")
    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Jumlah Data", fontsize=12)
    ax.set_title("Distribusi Jumlah Data per Tahun", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

    st.write("Grafik harga saham.")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], color="blue")
    ax.set_title("Grafik Harga Saham", fontsize=14)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("#### 3.5 Normalization")
    scaler = MinMaxScaler()
    df["Close Scaled"] = scaler.fit_transform(df[["Close"]])
    df_normalized = pd.DataFrame(df[["Date", "Close Scaled"]])
    df_normalized = df_normalized.rename(columns={"Date": "Tanggal", "Close Scaled": "Close yang Diskalasikan"})
    dataset_information_summary(df_normalized, "Data yang telah dinormalisasi tetapi belum disusun sebagai suatu sequence (urutan)", expanded=False)

    st.markdown("#### 3.6 Data Windowing")
    time_step = get_time_step(freq)
    data_scaled = df["Close Scaled"].values.reshape(-1, 1)

    X_seq, y_seq = create_dataset(data_scaled, time_step)

    df_seq = pd.DataFrame({
        "X (sequence)": [x.flatten().round(4).tolist() for x in X_seq],
        "y (target)": y_seq.flatten().round(4)
    })
    dataset_information_summary(df_seq, "Data yang telah dinormalisasi dan disusun sebagai sequence untuk input (X) dan target (y)", expanded=False)

    st.markdown("#### 3.7 Data Splitting")
    time_step = get_time_step(freq)
    X, y = create_dataset(data_scaled, time_step)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    full_data = np.concatenate([y_train, y_test])
    full_data_rescaled = scaler.inverse_transform(full_data.reshape(-1, 1))

    plot_dates = df["Date"].iloc[time_step:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_dates, full_data_rescaled, label="Harga", color="blue")

    split_date = plot_dates.iloc[split]
    ax.axvline(split_date, color="red", linestyle="--", label="Train/Test Split")

    ax.fill_between(plot_dates[:split], full_data_rescaled[:split, 0], color="lightblue", alpha=0.5, label="Train")
    ax.fill_between(plot_dates[split:], full_data_rescaled[split:, 0], color="orange", alpha=0.5, label="Test")

    ax.set_title("Pembagian Data Train dan Test")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    df_X_and_y_train = pd.DataFrame({
        "X_train (sequence)": X_train[i].flatten().round(4).tolist(),
        "y_train": round(float(y_train[i]), 4)
    } for i in range(len(X_train)))
    dataset_information_summary(df_X_and_y_train, "Kelompok data X_train dan y_train", expanded=False)

    df_X_and_y_test = pd.DataFrame({
        "X_test (sequence)": X_test[i].flatten().round(4).tolist(),
        "y_test": round(float(y_test[i]), 4)
    } for i in range(len(X_test)))
    dataset_information_summary(df_X_and_y_test, "Kelompok data X_test dan y_test", expanded=False)

    st.subheader("4. Modeling")
    if model_option == "Gunakan model dari database":
        model_path = find_model_file(freq, model_type="baseline")
        if model_path:
            st.success("Model baseline telah ditemukan.")
            params = load_info_model(freq=freq, model_type="baseline", info_type="metadata")
            if params is None:
                st.error("Metadata model baseline tidak ditemukan.")
                st.stop()
            else:
                time_step = params["time_step"]
                epochs = params["epochs"]
                batch_size = params["batch_size"]
                rmse = params["rmse"]
                mape = params["mape"]
                duration = params["duration"]

            X, y = create_dataset(data_scaled, time_step)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = load_info_model(model_path=model_path, info_type="model")
            start_time = time.time()
            y_pred = model.predict(X_test)
            duration = time.time() - start_time
            history = None

            time_step = time_step
            epochs = epochs
            batch_size = batch_size
        else:
            st.error("Model baseline tidak ditemukan di folder `models/`.")
            st.stop()
    elif model_option == "Latih model baru":
        with st.spinner("Melatih model dan melakukan prediksi..."):
            model, history, duration, epochs, batch_size = build_and_train_model(X_train, y_train, X_test, y_test, time_step)

    st.markdown("#### 4.1 Model Selection")
    generate_lstm_model_config(
        model=model,
        time_step=time_step,
        epochs=epochs,
        batch_size=batch_size,
        title="4.1.1 Model Configuration"
    )

    st.markdown("##### 4.1.2 Model Architecture Summary")
    model_architecture_summary(model)

    st.markdown("#### 4.2 Model Training")
    fig, ax = plt.subplots(figsize=(8, 4))

    if history is None:
        model = load_info_model(model_path=model_path, info_type="model")
        history_dict = load_info_model(model_path=model_path, info_type="history")
        history = type("History", (), {"history": history_dict}) if history_dict else None

    if history is not None:
        ax.plot(history.history["loss"], label="Training Loss", color="blue")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    else:
        st.info("History tidak ditemukan atau gagal dimuat dari file.")

    ax.set_title("Visualisasi Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.markdown("#### 4.3 Prediction Results")
    plot_dates_test = df["Date"].iloc[time_step + len(y_train):]

    df_result, rmse, mape = show_prediction_results(
        y_true_rescaled=y_test_rescaled,
        y_pred_rescaled=y_pred_rescaled,
        plot_dates=plot_dates_test,
        title="Hasil Prediksi"
    )

    baseline_tipe = "Baseline (dari Database)" if model_option == "Gunakan model dari database" else "Baseline"

    baseline_results = [{
        "time_step": time_step,
        "epochs": epochs,
        "batch_size": batch_size,
        "rmse": rmse,
        "mape": mape,
        "model": model,
        "history": history,
        "duration": duration,
        "tipe": baseline_tipe
    }]

    if model_option == "Latih model baru":
        if st.session_state.username == "admin":
            metadata = {
                "time_step": time_step,
                "epochs": epochs,
                "batch_size": batch_size,
                "rmse": rmse,
                "mape": mape,
                "duration": duration,
            }

            files_deleted = delete_old_model(freq, model_type="baseline")

            save_info_model(model, freq, model_type="baseline", history=history, metadata=metadata)

            if files_deleted:
                st.success(f"Model baseline lama berhasil dihapus. Model, histori training, dan metadata parameter baseline yang baru telah berhasil disimpan.")
            else:
                st.success(f"Model, histori training, dan metadata parameter baseline telah berhasil disimpan.")
                
    st.markdown("#### 4.4 Model Results")
    st.write("Hasil model baseline.")

    baseline_tipe = "Baseline (dari Database)" if model_option == "Gunakan model dari database" else "Baseline"

    baseline_table = pd.DataFrame([{
        "Tipe Model": baseline_tipe,
        "Time Step": time_step,
        "Epoch": epochs,
        "Batch Size": batch_size,
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape * 100, 2),
        "Durasi (detik)": round(duration, 2)
    }])
    baseline_table = (
        baseline_table
        .style
        .format({
            "RMSE": "{:.4f}",
            "MAPE (%)": "{:.2f}",
            "Durasi (detik)": "{:.2f}"
        })
    )
    st.dataframe(baseline_table)
else:
    st.info("Silakan isi parameter di sidebar, lalu tekan 'Mulai Prediksi'")