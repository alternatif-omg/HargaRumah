import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = 'housing.csv'  # Ganti dengan path ke file Anda
df = pd.read_csv(file_path)

# Pra-pemrosesan
df.dropna(inplace=True)  # Menghapus baris dengan nilai yang hilang

# Encoding kolom kategorikal
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Memisahkan fitur dan target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Model training
if 'model' not in st.session_state:
    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model in session state
    st.session_state.model = model

    # Prediksi untuk evaluasi
    st.session_state.y_test = y_test
    st.session_state.y_pred = model.predict(X_test)
    st.session_state.mae = mean_absolute_error(y_test, st.session_state.y_pred)
    st.session_state.rmse = np.sqrt(mean_squared_error(y_test, st.session_state.y_pred))

# Tampilan hasil di Streamlit
st.title("Prediksi Harga Rumah")
st.subheader("Hasil Evaluasi Model")
st.write(f"Mean Absolute Error (MAE): {st.session_state.mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {st.session_state.rmse:.2f}")

# Visualisasi perbandingan antara nilai aktual dan prediksi
st.subheader("Perbandingan Nilai Aktual dan Prediksi")
fig, ax = plt.subplots()
sns.scatterplot(x=st.session_state.y_test, y=st.session_state.y_pred, ax=ax)
ax.set_xlabel("Nilai Aktual")
ax.set_ylabel("Nilai Prediksi")
ax.set_title("Perbandingan Harga Rumah Aktual dan Prediksi")
st.pyplot(fig)

# Fitur input untuk prediksi harga rumah
st.subheader("Masukkan Data untuk Prediksi Harga Rumah")
input_data = {
    "longitude": st.number_input("Longitude", value=0.0),
    "latitude": st.number_input("Latitude", value=0.0),
    "housing_median_age": st.number_input("Median Age of Houses", value=1),
    "total_rooms": st.number_input("Total Rooms", value=1),
    "total_bedrooms": st.number_input("Total Bedrooms", value=1),
    "population": st.number_input("Population", value=1),
    "households": st.number_input("Households", value=1),
    "median_income": st.number_input("Median Income", value=0.0),
}

# Menambahkan encoding untuk kolom kategori (ocean_proximity)
ocean_proximity_values = df.filter(like='ocean_proximity_').columns.tolist()
for ocean in ocean_proximity_values:
    input_data[ocean] = 0  # Default value
    selected_ocean = st.selectbox(f"Ocean Proximity ({ocean})", ["NEAR BAY", "NEAR OCEAN", "ISLAND", "INLAND"])
    if selected_ocean == ocean.split("_")[-1]:  # Check if selected
        input_data[ocean] = 1  # Set to 1 if selected

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Melakukan prediksi hanya jika tombol ditekan
if st.button("Prediksi Harga"):
    prediction = st.session_state.model.predict(input_df)
    st.write(f"Harga Prediksi Rumah: ${prediction[0]:,.2f}")
