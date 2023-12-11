import streamlit as st
import librosa
import numpy as np
import scipy
import joblib
from sklearn.decomposition import PCA

# Muat model KNN yang telah dilatih
loaded_scaler = joblib.load('scaler.pkl')

# Muat model KNN yang telah dilatih
knn = joblib.load('knn_model.pkl')

# Muat matriks komponen PCA
pca_components = np.load('pca_components.npy')

# Fungsi ekstraksi fitur dari file audio
def extract_features(audio_file):
    # Gunakan librosa untuk ekstraksi fitur-fitur audio
    y, sr = librosa.load(audio_file)
    
    # Lakukan ekstraksi fitur audio sesuai kebutuhan Anda
    # Misalnya, ambil 21 fitur pertama dari ekstraksi
    freqs = np.fft.fftfreq(y.size)
    freqs_flat = freqs.flatten()
    mean = np.mean(freqs)
    std = np.std(freqs)
    maxv = np.amax(freqs)
    minv = np.amin(freqs)
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs,0.75)
    mode = scipy.stats.mode(freqs)[0][0]
    iqr = scipy.stats.iqr(freqs_flat)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_flat = zcr.flatten()
    zcr_mean = zcr.mean()
    zcr_median = np.median(zcr)
    zcr_kurt = scipy.stats.kurtosis(zcr_flat)
    zcr_skew = scipy.stats.skew(zcr_flat)
    zcr_std = zcr.std()
    rmse = librosa.feature.rms(y=y)
    rmse_flat = rmse.flatten()
    rmse_mean = rmse.mean()
    rmse_median = np.median(rmse)
    rmse_kurt = scipy.stats.kurtosis(rmse_flat)
    rmse_skew = scipy.stats.skew(rmse_flat)
    rmse_std = rmse.std()
    
    # Kumpulkan semua fitur dalam satu list
    features = np.array([mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr, zcr_mean, zcr_median, zcr_kurt, zcr_skew, zcr_std, rmse_mean, rmse_median, rmse_kurt, rmse_skew, rmse_std])

    #Menampilkan fitur sebelum di normalisasi
    st.write("Hasil Ekstraksi Audio Sebelum Normalisasi")
    st.write(np.array(features).reshape(1, -1))

    # Menggunakan scaler untuk transformasi data
    normalized_data = loaded_scaler.transform(features.reshape(1, -1))
    
    #Menampilkan fitur setelah di normalisasi
    st.write("Hasil Ekstraksi Audio Setelah Normalisasi")
    st.write(normalized_data)

    return normalized_data

# Fungsi prediksi menggunakan model KNN dan PCA
def predict_emotion(audio_features):
    # Transformasi fitur dengan PCA
    audio_features_pca = np.dot(audio_features, pca_components.T)

    #Menampilkan fitur setelah pca
    st.write("Hasil Ekstraksi Audio Setelah PCA (n_component=20)")
    st.write(np.array(audio_features_pca).reshape(1, -1))

    # Prediksi emosi menggunakan model KNN
    prediction = knn.predict(audio_features_pca)
    
    return prediction[0]

def main():
    st.title('Aplikasi Pengenalan Emosi dari Suara')
    st.sidebar.header('Unggah File Audio')

    # Widget untuk pengguna mengunggah file audio
    uploaded_file = st.sidebar.file_uploader("Pilih file audio", type=["wav"])

    if uploaded_file is not None:
        st.write("File berhasil diunggah!")

        # Ekstraksi fitur dari file audio yang diunggah
        audio_features=extract_features(uploaded_file)
        

        if st.sidebar.button('Prediksi Emosi'):
            prediction = predict_emotion(audio_features)
            st.write('Hasil Prediksi Emosi:', prediction)

if __name__ == "__main__":
    main()
