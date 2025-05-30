
# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Deskripsi Singkat
Repositori ini merupakan implementasi tugas besar kedua dari mata kuliah IF3270 Pembelajaran Mesin. Fokus utama tugas ini adalah implementasi dan eksperimen model **CNN**, **RNN**, dan **LSTM** untuk klasifikasi gambar dan teks, baik dengan Keras maupun menggunakan implementasi manual (from scratch).

### Fitur utama:
- Klasifikasi gambar menggunakan CNN (dataset: CIFAR-10)
- Klasifikasi teks Bahasa Indonesia menggunakan RNN dan LSTM (dataset: NusaX-Sentiment)
- Eksperimen variasi arsitektur (jumlah layer, neuron, arah propagasi, dll)
- Implementasi forward propagation dari nol (NumPy / PyTorch)
- Evaluasi menggunakan **macro F1-score**

---

## Struktur Folder
```
├── data/
│   └── nusaX-sentiment/ # dataset buat lstm dan rnn sudah dibagi jadi test, train, dan val
├── doc/ # berisi dokumen laporan
├── src/
│   ├── helper/ # berisi kelas-kelas pemabntu seperti vectorization
│   ├── models/ 
│   │   ├── cnn/ # implementasi cnn
│   │   ├── lstm/ # implementasi lstm
│   │   ├── nn/ # implementasi nn dari tubes 1 untuk keperluan dense layer
│   │   └── rnn/ # implementasi rnn
│   ├── cnn.ipynb file uji untuk cnn
│   ├── simple_lstm_back.ipynb # file uji backward manual pada lstm
│   ├── simple_lstm.ipynb # file uji untuk lstm
│   ├── simple_rnn.ipynb # file uji untuk rnn
├── .python-version
├── README.md
└── requirements.txt
```

---

## Cara Setup dan Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/mybajwk/ML-2-8.git
cd ML-2-8
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Jalankan Notebook
Masuk ke folder `src/`, lalu buka dan jalankan salah satu notebook:
- simple_rnn.ipynb (uji rnn)
- simple_lstm.ipynb (uji lstm)
- cnn.ipynb (uji cnn)

---
## Contoh Menggunakan Kode yang Kami Buat
### 1. CNN
```bash
# Inisialisasi model Keras
model = SimpleRNNKeras(
    max_vocab=10000,              # Ukuran maksimum kosakata (vocab size)
    max_len=100,                  # Panjang maksimum input sequence
    embedding_dim=128,            # Ukuran embedding vektor tiap token
    rnn_units=[64, 64],           # Jumlah neuron tiap layer RNN (2 layer, 64 unit)
    rnn_activations='tanh',       # Fungsi aktivasi (string tunggal atau list per layer)
    dense_units=[32, 3],          # Dense layer, output akhir = jumlah kelas (3)
    dense_activations=['relu', 'softmax'],  # Aktivasi dense layer terakhir = softmax
    bidirectional=True,           # Jika True, gunakan bidirectional RNN
    dropout=0.5,                  # Dropout setelah layer RNN
    learning_rate=1e-3            # Learning rate untuk Adam optimizer
)

# Set data vectorized
model.set_vectorized_data(X_train, y_train, X_valid, y_valid, X_test, y_test)

# Build & train model
model.build_model()
model.train(epochs=10, batch_size=64)

# Evaluasi macro F1-score
y_pred, f1 = model.evaluate()
print("F1-Score:", f1)

# Simpan bobot dan konfigurasi ke .npy (untuk load ke implementasi manual)
model.save_full_npy("model_simple_rnn.npy")
```

```bash
model = SimpleRNNManual()

# Load konfigurasi dan bobot dari hasil training Keras (format .npy)
model.load_full_npy("model_simple_rnn.npy")

# Prediksi (argmax dari output softmax)
x_token_ids = torch.tensor([...])   # Tensor integer token ID (batch_size, seq_len)
y_pred = model.predict(x_token_ids)

# Evaluasi macro F1-score
y_true = torch.tensor([...])
f1 = model.evaluate(x_token_ids, y_true)
```

### 2. RNN

### 3. LSTM

---

## Pembagian Tugas Anggota Kelompok
| Nama                | NIM        | Tugas                                                                 |
|---------------------|------------|-----------------------------------------------------------------------|
| Wilson Yusda        | 13522019   | Mengerjakan segala hal yang berhubungan dengan CNN          |
| Mesach Harmasendro  | 13522117   | Mengerjakan segala hal yang berhubungan dengan RNN     |
| Enrique Yanuar      | 13522077   | Mengerjakan segala hal yang berhubungan dengan LSTM      |

---

