# Web Absensi Face Recognition APM

Aplikasi web untuk sistem absensi menggunakan teknologi face recognition (pengenalan wajah) yang dibangun dengan Flask dan FaceNet. Sistem ini memungkinkan pencatatan kehadiran anggota secara otomatis melalui deteksi wajah.

## 📋 Fitur Utama

- **Face Recognition**: Deteksi dan pengenalan wajah menggunakan FaceNet
- **Manajemen Anggota**: Tambah, edit, dan hapus data anggota
- **Absensi Real-time**: Sistem absensi otomatis dengan webcam
- **Dashboard**: Tampilan data absensi harian dengan filter tanggal
- **Export Data**: Export data absensi ke format Excel
- **User Authentication**: Sistem login dan registrasi pengguna
- **Database Integration**: Menggunakan MySQL untuk penyimpanan data

## 🛠 Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Database**: MySQL dengan SQLAlchemy ORM
- **Face Recognition**: Keras-FaceNet, OpenCV
- **Machine Learning**: TensorFlow, Scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Export**: XlsxWriter untuk Excel export

## 📁 Struktur Project

```
web-apm/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Data storage directory
│   ├── wajah/           # Face images storage
│   ├── attendance.csv   # Attendance records
│   ├── embeddings.csv   # Face embeddings
│   ├── users.json       # User authentication data
│   └── database_absen_apm.sql  # Database schema
├── static/              # Static files (CSS, images)
├── templates/           # HTML templates
└── ...
```

## 🚀 Instalasi dan Setup

### Prerequisites

1. **Python 3.8+**
2. **MySQL Server**
3. **Webcam** (untuk fitur face recognition)

### Langkah Instalasi

1. **Clone atau download project**
   ```bash
   git clone <repository-url>
   cd web-apm
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Database MySQL**
   - Buat database baru bernama `absen_apm`
   - Import schema database:
   ```bash
   mysql -u root -p absen_apm < data/database_absen_apm.sql
   ```

5. **Konfigurasi Database**
   - Pastikan MySQL server berjalan
   - Sesuaikan konfigurasi database di `app.py` jika diperlukan:
   ```python
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/absen_apm'
   ```

## ▶️ Cara Menjalankan Aplikasi

1. **Aktivasi virtual environment** (jika menggunakan)
   ```bash
   venv\Scripts\activate  # Windows
   ```

2. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

3. **Akses aplikasi**
   - Buka browser dan kunjungi: `http://localhost:5000`
   - Atau: `http://127.0.0.1:5000`

## 👤 Penggunaan

### First Setup
1. Registrasi akun admin di `/register`
2. Login menggunakan akun yang telah dibuat
3. Tambah anggota baru di menu "Kelola Anggota"
4. Upload foto wajah untuk setiap anggota

### Absensi
1. Akses halaman "Scan Absensi"
2. Izinkan akses webcam
3. Posisikan wajah di dalam frame oval
4. Sistem akan otomatis mendeteksi dan mencatat kehadiran

### Dashboard
- Lihat data absensi harian
- Filter berdasarkan tanggal
- Export data ke Excel

## 🔧 Konfigurasi

### Database Connection
Edit konfigurasi database di `app.py`:
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://username:password@host/database_name'
```

### Secret Key
Ganti secret key untuk keamanan:
```python
app.secret_key = 'your_secret_key_here'
```

## 📊 Database Schema

- **anggota**: Data anggota (ID, nama, divisi, path foto)
- **vektor_wajah**: Embedding wajah untuk face recognition
- **absen_piket**: Record absensi harian

## 🐛 Troubleshooting

### Error Database Connection
- Pastikan MySQL server berjalan
- Check username, password, dan nama database
- Install PyMySQL: `pip install pymysql`

### Error Webcam
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Check permission browser untuk mengakses webcam

### Error Face Detection
- Pastikan pencahayaan cukup
- Posisikan wajah tegak dan menghadap kamera
- Hindari penggunaan mask atau kacamata gelap

---

**Dibangun untuk memenuhi tugas besar mata kuliah APM (Aplikasi Pembelajaran Mesin)**