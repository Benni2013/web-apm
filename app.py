import os
import cv2
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

IMAGE_FOLDER = "data/wajah/"
CSV_FILE = "data/keterangan.csv"
USER_JSON = "data/users.json"
EMBEDDINGS_FILE = "data/embeddings.csv"
ATTENDANCE_FILE = "data/attendance.csv"  # File baru untuk menyimpan data absensi

# Buat folder & file data kalau belum ada
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["ID", "Nama", "Path Wajah"])
    df.to_csv(CSV_FILE, index=False)
if not os.path.exists(USER_JSON):
    with open(USER_JSON, "w") as f:
        json.dump([], f)
if not os.path.exists(EMBEDDINGS_FILE):
    # Satu kolom nama + 512 kolom embedding float
    df = pd.DataFrame(columns=["Nama"] + [f"e{i}" for i in range(512)])
    df.to_csv(EMBEDDINGS_FILE, index=False)
if not os.path.exists(ATTENDANCE_FILE):
    # File untuk menyimpan data absensi
    df = pd.DataFrame(columns=["Nama", "Tanggal", "Waktu", "Status"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# USER JSON
def load_users():
    with open(USER_JSON, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_JSON, 'w') as f:
        json.dump(users, f, indent=2)

def get_user(username):
    users = load_users()
    for user in users:
        if user["username"] == username:
            return user
    return None

def add_user(username, password):
    users = load_users()
    if get_user(username) is not None:
        return False
    new_id = (max([u['id'] for u in users]) + 1) if users else 1
    users.append({
        "id": new_id,
        "username": username,
        "password_hash": generate_password_hash(password)
    })
    save_users(users)
    return True

def verify_user(username, password):
    user = get_user(username)
    if user and check_password_hash(user["password_hash"], password):
        return True
    return False

# KERAS-FACENET
embedder = FaceNet()

def extract_face_embedding_with_facenet(img):
    faces = embedder.extract(img, threshold=0.95)
    if faces:
        return faces[0]['embedding']
    else:
        return None

def save_face_data_and_embedding(id_, name_, img):
    # Save image to personal folder
    folder_path = os.path.join(IMAGE_FOLDER, f"{id_}_{name_}")
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join(folder_path, filename)
    cv2.imwrite(image_path, img)

    # Update CSV data keterangan
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([{"ID": id_, "Nama": name_, "Path Wajah": image_path}])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    # Extract embedding dan simpan ke embeddings.csv
    embedding = extract_face_embedding_with_facenet(image_path)
    if embedding is not None:
        df_e = pd.read_csv(EMBEDDINGS_FILE)
        # Kolom = Nama, e0, e1, ..., e511
        new_row = pd.DataFrame([[name_] + list(embedding)], columns=df_e.columns)
        df_e = pd.concat([df_e, new_row], ignore_index=True)
        df_e.to_csv(EMBEDDINGS_FILE, index=False)

# FUNGSI ABSENSI
def save_attendance(nama):
    """Menyimpan data absensi ke file CSV"""
    today = date.today().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%H:%M:%S')
    
    # Cek apakah sudah absen hari ini
    df_attendance = pd.read_csv(ATTENDANCE_FILE)
    if not df_attendance.empty:
        today_attendance = df_attendance[
            (df_attendance['Nama'] == nama) & 
            (df_attendance['Tanggal'] == today)
        ]
        if not today_attendance.empty:
            return False, "Anda sudah absen hari ini!"
    
    # Simpan absensi baru
    new_attendance = pd.DataFrame([{
        "Nama": nama,
        "Tanggal": today,
        "Waktu": now,
        "Status": "Hadir"
    }])
    
    df_attendance = pd.concat([df_attendance, new_attendance], ignore_index=True)
    df_attendance.to_csv(ATTENDANCE_FILE, index=False)
    return True, "Absensi berhasil!"

def get_today_attendance():
    """Mengambil data absensi hari ini"""
    today = date.today().strftime('%Y-%m-%d')
    return get_attendance_by_date(today)

def get_attendance_by_date(selected_date):
    """Mengambil data absensi berdasarkan tanggal tertentu"""
    try:
        df_attendance = pd.read_csv(ATTENDANCE_FILE)
        if df_attendance.empty:
            return []
        
        # Filter berdasarkan tanggal
        date_attendance = df_attendance[df_attendance['Tanggal'] == selected_date]
        
        # Konversi ke list of dict dan tambahkan informasi divisi
        attendance_list = []
        df_keterangan = pd.read_csv(CSV_FILE)
        
        for _, row in date_attendance.iterrows():
            attendance_data = {
                'Nama': row['Nama'],
                'Tanggal': row['Tanggal'],
                'Waktu': row['Waktu'],
                'Status': row['Status'],
                'Divisi': 'Pengembangan'  # Default divisi, bisa disesuaikan
            }
            
            # Cari divisi dari data keterangan jika ada
            member_info = df_keterangan[df_keterangan['Nama'] == row['Nama']]
            if not member_info.empty and 'Divisi' in member_info.columns:
                attendance_data['Divisi'] = member_info.iloc[0]['Divisi']
            
            attendance_list.append(attendance_data)
        
        return attendance_list
    
    except Exception as e:
        print(f"Error getting attendance by date: {e}")
        return []

def format_date_indonesia(date_str):
    """Format tanggal ke bahasa Indonesia"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        months = [
            'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
            'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
        ]
        return f"{date_obj.day} {months[date_obj.month - 1]} {date_obj.year}"
    except:
        return date_str

# LOGIN DECORATOR
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Silakan login dulu.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ROUTES
@app.route('/')
@login_required
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Ambil tanggal dari parameter atau gunakan hari ini
    selected_date = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    
    # Ambil data absensi berdasarkan tanggal
    attendance_list = get_attendance_by_date(selected_date)
    
    # Format tanggal untuk tampilan
    today_date_formatted = format_date_indonesia(selected_date)
    
    return render_template('home.html', 
                         username=session.get('user'),
                         attendance_list=attendance_list,
                         today_date=today_date_formatted,
                         today_date_iso=selected_date)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            flash('Password tidak cocok.', 'danger')
            return render_template('register.html')
        if len(password) < 5:
            flash('Password minimal 5 karakter.', 'danger')
            return render_template('register.html')
        if add_user(username, password):
            flash('Registrasi berhasil, silakan login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username sudah ada!', 'danger')
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            flash('Password tidak cocok.', 'danger')
            return render_template('forgot_password.html')
        users = load_users()
        user = get_user(username)
        if user is None:
            flash('Username tidak ditemukan.', 'danger')
            return render_template('forgot_password.html')
        # Update password
        for u in users:
            if u["username"] == username:
                u["password_hash"] = generate_password_hash(password)
        save_users(users)
        flash('Password berhasil direset. Silakan login!', 'success')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if verify_user(username, password):
            session['user'] = username
            flash('Login sukses!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('user', None)
    flash('Logout berhasil.', 'info')
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        id_ = request.form['id']
        name_ = request.form['name']
        img = None

        # 1. Dari file
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            file_path = os.path.join(IMAGE_FOLDER, filename)
            file.save(file_path)
            img = cv2.imread(file_path)

        # 2. Dari webcam (base64 string)
        elif 'webcam_image' in request.form and request.form['webcam_image'] != '':
            data_url = request.form['webcam_image']
            # format data_url: "data:image/jpeg;base64,....."
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Jika ada gambar, simpan dan embedding
        if img is not None:
            save_face_data_and_embedding(id_, name_, img)
            flash('Data wajah & embedding berhasil di-upload!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Gambar belum dipilih atau diambil!', 'danger')
    return render_template('upload.html')

@app.route('/export_pdf')
@login_required
def export_pdf():
    selected_date = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    
    # Ambil data absensi berdasarkan tanggal
    attendance_list = get_attendance_by_date(selected_date)
    
    # Format tanggal untuk judul laporan
    formatted_date = format_date_indonesia(selected_date)
    
    # Di sini Anda bisa menambahkan logic untuk generate PDF
    # Untuk saat ini, saya akan mengembalikan response sederhana
    flash(f'Export PDF untuk tanggal {formatted_date} - Total: {len(attendance_list)} orang', 'info')
    return redirect(url_for('dashboard', date=selected_date))

@app.route('/scan_auto_page')
@login_required
def scan_auto_page():
    return render_template('scan.html')

@app.route('/scan_auto', methods=['POST'])
@login_required
def scan_auto():
    try:
        data_url = request.form['webcam_image']
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Proses seperti biasa
        test_embedding = extract_face_embedding_with_facenet(img)
        if test_embedding is None:
            return jsonify({'success': False, 'message': 'Tidak terdeteksi wajah.'})
        
        df = pd.read_csv(EMBEDDINGS_FILE)
        if df.empty or len(df) == 0:
            return jsonify({'success': False, 'message': 'Dataset kosong.'})
        
        names = df["Nama"].values
        embeddings = df.drop(columns=["Nama"]).values.astype(float)
        sims = cosine_similarity([test_embedding], embeddings)
        best_idx = np.argmax(sims)
        best_score = sims[0][best_idx]
        
        if best_score > 0.7:
            name = names[best_idx]
            # Simpan absensi
            success, message = save_attendance(name)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'Selamat datang, {name}!',
                    'name': name,
                    'confidence': f'{best_score:.2f}'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': message,
                    'name': name
                })
        else:
            return jsonify({'success': False, 'message': 'Wajah tidak dikenal'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# Tambahkan route-route ini ke file app.py Anda

@app.route('/members')
@login_required
def members():
    """Halaman daftar semua anggota"""
    df = pd.read_csv(CSV_FILE)
    members_data = df.to_dict('records')
    return render_template('members.html', members=members_data)

@app.route('/add_member', methods=['GET', 'POST'])
@login_required
def add_member():
    """Tambah anggota baru"""
    if request.method == 'POST':
        id_ = request.form['id'].strip()
        name_ = request.form['name'].strip()
        divisi = request.form['divisi'].strip()
        
        if not id_ or not name_ or not divisi:
            flash('ID, Nama, dan Divisi harus diisi!', 'danger')
            return render_template('add_member.html')
        
        # Cek apakah ID sudah ada
        df = pd.read_csv(CSV_FILE)
        if not df.empty and id_ in df['ID'].astype(str).values:
            flash('ID sudah ada! Gunakan ID yang berbeda.', 'danger')
            return render_template('add_member.html')
        
        # Tambah anggota dengan kolom Divisi
        new_member = pd.DataFrame([{"ID": id_, "Nama": name_, "Divisi": divisi, "Path Wajah": ""}])
        df = pd.concat([df, new_member], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        
        flash(f'Anggota {name_} berhasil ditambahkan!', 'success')
        return redirect(url_for('members'))
    
    return render_template('add_member.html')

@app.route('/edit_member/<member_id>', methods=['GET', 'POST'])
@login_required
def edit_member(member_id):
    """Edit data anggota"""
    df = pd.read_csv(CSV_FILE)
    
    if df.empty or member_id not in df['ID'].astype(str).values:
        flash('Anggota tidak ditemukan!', 'danger')
        return redirect(url_for('members'))
    
    member = df[df['ID'].astype(str) == member_id].iloc[0]
    
    if request.method == 'POST':
        new_name = request.form['name'].strip()
        new_divisi = request.form['divisi'].strip()
        
        if not new_name or not new_divisi:
            flash('Nama dan Divisi harus diisi!', 'danger')
            return render_template('edit_member.html', member=member)
        
        # Update data anggota
        df.loc[df['ID'].astype(str) == member_id, 'Nama'] = new_name
        df.loc[df['ID'].astype(str) == member_id, 'Divisi'] = new_divisi
        df.to_csv(CSV_FILE, index=False)
        
        # Update nama di embeddings jika ada
        if os.path.exists(EMBEDDINGS_FILE):
            df_e = pd.read_csv(EMBEDDINGS_FILE)
            if not df_e.empty:
                df_e.loc[df_e['Nama'] == member['Nama'], 'Nama'] = new_name
                df_e.to_csv(EMBEDDINGS_FILE, index=False)
        
        flash(f'Data anggota berhasil diupdate!', 'success')
        return redirect(url_for('members'))
    
    return render_template('edit_member.html', member=member)

@app.route('/delete_member/<member_id>', methods=['POST'])
@login_required
def delete_member(member_id):
    """Hapus anggota"""
    df = pd.read_csv(CSV_FILE)
    
    if df.empty or member_id not in df['ID'].astype(str).values:
        flash('Anggota tidak ditemukan!', 'danger')
        return redirect(url_for('members'))
    
    member = df[df['ID'].astype(str) == member_id].iloc[0]
    member_name = member['Nama']
    
    # Hapus dari CSV utama
    df = df[df['ID'].astype(str) != member_id]
    df.to_csv(CSV_FILE, index=False)
    
    # Hapus dari embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        df_e = pd.read_csv(EMBEDDINGS_FILE)
        if not df_e.empty:
            df_e = df_e[df_e['Nama'] != member_name]
            df_e.to_csv(EMBEDDINGS_FILE, index=False)
    
    # Hapus folder foto jika ada
    folder_path = os.path.join(IMAGE_FOLDER, f"{member_id}_{member_name}")
    if os.path.exists(folder_path):
        import shutil
        shutil.rmtree(folder_path)
    
    flash(f'Anggota {member_name} berhasil dihapus!', 'success')
    return redirect(url_for('members'))

if __name__ == '__main__':
    app.run(debug=True)