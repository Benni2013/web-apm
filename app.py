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
from flask_sqlalchemy import SQLAlchemy
import io
from flask import send_file
from flask import send_from_directory
import shutil

app = Flask(__name__)
app.secret_key = 'your_secret_key'

basedir = os.path.abspath(os.path.dirname(__file__))
IMAGE_FOLDER = os.path.join(basedir, 'data', 'wajah')
CSV_FILE = "data/keterangan.csv"
USER_JSON = "data/users.json"
EMBEDDINGS_FILE = "data/embeddings.csv"
ATTENDANCE_FILE = "data/attendance.csv"  # File baru untuk menyimpan data absensi

app.config['SQLALCHEMY_DATABASE_URI']  = 'mysql+pymysql://root:@localhost/absen_apm'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Anggota(db.Model):
    __tablename__ = 'anggota'
    id_anggota = db.Column(db.String(20), primary_key=True)
    nama       = db.Column(db.String(100), nullable=False)
    divisi     = db.Column(db.String(50),  nullable=False)
    path_wajah = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(),
                            onupdate=db.func.current_timestamp())

class VektorWajah(db.Model):
    __tablename__ = 'vektor_wajah'
    id_vektor_wajah = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_anggota      = db.Column(db.String(20), db.ForeignKey('anggota.id_anggota',
                            onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    vektor          = db.Column(db.JSON, nullable=False)
    created_at      = db.Column(db.DateTime, default=db.func.current_timestamp())

class AbsenPiket(db.Model):
    __tablename__ = 'absen_piket'
    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_anggota = db.Column(db.String(20), db.ForeignKey('anggota.id_anggota',
                            onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    tanggal    = db.Column(db.Date,    nullable=False)
    waktu      = db.Column(db.Time,    nullable=False)
    status     = db.Column(db.Enum('Hadir','Tidak Hadir','Terlambat'),
                            nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    __table_args__ = (db.UniqueConstraint('id_anggota','tanggal',
                         name='unique_attendance'),)


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

def crop_face_oval(img):
    # 1. Buat mask oval di tengah frame (sesuai overlay CSS: 240×200 pada video 400×300)
    h, w = img.shape[:2]
    center = (w//2, h//2)
    axes = (int(w * 0.6 / 2), int(h * 0.6667 / 2))  # 0.6*400/2=120 ; 0.6667*300/2=100
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # 2. Terapkan mask: area di luar oval jadi hitam
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 3. Deteksi wajah hanya pada masked_img
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    # 4. Ambil crop bounding-box pertama dari masked_img
    x, y, w_box, h_box = faces[0]
    return masked_img[y:y+h_box, x:x+w_box]

def extract_face_embedding_with_facenet(img):
    face_img = crop_face_oval(img)
    if face_img is None:
        return None
    faces = embedder.extract(face_img, threshold=0.95)
    return faces[0]['embedding'] if faces else None


def save_face_data_and_embedding(id_, name_, divisi, img):
    # 1. Simpan file gambar ke disk (HANYA 1 FOTO)
    folder = os.path.join(IMAGE_FOLDER, f"{id_}_{name_}")
    os.makedirs(folder, exist_ok=True)
    fn = f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.jpg"
    path = os.path.join(folder, fn)
    cv2.imwrite(path, img)

    # Hitung path relatif ke IMAGE_FOLDER (untuk disimpan di DB)
    rel_path = os.path.relpath(path, IMAGE_FOLDER).replace('\\','/')

    # 2. Upsert ke tabel anggota (simpan rel_path, bukan absolute)
    anggota = Anggota.query.get(id_)
    if not anggota:
        anggota = Anggota(
            id_anggota=id_,
            nama=name_,
            divisi=divisi,
            path_wajah=rel_path
        )
        db.session.add(anggota)
    else:
        anggota.nama       = name_
        anggota.divisi     = divisi
        anggota.path_wajah = rel_path
    db.session.flush()

    # 3. Hapus semua vektor lama untuk anggota ini
    VektorWajah.query.filter_by(id_anggota=id_).delete()
    
    # 4. Extract embedding dari foto yang disimpan & simpan ke vektor_wajah
    embedding = extract_face_embedding_with_facenet(img)
    if embedding is not None:
        vw = VektorWajah(id_anggota=id_, vektor=list(map(float, embedding)))
        db.session.add(vw)

    db.session.commit()
    
def save_multiple_face_vectors(id_, vectors_data):
    """
    Simpan multiple vektor wajah ke database
    vectors_data: list of embedding arrays
    """
    try:
        # Hapus semua vektor lama untuk anggota ini
        VektorWajah.query.filter_by(id_anggota=id_).delete()
        
        # Simpan semua vektor baru
        for vector in vectors_data:
            if vector is not None and len(vector) > 0:
                vw = VektorWajah(
                    id_anggota=id_, 
                    vektor=list(map(float, vector))
                )
                db.session.add(vw)
        
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving vectors: {str(e)}")
        return False

# FUNGSI ABSENSI
def save_attendance(id_anggota):
    today = date.today()
    # Cek already absen
    exists = AbsenPiket.query.filter_by(
        id_anggota=id_anggota, tanggal=today).first()
    if exists:
        # Jika sudah absen hari ini, tampilkan nama anggota di pesan
        anggota = Anggota.query.get(id_anggota)
        return False, f"{anggota.nama} sudah absen hari ini!"
    # Simpan
    now = datetime.now().time()
    absen = AbsenPiket(id_anggota=id_anggota, tanggal=today,
                       waktu=now, status='Hadir')
    db.session.add(absen)
    db.session.commit()
    return True, "Absensi berhasil!"

def get_today_attendance():
    """Mengambil data absensi hari ini"""
    today = date.today().strftime('%Y-%m-%d')
    return get_attendance_by_date(today)

def get_attendance_by_date(selected_date):
    rows = (db.session.query(AbsenPiket, Anggota)
            .join(Anggota, AbsenPiket.id_anggota == Anggota.id_anggota)
            .filter(AbsenPiket.tanggal == selected_date)
            .all())
    result = []
    for absen, anggota in rows:
        result.append({
            'id_anggota' : anggota.id_anggota,
            'Nama': anggota.nama,
            'Divisi': anggota.divisi,
            'Tanggal': str(absen.tanggal),
            'Waktu': absen.waktu.strftime('%H:%M:%S'),
            'Status': absen.status
        })
    return result

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
@app.route('/wajah/<path:filename>')
@login_required
def wajah(filename):
    # kirim file di data/wajah/<filename>
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
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
    return redirect(url_for('dashboard'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # 1) Parse JSON payload
        data = request.get_json() or {}
        id_anggota    = data.get('id', '').strip()
        nama          = data.get('name', '').strip()
        divisi        = data.get('divisi', '').strip()
        webcam_image  = data.get('webcam_image', '')
        face_vectors  = data.get('face_vectors', [])  # Array of base64 images

        # 2) Validasi input wajib
        if not id_anggota or not nama or not divisi:
            return jsonify(success=False,
                           message='ID, Nama, dan Divisi harus diisi!')

        # 3) Cek duplikat ID anggota
        existing = Anggota.query.filter_by(id_anggota=id_anggota).first()
        if existing:
            return jsonify(success=False,
                           message=f'ID anggota "{id_anggota}" sudah terdaftar!')

        # 4) Decode gambar dari Base64
        if not webcam_image:
            return jsonify(success=False,
                           message='Silakan ambil foto sebelum submit!')
        try:
            header, encoded = webcam_image.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Decode gagal")
        except Exception:
            return jsonify(success=False,
                           message='Format gambar tidak valid!')

        # 5) Simpan foto utama dan data anggota
        save_face_data_and_embedding(id_anggota, nama, divisi, img)
        
        # 6) Proses dan simpan multiple vectors jika ada
        if face_vectors and len(face_vectors) > 0:
            vectors_data = []
            for vector_image in face_vectors:
                try:
                    if vector_image:
                        header, encoded = vector_image.split(',', 1)
                        img_bytes = base64.b64decode(encoded)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        vector_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Extract embedding
                        embedding = extract_face_embedding_with_facenet(vector_img)
                        if embedding is not None:
                            vectors_data.append(embedding)
                except Exception as e:
                    print(f"Error processing vector {len(vectors_data)}: {str(e)}")
                    continue
            
            # Simpan semua vectors ke database
            if vectors_data:
                save_multiple_face_vectors(id_anggota, vectors_data)
                return jsonify(success=True,
                               message=f'Anggota {nama} berhasil ditambahkan dengan {len(vectors_data)} vektor wajah!')
        
        return jsonify(success=True,
                       message=f'Anggota {nama} berhasil ditambahkan!')

    # GET: render halaman upload.html
    return render_template('upload.html',
                           username=session.get('user'))


@app.route('/export_excel')
@login_required
def export_excel():
    # 1. Ambil tanggal
    selected_date = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    # 2. Ambil data absensi
    attendance = get_attendance_by_date(selected_date)
    if not attendance:
        flash(f"Tidak ada data absensi pada {format_date_indonesia(selected_date)}", 'warning')
        return redirect(url_for('dashboard', date=selected_date))

    # 3. Buat DataFrame dan sisipkan kolom No
    df = pd.DataFrame(attendance)
    df.insert(0, 'No', range(1, len(df) + 1))   # kolom 'No' dari 1..n

    # 4. Tulis ke Excel di memori
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Absensi')
        worksheet = writer.sheets['Absensi']
        # Atur lebar kolom otomatis
        for idx, col in enumerate(df.columns):
            max_len = df[col].astype(str).map(len).max()
            header_len = len(col)
            width = max(max_len, header_len) + 2
            worksheet.set_column(idx, idx, width)

    output.seek(0)
    filename = f"Absensi_{selected_date}.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/scan_auto_page')
def scan_auto_page():
    return render_template('scan.html', username=session.get('user'))

@app.route('/scan_auto', methods=['POST'])
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
        
        # Ambil semua vektor dari database
        vectors = VektorWajah.query.all()
        if not vectors:
            return jsonify({'success': False, 'message': 'Dataset kosong.'})
        
        best_match = None
        best_score = 0
        
        for vector_row in vectors:
            try:
                stored_embedding = vector_row.vektor
                if stored_embedding and len(stored_embedding) > 0:
                    similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]
                    if similarity > best_score:
                        best_score = similarity
                        best_match = vector_row.id_anggota
            except Exception as e:
                print(f"Error processing vector: {str(e)}")
                continue
        
        if best_score > 0.7 and best_match:
            anggota = Anggota.query.get(best_match)
            if anggota:
                success, message = save_attendance(anggota.id_anggota)
                if success:
                    return jsonify({
                        'success': True, 
                        'message': f'Selamat datang, {anggota.nama}!',
                        'name': anggota.nama,
                        'confidence': f'{best_score:.2f}'
                    })
                else:
                    return jsonify({
                        'success': False, 
                        'message': message,
                        'name': anggota.nama
                    })
        
        return jsonify({'success': False, 'message': 'Wajah tidak dikenal'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/members')
@login_required
def members():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # jumlah data per halaman, sesuaikan kebutuhan

    pagination = Anggota.query.paginate(page=page, per_page=per_page, error_out=False)
    anggota_list = pagination.items

    for m in anggota_list:
        if m.path_wajah:
            m.path_rel = m.path_wajah
            m.display_path = f"data/wajah/{m.path_wajah}"
        else:
            m.path_rel = None
            m.display_path = None

    return render_template(
        'members.html',
        members=anggota_list,
        pagination=pagination,
        username=session.get('user')
    )

@app.route('/add_member', methods=['GET', 'POST'])
@login_required
def add_member():
    """Tambah anggota baru ke database"""
    if request.method == 'POST':
        id_ = request.form['id'].strip()
        name_ = request.form['name'].strip()
        divisi = request.form['divisi'].strip()
        
        if not id_ or not name_ or not divisi:
            flash('ID, Nama, dan Divisi harus diisi!', 'danger')
            return render_template('add_member.html')
        
        try:
            # Cek apakah ID sudah ada di database
            existing_anggota = Anggota.query.get(id_)
            if existing_anggota:
                flash('ID sudah ada! Gunakan ID yang berbeda.', 'danger')
                return render_template('add_member.html')
            
            # Tambah anggota baru ke database
            new_anggota = Anggota(
                id_anggota=id_,
                nama=name_,
                divisi=divisi,
                path_wajah=""  # Path kosong, akan diisi saat upload foto
            )
            db.session.add(new_anggota)
            db.session.commit()
            
            flash(f'Anggota {name_} berhasil ditambahkan!', 'success')
            return redirect(url_for('members'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error menambah anggota: {str(e)}', 'danger')
            return render_template('add_member.html')
    
    # return render_template('add_member.html')
    return redirect(url_for('members'))  # Redirect ke halaman anggota jika GET


@app.route('/edit_member/<member_id>', methods=['GET', 'POST'])
@login_required
def edit_member(member_id):
    """Edit data anggota via database"""
    anggota = Anggota.query.get(member_id)
    if not anggota:
        flash('Anggota tidak ditemukan!', 'danger')
        return redirect(url_for('members'))
    
    if request.method == 'POST':
        new_name = request.form.get('name', '').strip()
        new_divisi = request.form.get('divisi', '').strip()
        
        if not new_name or not new_divisi:
            flash('Nama dan Divisi harus diisi!', 'danger')
            return render_template('edit_member.html', member=anggota)
        
        # Update atribut anggota
        old_name = anggota.nama
        anggota.nama = new_name
        anggota.divisi = new_divisi
        
        # Commit perubahan
        db.session.commit()
        
        # Update nama di embeddings, jika ada
        embeddings = VektorWajah.query.filter_by(id_anggota=member_id).all()
        for e in embeddings:
            # Jika kamu simpan nama di embeddings (biasanya ID saja, kalau nama hapus ini)
            pass  # Biasanya embed simpan ID, jadi tidak perlu update
        
        flash('Data anggota berhasil diupdate!', 'success')
        return redirect(url_for('members'))
    
    return render_template('edit_member.html', member=anggota)

@app.route('/delete_member/<member_id>', methods=['POST'])
@login_required
def delete_member(member_id):
    """Hapus anggota dari database (folder foto dibiarkan untuk pembersihan manual)"""
    try:
        # 1. Cari anggota di DB
        anggota = Anggota.query.get(member_id)
        if not anggota:
            flash('Anggota tidak ditemukan!', 'danger')
            return redirect(url_for('members'))

        nama = anggota.nama
        
        # 2. Hapus dari database (CASCADE akan menghapus data terkait)
        db.session.delete(anggota)
        db.session.commit()
        
        # 3. Coba hapus folder foto (optional - tidak akan error jika gagal)
        try:
            folder_name = f"{member_id}_{nama}"
            folder_path = os.path.join(IMAGE_FOLDER, folder_name)
            
            if os.path.exists(folder_path):
                # Ganti nama folder dengan prefix "_DELETED_" jika tidak bisa dihapus
                import time
                timestamp = int(time.time())
                new_name = f"_DELETED_{timestamp}_{folder_name}"
                new_path = os.path.join(IMAGE_FOLDER, new_name)
                
                try:
                    shutil.rmtree(folder_path)
                    folder_msg = " Folder foto juga berhasil dihapus."
                except:
                    try:
                        os.rename(folder_path, new_path)
                        folder_msg = " Folder foto ditandai untuk penghapusan manual."
                    except:
                        folder_msg = " Folder foto tidak dapat dihapus (silakan hapus manual)."
            else:
                folder_msg = ""
                
        except Exception as folder_error:
            folder_msg = f" (Folder foto: {str(folder_error)})"
        
        flash(f'Anggota {nama} berhasil dihapus dari database!{folder_msg}', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error dalam delete_member: {str(e)}")
        flash(f'Error menghapus anggota: {str(e)}', 'danger')
    
    return redirect(url_for('members'))


if __name__ == '__main__':
    app.run(debug=True)