from flask import Flask, render_template, request
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import psycopg2
import pandas as pd
import re
import joblib
import os
from datetime import datetime
from preprocessing import preprocessing

app = Flask(__name__)
# app.secret_key = "rahasia123"

# API key youtube
API_KEY = "AIzaSyCGRpEG3tvWjh2Y9qbZO1MIGWKK2fueoH0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model
tfidf = joblib.load(os.path.join(BASE_DIR, "model/tfidf_vectorizer1.joblib"))
model = joblib.load(os.path.join(BASE_DIR, "model/svm_model1.joblib"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "model/label_encoder1.joblib"))

# load kamus NRC + Domain
nrc_path = os.path.join(BASE_DIR, "data/Indonesian-NRC-EmoLex (1).txt")
domain_path = os.path.join(BASE_DIR, "data/kamus-domain.csv")

kamus_nrc = {}

with open (nrc_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
    # skip header (baris pertama)
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 12:
            continue
        
        indonesian_word = parts[-1]
       # english_word = parts[0]
        
        # nama emosi sesuai urutan kolom
        emotion_mapping = {
            "anger": parts[1],
            "anticipation": parts[2],
            "disgust": parts[3],
            "fear": parts[4],
            "joy": parts[5],
            "sadness": parts[8],
            "surprise": parts[9],
            "trust": parts[10]
        }

        emotions = [label for label, val in emotion_mapping.items() if val == "1"]
                
        # simpan berdasarkan kata Indonesia
        if emotions:
            kamus_nrc[indonesian_word] = emotions
            #kamus_nrc[english_word] = emotions

# load kamus gabungan
domain_df = pd.read_csv(domain_path)
for _, row in domain_df.iterrows():
    kata = row['kata']
    # konversi dari format CSV ke format list seperti kamus_nrc
    emotions = [
        col for col in domain_df.columns 
        if col != 'kata' and row[col] == 1
    ]
    if emotions:
        kamus_nrc[kata] = emotions  


# tandai kata yang berpengaruh dalam emosi
def highlight_by_emotion(text, kamus_nrc, label):
    words = text.split()
    result = []

    for w in words:
        if w in kamus_nrc:
            daftar_emosi = kamus_nrc[w]
            if label in daftar_emosi:
                result.append(f"<span class='highlight-{label}' title='{label}'>{w}</span>")
            else:
                result.append(w)
        else:
            result.append(w)

    return " ".join(result)


# konek database
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/emosi_db")
try:
    db = psycopg2.connect(DATABASE_URL)
    cursor = db.cursor()

    # data kosong saat run ulang
    cursor.execute("DELETE FROM comments")
    db.commit()
except Exception as e:
    print("Database connection error:", e)
    db = None
    cursor = None

# ambil video ID dari link Youtube
def extract_video_id(link):
    try:
        link = link.strip()
        
        if"youtu.be/" in link:
            return link.split("youtu.be/")[1].split("?")[0].split("&")[0]
        
        url = urlparse(link)

        if"youtube.com" in url.netloc:
            qs = parse_qs(url.query)
            if "v" in qs:
                return qs["v"][0]
            
            match = re.search(r"/(shorts|embed|v)/([a-zA-Z0-9_-]{11})", url.path)
            if match:
                return match.group(2)

    except Exception:
        return None
    
    return None

# ekstrak beberapa URL 
def extract_multiple_video_ids(links_raw):
    lines = re.split(r"[\n,]+", links_raw)
    valid_ids = []
    invalid_urls = []
    seen = set()
    
    for line in lines:
        url = line.strip()
        if not url:
            continue
        
        vid = extract_video_id(url)
        
        if vid and vid not in seen:
            valid_ids.append(vid)
            seen.add(vid)
        elif not vid:
            invalid_urls.append(url)
        
    return valid_ids, invalid_urls

# Ambil komentar dari banyak video 
def get_comments_from_multiple_videos(video_ids, max_per_video=100):
    all_comments = []
    errors = []
 
    for vid in video_ids:
        try:
            comments = get_youtube_comments(vid, max_results=max_per_video)
            all_comments.extend(comments)
            print(f"[OK] Video {vid}: {len(comments)} komentar")
        except Exception as e:
            print(f"[ERROR] Video {vid}: {e}")
            errors.append(vid)
 
    return all_comments, errors


# satu video youtube
def get_youtube_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    comments = []
    next_page_token = None
    
    while len (comments) < max_results:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token,
            textFormat="plainText"

        ).execute()

        for item in response.get("items", []):
            if len(comments) >= max_results:
                break
            
            snippet = item["snippet"]["topLevelComment"]["snippet"]

        # ubah format datetime
            waktu_iso = snippet["publishedAt"]
            waktu_mysql = datetime.strptime(waktu_iso, "%Y-%m-%dT%H:%M:%SZ")

            comments.append({
                "video_id": video_id,
                "waktu": waktu_mysql,
                "komentar": snippet["textDisplay"],
                "likes": snippet["likeCount"]
            })
    
        next_page_token = response.get("nextPageToken")
        
        if not next_page_token:
            break

    return comments


# prediksi emosi
def prediksi_emosi(text):
    try:
        text = preprocessing(text)
        text_tfidf = tfidf.transform([text])
        pred = model.predict(text_tfidf)
        label = label_encoder.inverse_transform(pred)[0]
        return label
    except Exception as e:
        print("Error:", e)
        return "netral"
    

# Route Upload Link
@app.route("/", methods=["GET", "POST"])
def upload():
    
    # pagination setup
    page = int(request.args.get("page", 1))
    per_page = 10
    pesan_error = None

    if request.method == "POST":
        links_list = request.form.getlist("link")
        links_raw = "\n".join([link.strip() for link in links_list if link.strip()])
        video_ids, invalid_urls = extract_multiple_video_ids(links_raw)
        
        if invalid_urls:
            pesan_error = f"{len(invalid_urls)} URL tidak valid diabaikan: {', '.join(invalid_urls[:3])}"
      
        if video_ids:
            # hapus data lama
            cursor.execute("DELETE FROM comments") 
            db.commit() 
            
            # Scraping komentar dari semua video
            all_comments, errors = get_comments_from_multiple_videos(video_ids, max_per_video=100)

            # simpan ke database
            for c in all_comments:
                cursor.execute("""
                    INSERT INTO comments (video_id, waktu, komentar, likes)
                    VALUES (%s, %s, %s, %s)
                """, (c["video_id"],c["waktu"], c["komentar"], c["likes"]))

            db.commit()
 
            if errors:
                gagal_msg = f"Gagal mengambil komentar dari {len(errors)} video: {', '.join(errors)}"
                pesan_error = (pesan_error + " | " + gagal_msg) if pesan_error else gagal_msg
            
    # ambil data dari database
    cursor.execute("SELECT COUNT(*) FROM comments")
    total = cursor.fetchone()[0]
    
    start = (page - 1) * per_page
    
    cursor.execute("""
        SELECT video_id, waktu, komentar, likes
        FROM comments
        ORDER BY waktu DESC
        LIMIT %s OFFSET %s
    """, (per_page, start))
    
    rows = cursor.fetchall()
    
    data_komentar = []
    for r in rows:
        data_komentar.append({
            "video_id": r[0],
            "waktu": r[1],
            "komentar": r[2],
            "likes": r[3]
        })
        
    total_pages = (total // per_page) + (1 if total % per_page > 0 else 0)
    
    # Hitung jumlah video unik di DB untuk info
    cursor.execute("SELECT COUNT(DISTINCT video_id) FROM comments")
    total_videos = cursor.fetchone()[0]

    return render_template(
        "pages/uploadLink.html",
        data=data_komentar,
        page=page,
        total_pages=total_pages,
        total_videos=total_videos,
        pesan_error=pesan_error
    )


# Route Hasil Analisis
@app.route("/hasil")
def hasil():
    
    # ambil mode (umum/spesifik)
    mode = request.args.get("mode", "umum")
    video_filter = request.args.get("video_id", None)

    # ambil data dari database
    if video_filter:
        cursor.execute("""
            SELECT video_id, waktu, komentar, likes 
            FROM comments 
            WHERE video_id = %s
            LIMIT 500""", (video_filter,))
    else:
        cursor.execute("SELECT video_id, waktu, komentar, likes FROM comments LIMIT 500")
    rows = cursor.fetchall()

    data_komentar = []
    for r in rows:
        data_komentar.append({
            "video_id": r[0],
            "waktu": r[1],
            "komentar": r[2],
            "likes": r[3]
        })
    
    # Filter jika mode spesifik
    if mode == "topik":
        keywords = ["sawit", "kelapa sawit", "banjir", "perkebunan"]
    
        data_komentar = [
            c for c in data_komentar
            if any(k in c["komentar"].lower() for k in keywords)
        ]

    # hitung emosi
    data_emosi = {}
    
    # emosi per video utk perbandingan
    emosi_per_video = {}

    for c in data_komentar:
        teks_asli = c["komentar"]
        teks_pre = preprocessing(teks_asli) 
        emosi = prediksi_emosi(teks_asli)

        c["preprocessing"] = teks_pre  
        c["emosi"] = emosi
        c["preprocessing_highlight"] = highlight_by_emotion(
            teks_pre,
            kamus_nrc,
            emosi
        )

        data_emosi[emosi] = data_emosi.get(emosi, 0) + 1
        
        vid = c.get("video_id", "unknown")
        if vid not in emosi_per_video:
            emosi_per_video[vid] = {}
        emosi_per_video[vid][emosi] = emosi_per_video[vid].get(emosi,0) + 1

    # handle kalau data kosong
    if not data_emosi:
        data_emosi = {}
        total_data = 0
        emosi_dominan = "-"
        emosi_terendah = "-"
        labels = []
        values = []
    else:
        # hitung total
        total_data = sum(data_emosi.values())

        # cari emosi dominan dan terendah
        emosi_dominan = max(data_emosi, key=data_emosi.get)
        emosi_terendah = min(data_emosi, key=data_emosi.get)

        labels = list(data_emosi.keys())
        values = list(data_emosi.values())
    
    # Daftar video unik untuk filter dropdown
    #cursor.execute("SELECT DISTINCT video_id FROM comments")
    #daftar_video = [r[0] for r in cursor.fetchall()]
    
    return render_template(
        "pages/hasilAnalisis.html",
        total_data=total_data,
        emosi_dominan=emosi_dominan,
        emosi_terendah=emosi_terendah,
        data_emosi=data_emosi,
        labels=labels,
        values=values,
        data_komentar=data_komentar,
        mode=mode,
        emosi_per_video=emosi_per_video,
        #daftar_video=daftar_video,
        video_filter=video_filter
    )


# Route Kamus Emosi
@app.route("/kamus", methods=["GET", "POST"])
def kamus():
    hasil = None
    kata_input = ""

    if request.method == "POST":
        kata_input = request.form.get("kata", "").lower().strip()
        
        # kata_bersih = preprocessing(kata_input)
        tokens_asli = kata_input.lower().split()
        tokens_pre = preprocessing(kata_input).split()
        
        emosi_list = []
        
        if kata_input in kamus_nrc:
            emosi_list.extend(kamus_nrc[kata_input])
        else:
        
            for kata in tokens_asli:
                if kata in kamus_nrc:
                    emosi_list.extend(kamus_nrc[kata])
                    
            if not emosi_list:
                for kata in tokens_pre:
                    if kata in kamus_nrc:
                        emosi_list.extend(kamus_nrc[kata])
        
        if isinstance(emosi_list, list) and len(emosi_list):
            hasil = ", ".join(set(emosi_list))
        else:
            hasil = "Netral"

    return render_template("pages/kamusEmosi.html", hasil=hasil, kata_input=kata_input)


# Route auto complete
@app.route("/autocomplete")
def autocomplete():
    keyword = request.args.get("q", "").lower()

    cursor.execute("SELECT komentar FROM comments LIMIT 500")
    rows = cursor.fetchall()

    words = set()

    for r in rows:
        komentar = r[0].lower().split()

        for w in komentar:
            if keyword in w and len(w) > 3:
                words.add(w)

    hasil = list(words)[:10]  # batasi 10 rekomendasi

    return {"data": hasil}


# Run server
if __name__ == "__main__":
    app.run(debug=True)
