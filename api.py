from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import Review
import re
import io
import csv
import math
from collections import Counter
from typing import Optional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import pandas as pd

router = APIRouter(prefix="/api")

class PredictRequest(BaseModel):
    text: str

# ─── Smart Preprocessing Globals ───
factory = StemmerFactory()
stemmer = factory.create_stemmer()

kata_penting = {'tidak', 'bukan', 'belum', 'jangan', 'kurang', 'enggak', 'ga', 'gak', 'masalah', 'kendala', 'gagal', 'buruk', 'jelek'}
list_stopwords = set(stopwords.words('indonesian'))
custom_stopwords = set([kata for kata in list_stopwords if kata not in kata_penting])
custom_stopwords.update(['jakone', 'mobile', 'aplikasi', 'app', 'bank', 'dki', 'nya', 'sih', 'deh', 'dong', 'kok'])

slang_dict = {
    "tdk": "tidak", "gk": "tidak", "ga": "tidak", "gak": "tidak", "engga": "tidak",
    "apk": "aplikasi", "bgs": "bagus", "kureng": "kurang", "lemot": "lambat", 
    "gabisa": "tidak bisa", "gagal": "gagal", "bgt": "banget", "skrg": "sekarang",
    "gajelas": "tidak jelas"
}

whitelist_words = {'login', 'error', 'bug', 'top', 'up', 'ewallet', 'loading', 'cs', 'qris', 'gopay', 'ovo', 'dana', 'shopeepay'}

def preprocess_text(text: str) -> str:
    """Smart preprocessing: lowercase, regex, slang normalization, stopword removal, protected stemming."""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Regex: Keep only a-z and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Slang normalization & Stopwords & Stemming
    words = text.split()
    processed_words = []
    
    for word in words:
        # Normalization
        norm_word = slang_dict.get(word, word)
        
        # Stopword removal
        if norm_word not in custom_stopwords:
            # Stemming, protected by whitelist
            if norm_word in whitelist_words:
                processed_words.append(norm_word)
            else:
                stemmed = stemmer.stem(norm_word)
                if stemmed:
                    processed_words.append(stemmed)
                    
    return " ".join(processed_words).strip()

@router.post("/reviews/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check required columns
        required_cols = ['at', 'score', 'content']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail="CSV must contain 'at', 'score', and 'content' columns.")
        
        reviews_to_insert = []
        for _, row in df.iterrows():
            review = Review(
                tanggal=str(row['at']) if 'at' in row and pd.notna(row['at']) else None,
                rating=int(row['score']) if 'score' in row and pd.notna(row['score']) else None,
                teks_asli=str(row['content']) if 'content' in row and pd.notna(row['content']) else None,
                teks_bersih=str(row['content_bersih']) if 'content_bersih' in row and pd.notna(row['content_bersih']) else None,
                teks_final=str(row['content_final']) if 'content_final' in row and pd.notna(row['content_final']) else None,
                label_sentimen=str(row['label_sentimen']) if 'label_sentimen' in row and pd.notna(row['label_sentimen']) else None,
                prediksi_mesin=None
            )
            reviews_to_insert.append(review)
            
        db.bulk_save_objects(reviews_to_insert)
        db.commit()
        return {"message": f"Berhasil mengunggah {len(reviews_to_insert)} baris dataset."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal memproses file: {str(e)}")

@router.delete("/reviews/clear-all")
def clear_all_reviews(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(Review).delete()
        db.commit()
        return {"message": f"Berhasil menghapus {deleted_count} baris dataset."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reviews")
def get_reviews(
    page: int = 1, 
    size: int = 50, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = "tanggal",
    sort_order: Optional[str] = "desc",
    db: Session = Depends(get_db)
):
    """Fetch reviews with pagination, optional date filtering, and sorting."""
    if page < 1: page = 1
    if size < 1: size = 50
        
    query = db.query(Review)
    
    if start_date:
        query = query.filter(Review.tanggal >= start_date)
    if end_date:
        query = query.filter(Review.tanggal <= f"{end_date} 23:59:59")

    # Dynamic sorting
    allowed_sort_columns = {
        "tanggal": Review.tanggal,
        "rating": Review.rating,
        "label_sentimen": Review.label_sentimen,
        "prediksi_mesin": Review.prediksi_mesin,
    }
    sort_column = allowed_sort_columns.get(sort_by, Review.tanggal)
    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    total_count = query.count()
    offset = (page - 1) * size
    reviews = query.offset(offset).limit(size).all()
    
    return {
        "data": reviews,
        "total_count": total_count,
        "page": page,
        "size": size
    }

@router.get("/dashboard-stats")
def get_dashboard_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Return summary statistics with optional date filtering."""
    query = db.query(Review)
    
    if start_date:
        query = query.filter(Review.tanggal >= start_date)
    if end_date:
        query = query.filter(Review.tanggal <= f"{end_date} 23:59:59")

    total_reviews = query.count()
    
    # Existing metrics
    total_positive = query.filter(func.lower(Review.label_sentimen) == "positif").count()
    total_negative = query.filter(func.lower(Review.label_sentimen) == "negatif").count()
    total_predicted = query.filter(Review.prediksi_mesin.isnot(None)).count()
    
    # Data for Grouped Bar Chart (Asli vs Prediksi)
    pred_pos = query.filter(func.lower(Review.prediksi_mesin) == "positif").count()
    pred_neg = query.filter(func.lower(Review.prediksi_mesin) == "negatif").count()
    
    bar_data = [
        {"name": "Positif", "Asli": total_positive, "Prediksi": pred_pos},
        {"name": "Negatif", "Asli": total_negative, "Prediksi": pred_neg},
    ]
    
    # Data for Time-Series Line Chart (Trend)
    reviews_data = query.with_entities(Review.tanggal, Review.prediksi_mesin).all()
    trend = {}
    for r in reviews_data:
        if r.tanggal and r.prediksi_mesin:
            date_str = str(r.tanggal)[:7]  # Group by YYYY-MM
            if date_str not in trend:
                trend[date_str] = {"Positif": 0, "Negatif": 0}
            label = str(r.prediksi_mesin).lower()
            if label == "positif":
                trend[date_str]["Positif"] += 1
            elif label == "negatif":
                trend[date_str]["Negatif"] += 1
                
    line_data = [{"date": k, "Positif": v["Positif"], "Negatif": v["Negatif"]} for k, v in sorted(trend.items())]
    
    return {
        "total_reviews": total_reviews,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "total_predicted": total_predicted,
        "bar_data": bar_data,
        "line_data": line_data
    }

@router.post("/predict-all")
def predict_all(db: Session = Depends(get_db)):
    """Run SVM prediction on all unpredicted reviews with confidence scores."""
    from main import ml_models
    
    svm_model = ml_models.get("svm_model")
    tfidf_model = ml_models.get("tfidf_model")
    
    if not svm_model or not tfidf_model:
        raise HTTPException(status_code=500, detail="ML Models are not loaded into memory.")
        
    reviews_to_predict = db.query(Review).filter(Review.prediksi_mesin.is_(None)).all()
    
    if not reviews_to_predict:
        return {"message": "Semua ulasan telah diprediksi.", "predicted_count": 0}
        
    try:
        texts = [r.teks_final if r.teks_final else "" for r in reviews_to_predict]
        X_tfidf = tfidf_model.transform(texts)
        predictions = svm_model.predict(X_tfidf)
        
        # Compute confidence scores
        confidences = []
        try:
            probas = svm_model.predict_proba(X_tfidf)
            for proba in probas:
                confidences.append(round(float(max(proba)) * 100, 2))
        except AttributeError:
            decision_vals = svm_model.decision_function(X_tfidf)
            for dv in decision_vals:
                if hasattr(dv, '__len__'):
                    dv = float(max(dv, key=abs))
                else:
                    dv = float(dv)
                sigmoid_val = 1.0 / (1.0 + math.exp(-abs(dv)))
                confidences.append(round(sigmoid_val * 100, 2))
        
        for review, pred, conf in zip(reviews_to_predict, predictions, confidences):
            review.prediksi_mesin = str(pred)
            review.skor_kepercayaan = conf
            
        db.commit()
        return {"message": "Prediksi berhasil.", "predicted_count": len(reviews_to_predict)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@router.post("/reset-predictions")
def reset_predictions(db: Session = Depends(get_db)):
    """Reset all predictions and confidence scores."""
    try:
        updated_count = db.query(Review).filter(Review.prediksi_mesin.isnot(None)).update(
            {Review.prediksi_mesin: None, Review.skor_kepercayaan: None}
        )
        db.commit()
        return {"message": "Semua prediksi berhasil dihapus.", "reset_count": updated_count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error during prediction reset: {str(e)}")

@router.get("/evaluation-metrics")
def get_evaluation_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Calculate real-time Accuracy, Precision, and Recall from the database."""
    try:
        # Only evaluate rows where both ground truth and prediction exist
        query = db.query(Review).filter(
            Review.label_sentimen.isnot(None),
            Review.prediksi_mesin.isnot(None)
        )

        if start_date:
            query = query.filter(Review.tanggal >= start_date)
        if end_date:
            query = query.filter(Review.tanggal <= f"{end_date} 23:59:59")

        reviews = query.all()
        total_evaluated = len(reviews)

        if total_evaluated == 0:
            return {
                "total_evaluated": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0
            }

        # Confusion matrix: "positif" is the positive class
        tp = tn = fp = fn = 0
        for r in reviews:
            actual = r.label_sentimen.strip().lower()
            predicted = r.prediksi_mesin.strip().lower()

            if actual == "positif" and predicted == "positif":
                tp += 1
            elif actual == "negatif" and predicted == "negatif":
                tn += 1
            elif actual == "negatif" and predicted == "positif":
                fp += 1
            elif actual == "positif" and predicted == "negatif":
                fn += 1

        accuracy = round(((tp + tn) / total_evaluated) * 100, 2) if total_evaluated > 0 else 0.0
        precision = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0.0
        recall = round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0.0

        return {
            "total_evaluated": total_evaluated,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

@router.post("/predict-single")
def predict_single(req: PredictRequest):
    """Predict sentiment for a single raw string with confidence score."""
    from main import ml_models
    
    svm_model = ml_models.get("svm_model")
    tfidf_model = ml_models.get("tfidf_model")
    
    if not svm_model or not tfidf_model:
        raise HTTPException(status_code=500, detail="ML Models are not loaded into memory.")
        
    try:
        cleaned_text = preprocess_text(req.text)
        if not cleaned_text:
            return {"label": "Unknown", "confidence": 0.0, "teks_final": "", "message": "Text is empty after preprocessing."}
            
        X_tfidf = tfidf_model.transform([cleaned_text])
        prediction = svm_model.predict(X_tfidf)[0]
        
        # --- Confidence Score Extraction ---
        confidence = 0.0
        try:
            # Attempt 1: Use predict_proba (available if SVM was trained with probability=True)
            proba = svm_model.predict_proba(X_tfidf)[0]
            confidence = round(float(max(proba)) * 100, 2)
        except AttributeError:
            # Fallback: Use decision_function + sigmoid mapping
            # decision_function returns the signed distance from the hyperplane.
            # We convert this to a confidence percentage in the range [50%, 100%].
            decision_val = svm_model.decision_function(X_tfidf)[0]
            # For multi-class (OvO/OvR), decision_val might be an array; take the max absolute value.
            if hasattr(decision_val, '__len__'):
                decision_val = float(max(decision_val, key=abs))
            else:
                decision_val = float(decision_val)
            
            # Sigmoid maps any real number to (0, 1). We use abs(decision_val) to ensure
            # the output represents confidence in the *chosen* class regardless of sign.
            sigmoid_val = 1.0 / (1.0 + math.exp(-abs(decision_val)))
            # Scale from (0.5, 1.0) sigmoid range to (50%, 100%) percentage
            confidence = round(sigmoid_val * 100, 2)
        
        # Determine the display label
        label = str(prediction).capitalize()
        
        return {
            "label": label,
            "confidence": confidence,
            "teks_final": cleaned_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during single prediction: {str(e)}")

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Predict sentiment for a batch of texts uploaded as a CSV file."""
    from main import ml_models

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Hanya file .csv yang diperbolehkan.")

    svm_model = ml_models.get("svm_model")
    tfidf_model = ml_models.get("tfidf_model")

    if not svm_model or not tfidf_model:
        raise HTTPException(status_code=500, detail="ML Models are not loaded into memory.")

    try:
        # Read CSV content
        contents = await file.read()
        try:
            decoded = contents.decode("utf-8")
        except UnicodeDecodeError:
            decoded = contents.decode("latin-1")

        reader_io = io.StringIO(decoded)

        # Sniff to detect if there's a header
        sample = reader_io.read(4096)
        reader_io.seek(0)
        sniffer = csv.Sniffer()
        has_header = True
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = True  # assume header if sniffer fails

        if has_header:
            reader = csv.DictReader(reader_io)
            fieldnames = reader.fieldnames or []

            # Find the text column: prefer 'teks_ulasan', then common alternatives, then first column
            text_col = None
            preferred_names = ['teks_ulasan', 'teks_asli', 'text', 'review', 'ulasan', 'content']
            lower_fields = {f.lower().strip(): f for f in fieldnames}
            for pref in preferred_names:
                if pref in lower_fields:
                    text_col = lower_fields[pref]
                    break
            if text_col is None and fieldnames:
                text_col = fieldnames[0]

            if text_col is None:
                raise HTTPException(status_code=400, detail="File CSV kosong atau tidak memiliki kolom.")

            raw_texts = []
            for i, row in enumerate(reader):
                if i >= 5000:
                    break
                val = row.get(text_col, "")
                if val and val.strip():
                    raw_texts.append(val.strip())
        else:
            # No header — read as plain rows, take first column
            reader = csv.reader(reader_io)
            raw_texts = []
            for i, row in enumerate(reader):
                if i >= 5000:
                    break
                if row and row[0].strip():
                    raw_texts.append(row[0].strip())

        if not raw_texts:
            raise HTTPException(status_code=400, detail="Tidak ada teks ditemukan dalam file CSV.")

        # Preprocess all texts
        cleaned_texts = [preprocess_text(t) for t in raw_texts]

        # Transform and predict
        X_tfidf = tfidf_model.transform(cleaned_texts)
        predictions = svm_model.predict(X_tfidf)

        # Extract confidence scores
        confidences = []
        try:
            probas = svm_model.predict_proba(X_tfidf)
            for proba in probas:
                confidences.append(round(float(max(proba)) * 100, 2))
        except AttributeError:
            decision_vals = svm_model.decision_function(X_tfidf)
            for dv in decision_vals:
                if hasattr(dv, '__len__'):
                    dv = float(max(dv, key=abs))
                else:
                    dv = float(dv)
                sigmoid_val = 1.0 / (1.0 + math.exp(-abs(dv)))
                confidences.append(round(sigmoid_val * 100, 2))

        # Build results
        results = []
        for orig, cleaned, pred, conf in zip(raw_texts, cleaned_texts, predictions, confidences):
            results.append({
                "teks_asli": orig,
                "teks_final": cleaned,
                "label": str(pred).capitalize(),
                "confidence": conf
            })

        return {"results": results, "total": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")

@router.get("/wordcloud")
def get_wordcloud(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Return top 100 word frequencies for the frontend word cloud."""
    try:
        query = db.query(Review.teks_final).filter(Review.teks_final.isnot(None))
        
        if start_date:
            query = query.filter(Review.tanggal >= start_date)
        if end_date:
            query = query.filter(Review.tanggal <= f"{end_date} 23:59:59")
        
        reviews = query.all()
        
        word_counter = Counter()
        for r in reviews:
            text = r[0]
            if text:
                words = text.split()
                word_counter.update(words)
                
        top_words = word_counter.most_common(100)
        formatted_data = [{"text": word, "value": count} for word, count in top_words]
        
        return formatted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating wordcloud: {str(e)}")

@router.get("/export-csv")
def export_csv(
    limit: str = "all",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export reviews to CSV with optional date filtering and row limit."""
    try:
        query = db.query(Review).order_by(Review.id.asc())
        
        if start_date:
            query = query.filter(Review.tanggal >= start_date)
        if end_date:
            query = query.filter(Review.tanggal <= f"{end_date} 23:59:59")
        
        if limit != "all":
            try:
                limit_int = int(limit)
                query = query.limit(limit_int)
            except ValueError:
                pass
        
        reviews = query.all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Tanggal", "Rating", "Teks Asli", "Teks Bersih", "Teks Final", "Label Sentimen", "Prediksi Mesin", "Skor Kepercayaan"])
        
        for r in reviews:
            writer.writerow([
                r.id,
                r.tanggal or "",
                r.rating or "",
                r.teks_asli or "",
                r.teks_bersih or "",
                r.teks_final or "",
                r.label_sentimen or "",
                r.prediksi_mesin or "",
                r.skor_kepercayaan if r.skor_kepercayaan is not None else ""
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=export_ulasan.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")
