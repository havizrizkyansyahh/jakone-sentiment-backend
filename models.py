from sqlalchemy import Column, Integer, Float, String, Text
from database import Base

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    tanggal = Column(String)  # Using String for 'at' column, simple to parse later if needed
    rating = Column(Integer)  # 1 to 5
    teks_asli = Column(Text)
    teks_bersih = Column(Text)
    teks_final = Column(Text)
    label_sentimen = Column(String)  # Positif / Negatif
    prediksi_mesin = Column(String, nullable=True)  # Filled by SVM prediction later
    skor_kepercayaan = Column(Float, nullable=True)  # Confidence score from SVM

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    nama_lengkap = Column(String)
