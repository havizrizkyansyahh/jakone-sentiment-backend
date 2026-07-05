import pandas as pd
from database import SessionLocal, engine
from models import Review, Base
import os

# Ensure tables are created just in case
Base.metadata.create_all(bind=engine)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_jakone_ready.csv")

def seed_data():
    db = SessionLocal()
    try:
        # Check if table already has data to avoid duplicates
        existing_count = db.query(Review).count()
        if existing_count > 0:
            print(f"Database already contains {existing_count} records. Seeding skipped to avoid duplicates.")
            return

        if not os.path.exists(DATA_PATH):
            print(f"Data file not found at {DATA_PATH}. Please ensure the file exists.")
            return

        print(f"Reading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

        # Handle empty/NaN values gracefully (convert them to None)
        df = df.where(pd.notnull(df), None)

        reviews_to_insert = []
        for index, row in df.iterrows():
            review = Review(
                tanggal=str(row.get('at', '')) if row.get('at') else None,
                rating=int(row.get('score', 0)) if pd.notna(row.get('score')) and row.get('score') is not None else None,
                teks_asli=str(row.get('content', '')) if row.get('content') else None,
                teks_bersih=str(row.get('content_bersih', '')) if row.get('content_bersih') else None,
                teks_final=str(row.get('content_final', '')) if row.get('content_final') else None,
                label_sentimen=str(row.get('label_sentimen', '')) if row.get('label_sentimen') else None,
                prediksi_mesin=None  # Explicitly setting this to None initially
            )
            reviews_to_insert.append(review)

        print(f"Inserting {len(reviews_to_insert)} records into the database...")
        db.bulk_save_objects(reviews_to_insert)
        db.commit()
        print("Database seeded successfully!")

    except Exception as e:
        print(f"Error during seeding: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
