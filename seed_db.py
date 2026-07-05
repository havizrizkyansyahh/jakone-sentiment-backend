import pandas as pd
import os
from database import SessionLocal, engine
from models import Base, Review

def seed_database():
    # 1. Initialize database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # 2. Connect to the database
    db = SessionLocal()
    
    try:
        # 3. Clear existing rows to avoid duplication
        print("Clearing existing records in the 'reviews' table...")
        deleted_count = db.query(Review).delete()
        db.commit()
        print(f"Deleted {deleted_count} old records.")
        
        # 4. Read the V2 CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "data", "dataset_jakone_ready_v2.csv")
        print(f"Reading dataset from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"ERROR: File not found at {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        print(f"Found {total_rows} rows in CSV. Starting insertion...")
        
        # 5. Iterate through rows and insert
        batch_size = 1000
        reviews_to_insert = []
        
        for index, row in df.iterrows():
            review = Review(
                tanggal=str(row['at']) if pd.notna(row['at']) else None,
                rating=int(row['score']) if pd.notna(row['score']) else None,
                teks_asli=str(row['content']) if pd.notna(row['content']) else None,
                teks_bersih=str(row['content_bersih']) if pd.notna(row['content_bersih']) else None,
                teks_final=str(row['content_final']) if pd.notna(row['content_final']) else None,
                label_sentimen=str(row['label_sentimen']) if pd.notna(row['label_sentimen']) else None,
                prediksi_mesin=None  # Leave as NULL for the dashboard UI to predict
            )
            reviews_to_insert.append(review)
            
            # Batch commit to speed up insertion
            if len(reviews_to_insert) >= batch_size:
                db.bulk_save_objects(reviews_to_insert)
                db.commit()
                print(f"Inserted {index + 1} / {total_rows} records...")
                reviews_to_insert = []
                
        # Insert any remaining records
        if reviews_to_insert:
            db.bulk_save_objects(reviews_to_insert)
            db.commit()
            print(f"Inserted {total_rows} / {total_rows} records...")
            
        print("Database seeded successfully with V2 data!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
