import sys
import getpass
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Admin, Base
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def main():
    print("=== JakOne Admin Creator ===")
    username = input("Username: ").strip()
    if not username:
        print("Error: Username cannot be empty.")
        return

    nama_lengkap = input("Nama Lengkap: ").strip()
    if not nama_lengkap:
        print("Error: Nama Lengkap cannot be empty.")
        return

    password = getpass.getpass("Password: ")
    if not password:
        print("Error: Password cannot be empty.")
        return
    
    password_confirm = getpass.getpass("Confirm Password: ")
    if password != password_confirm:
        print("Error: Passwords do not match.")
        return

    # Ensure all tables exist in the PostgreSQL database before proceeding
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        existing_admin = db.query(Admin).filter(Admin.username == username).first()
        if existing_admin:
            print(f"Error: Admin with username '{username}' already exists.")
            return

        hashed_pw = get_password_hash(password)
        new_admin = Admin(
            username=username,
            password_hash=hashed_pw,
            nama_lengkap=nama_lengkap
        )
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        print(f"Success: Admin '{username}' (ID: {new_admin.id}) created successfully!")
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
