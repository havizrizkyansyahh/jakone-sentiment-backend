from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Connection string to PostgreSQL (reads from DATABASE_URL env var, falls back to local)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/db_skripsi_jakone")

# Render/Supabase may provide postgres:// instead of postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine with pool_pre_ping for production connection resilience
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create a sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base for our models
Base = declarative_base()

# Dependency to get the DB session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
