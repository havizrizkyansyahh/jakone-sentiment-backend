from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager
import joblib
import os
import logging
from database import engine
from models import Base
from auth import router as auth_router, get_current_admin
from api import router as api_router
from fastapi import Depends
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to store models
ml_models = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model_v2.pkl")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_model_v2.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load models on startup and clean up on shutdown.
    """
    # Auto-create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")

    try:
        # Load SVM Model
        if os.path.exists(SVM_MODEL_PATH):
            ml_models["svm_model"] = joblib.load(SVM_MODEL_PATH)
            logger.info("SVM model loaded successfully.")
        else:
            logger.warning(f"SVM model not found at {SVM_MODEL_PATH}. Please ensure the file exists.")
            ml_models["svm_model"] = None

        # Load TF-IDF Model
        if os.path.exists(TFIDF_MODEL_PATH):
            ml_models["tfidf_model"] = joblib.load(TFIDF_MODEL_PATH)
            logger.info("TF-IDF model loaded successfully.")
        else:
            logger.warning(f"TF-IDF model not found at {TFIDF_MODEL_PATH}. Please ensure the file exists.")
            ml_models["tfidf_model"] = None
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        ml_models["svm_model"] = None
        ml_models["tfidf_model"] = None
        
    yield # Application handles requests here
    
    # Clean up ML models on shutdown
    ml_models.clear()
    logger.info("Models unloaded and resources cleared.")

app = FastAPI(title="JakOne Review Analysis API", lifespan=lifespan)

# Konfigurasi CORS — reads from ALLOWED_ORIGINS env var (comma-separated)
# Example: ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5173
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Authentication Endpoints
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

# Include the core API endpoints and protect them with get_current_admin
app.include_router(api_router, dependencies=[Depends(get_current_admin)])

@app.get("/api/health")
def health_check():
    """
    Health check endpoint to verify the server is running and models are loaded.
    """
    return {
        "status": "healthy",
        "models_loaded": {
            "svm_model": ml_models.get("svm_model") is not None,
            "tfidf_model": ml_models.get("tfidf_model") is not None
        }
    }

# ─── Render / Production Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
