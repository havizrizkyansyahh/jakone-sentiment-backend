from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_db
from models import Admin

import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jakone_super_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

class UpdateProfileRequest(BaseModel):
    nama_lengkap: str
    old_password: Optional[str] = None
    new_password: Optional[str] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    admin = db.query(Admin).filter(Admin.username == username).first()
    if admin is None:
        raise HTTPException(status_code=401, detail="Admin not found")
    return admin

@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    admin = db.query(Admin).filter(Admin.username == request.username).first()
    if not admin or not verify_password(request.password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(data={"sub": admin.username})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "admin": {
            "username": admin.username,
            "nama_lengkap": admin.nama_lengkap
        }
    }

@router.put("/update-profile")
def update_profile(
    request: UpdateProfileRequest, 
    current_admin: Admin = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    current_admin.nama_lengkap = request.nama_lengkap
    
    if request.old_password and request.new_password:
        if not verify_password(request.old_password, current_admin.password_hash):
            raise HTTPException(status_code=400, detail="Incorrect old password")
        current_admin.password_hash = get_password_hash(request.new_password)
    
    db.commit()
    db.refresh(current_admin)
    
    return {
        "message": "Profile updated successfully",
        "admin": {
            "username": current_admin.username,
            "nama_lengkap": current_admin.nama_lengkap
        }
    }
