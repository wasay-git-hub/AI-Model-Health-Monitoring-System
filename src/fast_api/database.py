import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# Setup Connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy 2.0 Base
class Base(DeclarativeBase):
    pass

# Table Model
class ModelHealthLog(Base):
    __tablename__ = "model_health_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Use timezone-aware UTC for modern standards
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Increased lengths to prevent "StringDataRightTruncation" errors
    model_type = Column(String(100)) 
    endpoint = Column(String(100))
    dataset_source = Column(String(500))
    
    row_count = Column(Integer)
    rmspe = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    latency_ms = Column(Float)

def init_db():
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables initialized successfully in Supabase!")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    init_db()