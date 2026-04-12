import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Creating the Engine
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Initializing the Base Class
Base = declarative_base()

# Making the session
Session = sessionmaker(autoflush=False, autocommit=False, bind=engine)

class ModelHealthLog(Base):
    __tablename__ = "model_health_logs"

    log_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    model_type = Column(String(20)) 

    endpoint = Column(String(30)) # which API endpoint was called
    dataset_source = Column(String(30)) # name of the csv file evaluated
    row_count = Column(Integer) # batch size of the data being evaluated

    rmspe = Column(Float) # primary metric
    mae = Column(Float)
    mape = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)

# Syncronizing python code with cloud
def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized in the cloud.")

if __name__ == "__main__":
    init_db()