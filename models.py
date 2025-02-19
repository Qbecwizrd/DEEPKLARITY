from sqlalchemy import Column, Integer, String, ARRAY, DateTime, Text, Float
from database import Base
from datetime import datetime

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)  # Some resumes might not have a name
    email = Column(String, unique=True, nullable=True)  # Allow nullable for missing emails
    phone = Column(String, nullable=True)
    core_skills = Column(ARRAY(Text), nullable=True)  # Use ARRAY for structured data
    soft_skills = Column(ARRAY(Text), nullable=True)  # Use ARRAY for structured data
    work_experience = Column(ARRAY(Text), nullable=True)  # Store work experiences
    resume_rating = Column(String, nullable=True)          # resume_rating
    improvement_areas = Column(ARRAY(Text), nullable=True)  # Use Text for longer content
    upskill_suggestions = Column(ARRAY(Text), nullable=True)
    career_success_score = Column(Integer, nullable=True)  # Store the score
    positive_words = Column(Integer, nullable=True)  # Count of positive words
    negative_words = Column(Integer, nullable=True)  # Count of negative words
    neutral_words = Column(Integer, nullable=True)  # Count of neutral words
    filename = Column(String, nullable=False)  # Store the original file name
    created_at = Column(DateTime, default=datetime.utcnow)

