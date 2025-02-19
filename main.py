from database import engine
import models
from fastapi import FastAPI, File, UploadFile, Request,Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from fastapi.staticfiles import StaticFiles
from models import Resume
from fastapi.responses import HTMLResponse
import shutil
import os
import httpx  # Use httpx for async requests
from resume_parser import parse_resume
from pathlib import Path
from LLM import llm_response
from fastapi.templating import Jinja2Templates


app = FastAPI()

# Mount static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static") 


# Create database tables automatically
models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

API_URL = "http://127.0.0.1:8000/process"  # Endpoint for processing

@app.get("/")
async def serve_upload_page(request: Request):
    return templates.TemplateResponse("uploads.html", {"request": request})



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



from fastapi.responses import JSONResponse

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_path = UPLOAD_DIR / file.filename

        # Save the uploaded file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        parsed_data = parse_resume(file_path)

        response_data = {
            "filename": file.filename,
            "content_type": file.content_type,
        }
        text_content=parsed_data['text']
        #print(text_content)
        llm_response_text = llm_response(text_content)  # Add 'await' if it's async
        parsed_data['resume_rating']=llm_response_text['resume_rating']
        parsed_data['improvement_areas']=llm_response_text['improvement_areas']
        parsed_data['upskill_suggestions']=llm_response_text['upskill_suggestions']
        data_combine = {
            **response_data,
            **parsed_data     
        }
        # Store data in PostgreSQL
        new_resume = Resume(
            name=data_combine.get("name"),
            email=data_combine.get("email"),
            phone=data_combine.get("phone"),
            core_skills=data_combine.get("core_skills"),
            soft_skills=data_combine.get("soft_skills"),
            work_experience=data_combine.get("work_experience"),
            resume_rating=str(data_combine.get("resume_rating")),
            improvement_areas=data_combine.get("improvement_areas"),
            upskill_suggestions=data_combine.get("upskill_suggestions"),
            career_success_score=data_combine.get("Career Success Score"),
            positive_words=data_combine.get("Positive Words"),
            negative_words=data_combine.get("Negative Words"),
            neutral_words=data_combine.get("Neutral Words"),
            filename=file.filename  # Store the original file name
        )
        #  Check if email already exists in the database
        existing_resume = db.query(Resume).filter(Resume.email == data_combine.get("email")).first()

        if existing_resume:
            return JSONResponse(
                content={"error": " Email already exists. Use a different email or update the existing resume."},
                status_code=400,)
        db.add(new_resume)
        db.commit()
        db.refresh(new_resume)  
        response_data_combine = {
            "message": "Resume uploaded and stored successfully!",
            "resume_id": new_resume.id,  # Return stored ID
            "filename": file.filename,
            **data_combine
        }

        return JSONResponse(content=response_data_combine, status_code=200)

    except Exception as e:
        print(" Error:", str(e))  # Log errors if any
        return JSONResponse(content={"error": str(e)}, status_code=500)







# Route to display resume history
@app.get("/history")
async def get_history(request: Request, db: Session = Depends(get_db)):
    resumes = db.query(Resume).all()  # Fetch all resumes from DB
    return templates.TemplateResponse("history.html", {"request": request, "resumes": resumes})


import psycopg2
from typing import Dict
# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="resumedb",
    user="postgres",
    password="12345",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

@app.get("/resume_details/{resume_id}")
def get_resume_details(resume_id: int) -> Dict:
    cursor.execute("SELECT * FROM resumes WHERE id = %s", (resume_id,))
    resume = cursor.fetchone()

    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    # Convert the result into JSON format
    return {
        "id": resume[0],
        "filename": resume[1],
        "name": resume[2],
        "email": resume[3],
        "phone": resume[4],
        "core_skills": resume[5],
        "soft_skills": resume[6],
        "work_experience": resume[7],
        "resume_rating": resume[8],
        "improvement_areas": resume[9],
        "upskill_suggestions": resume[10],
        "career_success_score": resume[11],
        "positive_words": resume[12],
        "negative_words": resume[13],
        "neutral_words": resume[14],
        "created_at": resume[15],
    }
