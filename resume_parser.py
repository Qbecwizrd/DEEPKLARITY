import spacy
import re
from PyPDF2 import PdfReader
from docx import Document

##libraries for llm implementation
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load spaCy English NLP model
nlp = spacy.load("en_core_web_sm")
text="Mohammed Abdul Jabbar Khan Email : abduljabbarkhanh0@gmail.com\nhttp://www.officialjabbi.com Mobile : +91 7093493924\n/githubQbecwizrd /linkedinabdul-jabbar-khan-74886b249\nEducation\n•Gokaraju Rangaraju Institute of Engineering and Technology Hyderabad\nComputer Science and Engineering(Artificial Intelligence and Machine Learning) CGPA:8.77 2021 - 2025\nExperience\n•Smart interviews Hyderabad\nStudent Mentor Mar 2024 - Present\n◦Mentored 200+ students in DSA, helping them build strong problem-solving skills, leading to a 40\npercent boost in challenges solved and a 25percent improvement in algorithm understanding within set\ntimeframes.\nProjects\n◦Sentiment Analysis with RNN : Developed a Recurrent Neural Network for sentiment analysis on IMDb\nmovie reviews using TensorFlow andKeras . Deployed on a Streamlit app, providing real-time positive/negative\nsentiment predictions. Optimized data preprocessing and model performance to deliver accurate classification.\n◦Deep Learning-based LSTM Model for Next-Word Prediction : Engineered and Deployed Next Word\nPrediction Model using LSTM withStreamlit ,TensorFlow andKeras . Trained on Shakespeare’s Hamlet data .\nIntegrated early stopping andDropout for regularization.\n◦Corpus Evaluation System : Devised a Python-based system for extracting, cleaning, and analyzing articles using\nBeautifulSoup, pandas, and NLTK . Performed sentiment analysis, readability scoring (Fog Index, complexity, and\nword length), and extracted linguistic features like word counts.\n◦Contextual Chat History Retrieval System with RAG : Developed a Chat History-Aware Retrieval System\nusing LangChain, Hugging Face Embeddings, and Chroma . Integrated RAG and document retrieval for\naccurate responses, utilizing web scraping and dynamic chat history processing.\n◦AI-Powered PDF Chat Assistant with RAG : Developed a Conversational Retrieval-Augmented Generation\nsystem using Streamlit for PDF document uploads and chat history management. Integrated Groq API for LLM\ninteractions, Hugging Face embeddings for semantic search, and Chroma for vector store management to\nfacilitate real-time, context-driven information retrieval and interaction.\nProgramming Skills\n◦Languages : Python, SQL, Java, R-programming, CSS .\n◦Technologies : Pandas, NumPy, Scikit learn, Natural Language Processing, Docker, Data Analysis, Feature\nEngineering, Flask(RESTful API), HuggingFace, Groq.\n◦Miscellaneous : Detail-oriented, Mentoring, Teamwork, customer-focused, Engaging Presentation\nAchievements\n2023- Secured 5th,11th ,33rd Rank in Coding contests conducted by Smart Interviews at GRIET\n2022- Secured 2nd Position at a football tournament held at VNRVJIT\n2023- Completed Supervised Machine Learning from Stanford DeepLearning AI.\n2023- Capped off a BootCamp at IDS Inc on BlockChain HyperLedger\n2023- Part of Research Apprenticeship at Centella Scientific\n2024- Accomplished a Data Science Job Simulation at BRITISH AIRWAYS"


# Predefined skill categories
TECH_SKILLS = {
    "python", "java", "c", "c++", "c#", "go", "ruby", "swift", "kotlin", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "oracledb", "redis", "cassandra", "graphql", "dynamodb",
    "machine learning", "deep learning", "artificial intelligence", "computer vision", "reinforcement learning", "nlp",
    "data science", "big data", "data analytics", "data engineering", "etl", "data warehousing", "hadoop", "spark", "kafka",
    "aws", "azure", "google cloud", "cloud computing", "devops", "kubernetes", "terraform", "jenkins", "ansible", "ci/cd",
    "git", "github", "gitlab", "tensorflow", "pytorch", "scikit-learn", "keras", "opencv", "nltk", "hugging face", "llamaindex",
    "docker", "flask", "fastapi", "django", "spring boot", "react", "angular", "vue.js", "node.js", "express.js",
    "cybersecurity", "blockchain", "cryptography", "edge computing", "quantum computing", "microservices", "serverless computing"
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem-solving", "adaptability",
    "time management", "creativity", "collaboration", "critical thinking", "emotional intelligence",
    "decision-making", "conflict resolution", "empathy", "active listening", "public speaking",
    "interpersonal skills", "negotiation", "resilience", "work ethic", "self-motivation",
    "attention to detail", "flexibility", "self-awareness", "accountability", "patience",
    "diversity awareness", "cultural intelligence", "growth mindset", "coaching", "mentoring",
    "delegation", "persuasion", "innovation", "stress management", "multitasking",
    "organizational skills", "professionalism", "assertiveness", "networking",
    "decision-making under pressure", "self-confidence", "emotional regulation", "customer service",
    "storytelling", "agility", "positive attitude", "body language", "social intelligence", "integrity",
    "ethical decision-making", "mindfulness", "open-mindedness", "constructive criticism", "situational awareness"
}



# Step 1: Master Dictionary
MASTER_DICT = {
    "positive": {
        "achieved": 3, "led": 3, "developed": 3, "optimized": 3, "innovative": 3,
        "accomplished": 3, "managed": 3, "improved": 3, "increased": 3, "enhanced": 3,
        "delivered": 3, "successfully": 3, "initiated": 3, "launched": 3, "spearheaded": 3,
        "mentored": 3, "trained": 3, "designed": 3, "created": 3, "built": 3,
        "formulated": 3, "streamlined": 3, "engineered": 3, "executed": 3, "collaborated": 3,
        "resolved": 3, "pioneered": 3, "innovated": 3, "inspired": 3, "delivered": 3,
        "strategized": 3, "organized": 3, "outperformed": 3, "earned": 3, "recognized": 3
    },
    "negative": {
        "struggled": -2, "difficulties": -2, "failed": -3, "mistakes": -2, "inefficient": -2,
        "delayed": -2, "declined": -2, "poor": -2, "missed": -2, "rejected": -3,
        "incomplete": -2, "error": -2, "insufficient": -2, "weak": -2, "lost": -3,
        "downgraded": -2, "shortcomings": -2, "problematic": -2, "conflicted": -2, "criticized": -3,
        "terminated": -3, "unsuccessful": -3, "mismanaged": -3, "ineffective": -2, "underperformed": -3
    },
    "neutral": {
        "worked": 1, "assisted": 1, "helped": 1, "contributed": 1, "participated": 1,
        "handled": 1, "performed": 1, "supported": 1, "engaged": 1, "reviewed": 1,
        "researched": 1, "analyzed": 1, "documented": 1, "communicated": 1, "processed": 1,
        "collaborated": 1, "tested": 1, "evaluated": 1, "monitored": 1, "examined": 1
    }
}


# Step 2: Function to Tokenize Without NLTK
def tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split by spaces
    return words

# Step 3: Scoring Function
def calculate_resume_score(resume_text):
    tokens = tokenize(resume_text)
    score = 0
    word_count = {"positive": 0, "negative": 0 , 'neutral':0}

    for word in tokens:
        for category, words in MASTER_DICT.items():
            if word in words:
                score += words[word]  # Add word score
                word_count[category] += 1

    return {"Career Success Score": score, "Positive Words": word_count["positive"], "Negative Words": word_count["negative"],'Neutral Words':word_count['neutral']}


def extract_work_experience(text):
    """Extract work experience from the resume text."""
    experience_pattern = r"(?i)(\d+ years?|\d+ months?|experience in .*?)(?=\.)"
    matches = re.findall(experience_pattern, text)
    
    return matches if matches else None


def parse_pdf(file_path: str):
    """Parse the content of a PDF file and return the extracted text."""
    try:
        # Read the PDF file
        reader = PdfReader(file_path)
        text = ""
        
        # Iterate over all pages and extract text
        for page in reader.pages: 
            text += page.extract_text()  # Extract text from each page

        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def parse_docx(file_path: str):
    """Parse the content of a DOCX file and return the extracted text."""
    try:
        # Open DOCX file
        doc = Document(file_path)
        text = ""
        
        # Iterate through all paragraphs and extract text
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def extract_resume_details(text):
    """Extract structured resume details using NLP."""
    
    doc = nlp(text)

    # Extract name (first PERSON entity found)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None)

    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else None

    # Extract phone number
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}', text)
    phone = phone_match.group(0) if phone_match else None

    work_experience = extract_work_experience(text)

    # Extract technical and soft skills
    extracted_skills1 = {token.text for token in doc if token.text.lower() in TECH_SKILLS}
    extracted_skills2 = {token.text for token in doc if token.text.lower() in SOFT_SKILLS}
    core_skills = list(extracted_skills1)
    soft_skills = list(extracted_skills2)

    score=calculate_resume_score(text)
    info={
        "name": name,
        "email": email,
        "phone": phone,
        "core_skills": core_skills,
        "soft_skills": soft_skills,
        "resume_rating": None,  # To be filled by LLaMA later
        "improvement_areas": None,  # To be filled by LLaMA later
        "upskill_suggestions": None , # To be filled by LLaMA later
        'text':text,
        'work_experience': work_experience

    }
    return {**info,**score}

import os

def parse_resume(file_path: str):
    """Determine file type based on extension and extract text accordingly."""
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None

    # Get the file extension (pdf or docx)
    file_extension = str(file_path).split('.')[-1].lower()

    # Handle PDF or DOCX based on file extension
    if file_extension == 'pdf':
        text = parse_pdf(file_path)
    elif file_extension == 'docx':
        text = parse_docx(file_path)
    else:
        print("Unsupported file type. Only PDF and DOCX are supported.")
        return None
    
    # If text is successfully extracted, proceed with further processing
    if text:
        return extract_resume_details(text) # Pass the extracted text for further processing
    else:
        print("Failed to extract text from the file.")
        return None

