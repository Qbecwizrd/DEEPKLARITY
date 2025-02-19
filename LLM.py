import os
from dotenv import load_dotenv
load_dotenv()
# from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq  # Using Groq instead of Ollama
import re

##lanchain tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"

text='"Mohammed Abdul Jabbar Khan Email : abduljabbarkhanh0@gmail.com\nhttp://www.officialjabbi.com Mobile : +91 7093493924\n/githubQbecwizrd /linkedinabdul-jabbar-khan-74886b249\nEducation\n•Gokaraju Rangaraju Institute of Engineering and Technology Hyderabad\nComputer Science and Engineering(Artificial Intelligence and Machine Learning) CGPA:8.77 2021 - 2025\nExperience\n•Smart interviews Hyderabad\nStudent Mentor Mar 2024 - Present\n◦Mentored 200+ students in DSA, helping them build strong problem-solving skills, leading to a 40\npercent boost in challenges solved and a 25percent improvement in algorithm understanding within set\ntimeframes.\nProjects\n◦Sentiment Analysis with RNN : Developed a Recurrent Neural Network for sentiment analysis on IMDb\nmovie reviews using TensorFlow andKeras . Deployed on a Streamlit app, providing real-time positive/negative\nsentiment predictions. Optimized data preprocessing and model performance to deliver accurate classification.\n◦Deep Learning-based LSTM Model for Next-Word Prediction : Engineered and Deployed Next Word\nPrediction Model using LSTM withStreamlit ,TensorFlow andKeras . Trained on Shakespeare’s Hamlet data .\nIntegrated early stopping andDropout for regularization.\n◦Corpus Evaluation System : Devised a Python-based system for extracting, cleaning, and analyzing articles using\nBeautifulSoup, pandas, and NLTK . Performed sentiment analysis, readability scoring (Fog Index, complexity, and\nword length), and extracted linguistic features like word counts.\n◦Contextual Chat History Retrieval System with RAG : Developed a Chat History-Aware Retrieval System\nusing LangChain, Hugging Face Embeddings, and Chroma . Integrated RAG and document retrieval for\naccurate responses, utilizing web scraping and dynamic chat history processing.\n◦AI-Powered PDF Chat Assistant with RAG : Developed a Conversational Retrieval-Augmented Generation\nsystem using Streamlit for PDF document uploads and chat history management. Integrated Groq API for LLM\ninteractions, Hugging Face embeddings for semantic search, and Chroma for vector store management to\nfacilitate real-time, context-driven information retrieval and interaction.\nProgramming Skills\n◦Languages : Python, SQL, Java, R-programming, CSS .\n◦Technologies : Pandas, NumPy, Scikit learn, Natural Language Processing, Docker, Data Analysis, Feature\nEngineering, Flask(RESTful API), HuggingFace, Groq.\n◦Miscellaneous : Detail-oriented, Mentoring, Teamwork, customer-focused, Engaging Presentation\nAchievements\n2023- Secured 5th,11th ,33rd Rank in Coding contests conducted by Smart Interviews at GRIET\n2022- Secured 2nd Position at a football tournament held at VNRVJIT\n2023- Completed Supervised Machine Learning from Stanford DeepLearning AI.\n2023- Capped off a BootCamp at IDS Inc on BlockChain HyperLedger\n2023- Part of Research Apprenticeship at Centella Scientific\n2024- Accomplished a Data Science Job Simulation at BRITISH AIRWAYS"'
##lanchain prompt
from langchain.prompts import ChatPromptTemplate
# def llm_response(text):
#     prompt = ChatPromptTemplate.from_messages([
#     ("system", "Hey, you are a helpful assistant. Please provide resume rating,improvement areas and upskill suggestions."),
#     ("user", "Question: {question}")
#     ])
#     ##call the ollama model
#     llm=OllamaLLM(model='phi:latest')
#     output_parser=StrOutputParser()
#     chain=prompt|llm|output_parser
#     input_text=text
#     return chain.invoke({"question":input_text})

def llm_response(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI resume reviewer. Please analyze the resume text and provide the following:\n"
                   "1. Resume Rating (out of 10) based on structure, clarity.\n"
                   "2. Improvement Areas (specific feedback on what needs to be improved).\n"
                   "3. Upskill Suggestions (skills, courses, or tools to enhance the resume)."
                   "Give all the output in 3 diff lines "),
        ("user", "Resume Text: {resume_text}")
    ])
    
    # Initialize LLM
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    output_parser = StrOutputParser()
    
    # Chain execution
    chain = prompt | llm | output_parser
    response_text = chain.invoke({"resume_text": text})

    # Split sections using regex patterns
    rating_match = re.search(r"Resume Rating:\s*(.+)", response_text)
    improvement_match = re.search(r"Improvement Areas:\s*([\s\S]*?)\n\n", response_text)
    upskill_match = re.search(r"Upskill Suggestions:\s*([\s\S]*)", response_text)

    resume_rating = rating_match.group(1).strip() if rating_match else "No rating provided"
    improvement_areas = improvement_match.group(1).strip() if improvement_match else "No improvement areas provided"
    upskill_suggestions = upskill_match.group(1).strip() if upskill_match else "No upskill suggestions provided"

    return {
        "resume_rating": str(resume_rating),
        "improvement_areas": improvement_areas.split("\n- ") if improvement_areas != "No improvement areas provided" else "No improvement areas provided",
        "upskill_suggestions": upskill_suggestions.split("\n- ") if upskill_suggestions != "No upskill suggestions provided" else "No upskill suggestions provided"
    }

