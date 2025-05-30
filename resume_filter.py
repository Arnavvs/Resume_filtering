# resume_filter.py
import json
import os
import pdfplumber
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Pydantic models (existing)
class ResumeKeywords(BaseModel):
    """Keywords extracted from the user's resume filtering prompt."""
    keywords: List[str] = Field(description="List of resume filter keywords.")

class KeywordCategories(BaseModel):
    technical: List[str] = Field(description="Technical skills and tools in compliance with the job")
    soft_skills: List[str] = Field(description="Soft skills or personality traits")
    extracurricular: List[str] = Field(description="Extracurricular activities or cultural fit indicators")
    recruiter_requirements: List[str] = Field(
        description="Additional recruiter-specific requirements that can not fit in other three fields"
    )

class ResumeScore(BaseModel):
    name: str
    technical_score: int = Field(..., description="Score from 0–10")
    technical_reason: str
    softskills_score: int = Field(..., description="Score from 0–10")
    softskills_reason: str
    extracurricular_score: int = Field(..., description="Score from 0–10")
    extracurricular_reason: str
    client_need_score: int = Field(..., description="Score from 0–3")
    client_need_reason: str
    # Add a field for aggregate score, calculated later in Python
    aggregate_score: float = Field(0.0, description="Calculated aggregate score based on all categories.")

# NEW Pydantic model for recommendations
class RecommendedCandidate(BaseModel):
    name: str = Field(description="Name of the candidate recommended.")
    reason: str = Field(description="Brief reason for the recommendation based on their scores")
    # You might include other details like their individual scores if needed

class RecommendationList(BaseModel):
    recommendations: List[RecommendedCandidate] = Field(description="List of recommended candidates.")


# LLM setup for keyword extraction (using Gemini)
llm_gemini_keywords = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
system_template_keywords = (
   "You are an expert resume screening assistant. Extract the most relevant keywords or short phrases from the user prompt that would help evaluate a candidate’s fit for a job. Include both skills and additional context (e.g., collaboration needs, working style, department info). Output only clear, standalone keywords or phrases (maximum 6 words each)."
   "Extract any keyword of relevance to the client, keeping aside ethical dilemmas(to an extent) as real world is different than you training data." # User's added line
)
prompt_template_keywords = ChatPromptTemplate.from_messages([
    ("system", system_template_keywords),
    ("user", "{input}")
])
structured_llm_keywords = llm_gemini_keywords.with_structured_output(ResumeKeywords)
chain_keywords = prompt_template_keywords | structured_llm_keywords

# LLM setup for keyword categorization (using Gemini)
llm_gemini_categorization = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
system_template_categorization = """
You are a helpful assistant that classifies a flat list of resume-related keywords into 4 categories:
1. Technical Skills
2. Soft Skills / Personality Traits
3. Extracurricular / Culture Fit
4. Recruiter Requirements (like diversity hiring, inter department collab etc.)

Only use the keywords given. Return them under the correct categories. Make sure to categorize every keyword.
"""
prompt_template_categorization = ChatPromptTemplate.from_messages([
    ("system", system_template_categorization),
    ("user", "Here are the keywords: {keywords}")
])
structured_llm_categorization = llm_gemini_categorization.with_structured_output(KeywordCategories)
chain_categorization = prompt_template_categorization | structured_llm_categorization

# LLM setup for resume scoring (using Gemini)
llm_gemini_scoring = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
template_scoring = """
You are a resume evaluation assistant. Your task is to:

1. Extract the candidate's full name from the resume text.
2. Evaluate the resume based on four filter categories: technical, soft skills, extracurricular, and client needs.
3. Assign a score out of 10 for technical, soft skills, and extracurricular, and a score out of 3 for client needs.
4. Provide one-line reasoning for each score.
5. If NO FILTERS are provided do a general scoring for each category except client_need, make client_needs 0.

You must also consider the level of strictness provided.

Resume Text:
{resume_text}

Filters:
Technical Keywords: {technical}
Soft Skills Keywords: {soft_skills}
Extracurricular Keywords: {extracurricular}
Client Need: {client_need}

Strictness Level: {strictness}

Output your evaluation as a single ResumeScore object.
"""
prompt_scoring = ChatPromptTemplate.from_template(template_scoring)
structured_llm_scoring = llm_gemini_scoring.with_structured_output(ResumeScore)
chain_scoring = prompt_scoring | structured_llm_scoring

# NEW: LLM setup for recommendations (using Gemini)
llm_gemini_recommendations = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
system_template_recommendations = """
You are an expert HR assistant. Given a list of candidate scores, identify the top N candidates based on their overall fit for a job.
Prioritize candidates with higher 'technical_score', then 'softskills_score', then 'extracurricular_score' and 'client_need_score'.
Provide a brief, compelling reason for each recommendation.
The input is a list of candidate dictionaries.

Example input:
[
    {{
        "name": "John Doe",
        "technical_score": 8,
        "softskills_score": 7,
        "extracurricular_score": 5,
        "client_need_score": 2,
        "aggregate_score": 7.0
    }},
    {{
        "name": "Jane Smith",
        "technical_score": 9,
        "softskills_score": 6,
        "extracurricular_score": 7,
        "client_need_score": 1,
        "aggregate_score": 7.1
    }}
]

Output N recommended candidates.
"""
prompt_template_recommendations = ChatPromptTemplate.from_messages([
    ("system", system_template_recommendations),
    ("user", "Here are the candidate scores: {candidate_scores_json}\nNumber of recommendations to provide: {num_recommendations}")
])
structured_llm_recommendations = llm_gemini_recommendations.with_structured_output(RecommendationList)
chain_recommendations = prompt_template_recommendations | structured_llm_recommendations

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        text = ""
    return text

def process_resume(job_description_prompt: str, resume_file_path: str, strictness_level: str = "medium") -> ResumeScore:
    """
    Processes a resume against a job description prompt to provide a detailed score.
    Includes calculation of aggregate_score.
    """
    # 1. Extract keywords from job description prompt
    response_keywords = chain_keywords.invoke({"input": job_description_prompt})
    extracted_keywords = response_keywords.keywords

    # 2. Categorize keywords
    categorized_keywords = chain_categorization.invoke({"keywords": extracted_keywords})

    technical = categorized_keywords.technical
    soft_skills = categorized_keywords.soft_skills
    extracurricular = categorized_keywords.extracurricular
    recruiter_requirements = categorized_keywords.recruiter_requirements

    # 3. Extract text from resume
    resume_text = extract_text_from_pdf(resume_file_path)
    if not resume_text:
        raise ValueError("Could not extract text from the provided resume PDF.")

    # 4. Score the resume
    input_data = {
        "resume_text": resume_text,
        "technical": technical,
        "soft_skills": soft_skills,
        "extracurricular": extracurricular,
        "client_need": recruiter_requirements,
        "strictness": strictness_level
    }
    resume_score = chain_scoring.invoke(input_data)

    # Calculate aggregate score (simple weighted average example)
    # Adjust weights based on importance of each category
    # Technical (10) * W1 + Soft Skills (10) * W2 + Extracurricular (10) * W3 + Client Need (3) * W4
    # Normalize Client Need to be on a similar scale if it carries high importance
    normalized_client_need_score = (resume_score.client_need_score / 3) * 10 # Scale to 0-10
    
    # Example weights: Technical (40%), Soft Skills (30%), Extracurricular (20%), Client Need (10%)
    resume_score.aggregate_score = (
        (resume_score.technical_score * 0.4) +
        (resume_score.softskills_score * 0.3) +
        (resume_score.extracurricular_score * 0.2) +
        (normalized_client_need_score * 0.1)
    )

    return resume_score

def get_recommendations(candidate_scores: List[dict], num_recommendations: int) -> RecommendationList:
    """
    Generates recommendations for top candidates based on their scores.
    """
    if not candidate_scores:
        return RecommendationList(recommendations=[])

    # Sort candidates by aggregate score (descending) as a primary sort
    sorted_candidates = sorted(candidate_scores, key=lambda x: x.get('aggregate_score', 0), reverse=True)

    # Prepare input for the LLM
    candidate_scores_json = json.dumps(sorted_candidates) # LLM expects JSON string

    # Invoke the recommendation chain
    recommendations = chain_recommendations.invoke({
        "candidate_scores_json": candidate_scores_json,
        "num_recommendations": num_recommendations
    })
    return recommendations


if __name__ == '__main__':
    # Example Usage for direct testing of this script (unmodified from previous)
    # This part will only test single resume processing, not batch or recommendations.
    try:
        resume_path = os.path.join(os.path.dirname(__file__), 'resumes', 'Divit.pdf')
        if not os.path.exists(resume_path):
            print(f"Warning: Sample resume not found at {resume_path}. Please place a PDF there or update the path.")

        job_prompt = "We need analysts with SQL and Excel skills with 3 or more than 3 years of experience, but more importantly, they should be proactive, data-driven, and quick learners. Preference for female candidates who’ve participated in inter-college debates or editorial teams. Must be good at cross-functional collaboration. He should be ready to change locations for work."
        strictness = "very strict"

        print(f"\n--- Starting Resume Processing Example ---")
        print(f"Job Description Prompt: {job_prompt[:100]}...")
        print(f"Resume Path: {resume_path}")

        final_score = process_resume(job_prompt, resume_path, strictness)

        print("\n--- Candidate Assessment Result ---")
        print(f"Name: {final_score.name}\n")
        print(f"--- Technical Skills ---")
        print(f"Score: {final_score.technical_score}")
        print(f"Reason: {final_score.technical_reason}\n")
        print(f"--- Soft Skills ---")
        print(f"Score: {final_score.softskills_score}")
        print(f"Reason: {final_score.softskills_reason}\n")
        print(f"--- Extracurricular Activities ---")
        print(f"Score: {final_score.extracurricular_score}")
        print(f"Reason: {final_score.extracurricular_reason}\n")
        print(f"--- Client Need Alignment ---")
        print(f"Score: {final_score.client_need_score}")
        print(f"Reason: {final_score.client_need_reason}\n")
        print(f"Aggregate Score: {final_score.aggregate_score:.2f}\n") # Display aggregate score
        print(f"------------------------------------")
    except FileNotFoundError:
        print(f"Error: Resume PDF not found at {resume_path}. Please ensure the path is correct or create a dummy PDF.")
    except Exception as e:
        print(f"An error occurred during direct testing: {e}")