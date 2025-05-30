# resume_filter.py
import json
import os
import pdfplumber
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError # Ensure ValidationError is imported
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import io # NEW: Import io for in-memory file handling

load_dotenv() # Load environment variables for local testing (but not deployed with .env)

# Pydantic models (existing - ensure all are defined)
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
    name: str = Field(description="The name of the candidate found in the resume")
    technical_score: int = Field(..., description="Score from 0–10")
    technical_reason: str = Field(description="Reasoning for the technical score")
    softskills_score: int = Field(..., description="Score from 0–10")
    softskills_reason: str = Field(description="Reasoning for the soft skills score")
    extracurricular_score: int = Field(..., description="Score from 0–10")
    extracurricular_reason: str = Field(description="Reasoning for the extracurricular score")
    client_need_score: int = Field(..., description="Score from 0–10")
    client_need_reason: str = Field(description="Reasoning for the client need alignment score")
    aggregate_score: float = Field(description="Overall aggregate score of the resume (0-10)") # Added aggregate_score

class CandidateRecommendation(BaseModel):
    name: str = Field(description="Name of the recommended candidate")
    score: float = Field(description="Aggregate score of the candidate")
    reason: str = Field(description="Reason for recommending this candidate based on job requirements")

class RecommendationList(BaseModel):
    recommendations: List[CandidateRecommendation] = Field(description="List of recommended candidates")


# --- Core Functions ---

def extract_text_from_pdf(pdf_file_object: io.BytesIO) -> str: # Change signature to accept io.BytesIO
    """Extracts text from a PDF file-like object."""
    text = ""
    try:
        # pdfplumber.open can directly accept file-like objects (like io.BytesIO)
        with pdfplumber.open(pdf_file_object) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = ""
    return text

# NEW FUNCTION: Wrapper to handle bytes input for the main processing logic
def process_resume_from_bytes(job_description_prompt: str, resume_bytes: bytes, strictness_level: str = "medium") -> ResumeScore:
    """
    Processes a resume (provided as bytes) against a job description prompt.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your Vercel project settings.")

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, google_api_key=google_api_key)

    # 1. Define prompt for keyword extraction
    keyword_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert HR assistant. Extract a comprehensive list of keywords and requirements from the following job description that are crucial for a candidate to possess. Include technical skills, soft skills, extracurricular activities, and any other specific requirements mentioned. Be concise and precise."),
            ("human", "Job Description: {job_description}"),
        ]
    )

    keyword_extraction_chain = keyword_extraction_prompt | llm.with_structured_output(ResumeKeywords)
    extracted_keywords_obj: ResumeKeywords = keyword_extraction_chain.invoke({"job_description": job_description_prompt})
    extracted_keywords_list = extracted_keywords_obj.keywords

    # 2. Categorize keywords based on a predefined prompt
    keyword_categorization_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Categorize the following keywords into 'technical', 'soft_skills', 'extracurricular', and 'recruiter_requirements'. Be precise and allocate each keyword to only one category."),
            ("human", "Keywords: {keywords}"),
        ]
    )
    keyword_categorization_chain = keyword_categorization_prompt | llm.with_structured_output(KeywordCategories)
    keyword_categories: KeywordCategories = keyword_categorization_chain.invoke({"keywords": ", ".join(extracted_keywords_list)})

    # 3. Extract text from resume using the bytes
    # Pass the bytes wrapped in io.BytesIO to pdfplumber
    resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
    if not resume_text:
        raise ValueError("Could not extract text from the provided resume PDF bytes.")

    # 4. Define prompt for resume scoring
    resume_scoring_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI-powered resume screener. Your task is to evaluate a candidate's resume against a given job description and categorize extracted keywords.
            Provide a score from 0-10 for each category (technical, soft skills, extracurricular, client need alignment) and a concise reasoning for each score.
            Finally, calculate an aggregate score (0-10) considering all aspects.
            
            Strictness Level: {strictness_level}
            - 'low': Be lenient, score higher if there's any resemblance.
            - 'medium': Be balanced, match significant keywords.
            - 'high': Be very strict, require exact matches and strong evidence.
            - 'very strict': Only perfect matches and strong evidence will score high.

            Job Description: {job_description}

            Categorized Keywords for evaluation:
            Technical Skills: {technical_keywords}
            Soft Skills: {soft_skills_keywords}
            Extracurricular/Cultural Fit: {extracurricular_keywords}
            Recruiter Requirements: {recruiter_requirements}

            Candidate Resume Text: {resume_text}

            Provide your assessment in the specified JSON format, including the candidate's name.
            """),
            ("human", "Evaluate the candidate's resume against the job description and provide a score and reasoning for each category and an overall aggregate score.")
        ]
    )

    resume_scoring_chain = resume_scoring_prompt | llm.with_structured_output(ResumeScore)

    # Invoke the chain with all necessary inputs
    resume_score: ResumeScore = resume_scoring_chain.invoke(
        {
            "strictness_level": strictness_level,
            "job_description": job_description_prompt,
            "technical_keywords": ", ".join(keyword_categories.technical),
            "soft_skills_keywords": ", ".join(keyword_categories.soft_skills),
            "extracurricular_keywords": ", ".join(keyword_categories.extracurricular),
            "recruiter_requirements": ", ".join(keyword_categories.recruiter_requirements),
            "resume_text": resume_text,
        }
    )

    # Manually calculate and add aggregate_score if not directly provided by LLM (adjust based on LLM output)
    # The Pydantic model now includes aggregate_score, so LLM should ideally provide it.
    # If not, you might add logic here:
    if not resume_score.aggregate_score:
        total_score = (
            resume_score.technical_score +
            resume_score.softskills_score +
            resume_score.extracurricular_score +
            resume_score.client_need_score
        )
        # Assuming all categories contribute equally to a max score of 40 (4 categories * 10 max score)
        resume_score.aggregate_score = (total_score / 40) * 10

    return resume_score


def get_recommendations(candidate_scores: List[Dict[str, Any]], num_recommendations: int) -> RecommendationList:
    """
    Generates recommendations for candidates based on their scores.
    Expects a list of dictionaries, where each dict is a ResumeScore (or its model_dump).
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your Vercel project settings.")

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, google_api_key=google_api_key)

    # Sort candidates by aggregate score in descending order
    # Ensure 'aggregate_score' is present in the dictionaries
    sorted_candidates = sorted(
        [s for s in candidate_scores if 'aggregate_score' in s],
        key=lambda x: x['aggregate_score'],
        reverse=True
    )

    # Take the top N candidates
    top_n_candidates = sorted_candidates[:num_recommendations]

    # Convert to a string format suitable for the prompt
    candidate_data_str = json.dumps(top_n_candidates, indent=2)

    recommendation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI HR assistant. Based on the provided list of candidate scores (sorted by aggregate score, highest first),
            identify the top candidates. For each of the top candidates, provide a concise reason for their recommendation, focusing on their strengths
            as inferred from their scores and reasons. Ensure the reasons are distinct and highlight key aspects that make them suitable.
            Provide exactly {num_recommendations} recommendations, focusing on the highest-scoring candidates.
            """),
            ("human", "Candidate Scores: {candidate_data}"),
        ]
    )

    recommendation_chain = recommendation_prompt | llm.with_structured_output(RecommendationList)

    recommendations: RecommendationList = recommendation_chain.invoke(
        {"candidate_data": candidate_data_str, "num_recommendations": num_recommendations}
    )

    return recommendations

# Remove or comment out the example `if __name__ == '__main__':` block as it's for local CLI usage
# and not needed for deployment.
# if __name__ == '__main__':
#     # Example usage remains the same if you want to test locally via CLI
#     # ... (your existing example usage) ...