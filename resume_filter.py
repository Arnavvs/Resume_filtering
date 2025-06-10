# resume_filter.py
import json
import os
import pdfplumber
from typing import List, Dict, Any, Optional # Added Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import io

load_dotenv()

# Pydantic models (existing and new)
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
    aggregate_score: float = Field(description="Overall aggregate score of the resume (0-10)")

class CandidateRecommendation(BaseModel):
    name: str = Field(description="Name of the recommended candidate")
    score: float = Field(description="Aggregate score of the candidate")
    reason: str = Field(description="Reason for recommending this candidate based on job requirements")

class RecommendationList(BaseModel):
    recommendations: List[CandidateRecommendation] = Field(description="List of recommended candidates")

# NEW Pydantic Models for the modules
class RedFlags(BaseModel):
    job_hopping: Optional[str] = Field(None, description="Details if frequent job hopping is detected.")
    employment_gaps: Optional[str] = Field(None, description="Details if significant employment gaps are detected.")
    buzzword_overuse: Optional[str] = Field(None, description="Details if there is an overuse of buzzwords.")
    unrealistic_achievements: Optional[str] = Field(None, description="Details if achievements seem unrealistic or exaggerated.")
    inconsistent_timelines: Optional[str] = Field(None, description="Details if timelines or formats are inconsistent.")
    overall_assessment: str = Field(description="Overall assessment of red flags, if any.")
    red_flags_found: bool = Field(description="True if any red flags were found, False otherwise.")

class SalaryEstimation(BaseModel):
    estimated_salary_range: str = Field(description="Estimated annual salary range (e.g., '$70,000 - $90,000').")
    reasoning: str = Field(description="Reasoning for the estimated salary, considering role, experience, skills, and industry.")

class ConsistencyCheck(BaseModel):
    education_consistency: Optional[str] = Field(None, description="Consistency check for educational details.")
    job_title_vs_responsibilities: Optional[str] = Field(None, description="Consistency check between job titles and stated responsibilities.")
    employment_date_consistency: Optional[str] = Field(None, description="Consistency check for employment dates.")
    known_issues: Optional[str] = Field(None, description="Flags if any known fake company names or degrees are potentially mentioned.")
    overall_consistency_assessment: str = Field(description="Overall assessment of background consistency.")
    inconsistencies_found: bool = Field(description="True if any inconsistencies were found, False otherwise.")

class FitScore(BaseModel):
    role_fit_score: int = Field(..., description="Score from 0-10 for role fit based on job description keywords.")
    role_fit_reason: str = Field(description="Reasoning for the role fit score.")
    culture_fit_score: int = Field(..., description="Score from 0-10 for culture fit based on soft skills, language tone, and past environments.")
    culture_fit_reason: str = Field(description="Reasoning for the culture fit score.")
    overall_fit_assessment: str = Field(description="Overall assessment of candidate fit.")

# --- Core Functions ---

def extract_text_from_pdf(pdf_file_object: io.BytesIO) -> str:
    """Extracts text from a PDF file-like object."""
    text = ""
    try:
        with pdfplumber.open(pdf_file_object) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = ""
    return text

def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your Vercel project settings.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=google_api_key)

def process_resume_from_bytes(job_description_prompt: str, resume_bytes: bytes, strictness_level: str = "medium") -> ResumeScore:
    """
    Processes a resume (provided as bytes) against a job description prompt.
    """
    llm = get_llm()

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
    resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
    if not resume_text:
        raise ValueError("Could not extract text from the provided resume PDF bytes.")

    # 4. Define prompt for resume scoring
    resume_scoring_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI-powered resume screener. Your task is to evaluate a candidate's resume against a given job description and categorized keywords.
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

    if not resume_score.aggregate_score:
        total_score = (
            resume_score.technical_score +
            resume_score.softskills_score +
            resume_score.extracurricular_score +
            resume_score.client_need_score
        )
        resume_score.aggregate_score = (total_score / 40) * 10

    return resume_score


def get_recommendations(candidate_scores: List[Dict[str, Any]], num_recommendations: int) -> RecommendationList:
    """
    Generates recommendations for candidates based on their scores.
    Expects a list of dictionaries, where each dict is a ResumeScore (or its model_dump).
    """
    llm = get_llm()

    sorted_candidates = sorted(
        [s for s in candidate_scores if 'aggregate_score' in s],
        key=lambda x: x['aggregate_score'],
        reverse=True
    )

    top_n_candidates = sorted_candidates[:num_recommendations]
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

# --- NEW MODULE FUNCTIONS (UNCHANGED) ---

def detect_red_flags(resume_text: str) -> RedFlags:
    """
    Highlights potential red flags in a resume.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an HR compliance AI. Analyze the provided resume text for potential red flags.
            Look for:
            - Frequent job-hopping (short tenures, many companies in a short period)
            - Significant employment gaps (periods of unemployment not explained)
            - Overuse of buzzwords/fluff without substance
            - Unrealistic achievements or exaggerated claims
            - Inconsistent timelines or formatting issues that suggest manipulation.
            Provide details for each detected red flag, or indicate if none are found.
            Finally, provide an overall assessment and a boolean indicating if any red flags were found.
            """),
            ("human", "Resume Text: {resume_text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(RedFlags)
    red_flags: RedFlags = chain.invoke({"resume_text": resume_text})
    return red_flags

def estimate_salary(job_description: str, resume_text: str) -> SalaryEstimation:
    """
    Estimates candidate's expected salary based on provided information.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a salary estimation AI. Based on the provided job description and candidate resume,
            estimate an annual salary range for a position in India. Consider the role, location (if implied), experience,
            industry averages, current company/designation (if present), and skillset rarity or market demand.
            Provide a reasoned estimated salary range and your justification.
            Assume the role is in a major Indian metropolitan city (e.g., Bangalore, Delhi NCR, Mumbai) if no specific location is mentioned.
            """),
            ("human", "Job Description: {job_description}\nResume Text: {resume_text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(SalaryEstimation)
    salary_estimate: SalaryEstimation = chain.invoke({"job_description": job_description, "resume_text": resume_text})
    return salary_estimate

def check_background_consistency(resume_text: str) -> ConsistencyCheck:
    """
    Verifies and flags inconsistencies in background details from a resume.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an HR verification AI. Analyze the provided resume text for inconsistencies in background information.
            Check for:
            - Education inconsistencies (e.g., degree mismatch with field, unusual timelines).
            - Discrepancies between job titles and described responsibilities.
            - Inconsistencies in employment dates (overlapping, unexplained gaps).
            - Mention of entities that are commonly associated with fake credentials or known to be fraudulent (e.g., specific fake universities, diploma mills). Note: You do not have access to a real-time database, rely on your training data for general knowledge.
            Provide details for each inconsistency found, or indicate if none are found.
            Provide an overall consistency assessment and a boolean indicating if any inconsistencies were found.
            """),
            ("human", "Resume Text: {resume_text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(ConsistencyCheck)
    consistency_check: ConsistencyCheck = chain.invoke({"resume_text": resume_text})
    return consistency_check

def calculate_fit_score(job_description: str, resume_text: str) -> FitScore:
    """
    Calculates role and culture fit scores for a candidate.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a candidate fit AI. Evaluate the candidate's resume against the job description to provide two distinct fit scores:
            1. Role Fit (0-10): Based on the match of technical skills, experience, and responsibilities mentioned in the resume with the job description keywords.
            2. Culture Fit (0-10): Based on an assessment of soft skills, communication tone (as inferred from descriptions), and past work environments (e.g., startup vs. MNC, team-oriented vs. individual contributor roles) that might align with typical company cultures.
            Provide a score and a concise reason for each. Also, give an overall assessment of candidate fit.
            """),
            ("human", "Job Description: {job_description}\nResume Text: {resume_text}"),
        ]
    )
    chain = prompt | llm.with_structured_output(FitScore)
    fit_score: FitScore = chain.invoke({"job_description": job_description, "resume_text": resume_text})
    return fit_score