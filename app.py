# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from resume_filter import process_resume_from_bytes, ResumeScore, get_recommendations, RecommendationList
from pydantic import ValidationError
import json
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes (important for Streamlit frontend)

@app.route('/ping', methods=['GET'])
def ping():
    """Simple health check endpoint."""
    return 'Server is alive!', 200

@app.route('/screen', methods=['POST'])
def screen_resume():
    """
    API endpoint to screen a single resume.
    Expects a PDF file under 'resume' and a 'job_description' string in the form data.
    Optional 'strictness' parameter.
    """
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    if 'job_description' not in request.form:
        return jsonify({"error": "No job description provided"}), 400

    resume_file = request.files['resume']
    job_description_prompt = request.form['job_description']
    strictness_level = request.form.get('strictness', 'medium')

    if resume_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if resume_file:
        resume_content = resume_file.read() # Read bytes from the uploaded file

        try:
            result: ResumeScore = process_resume_from_bytes(job_description_prompt, resume_content, strictness_level)
            return jsonify(result.model_dump()), 200
        except ValidationError as e:
            return jsonify({"error": "Data validation error from LLM output", "details": e.errors()}), 500
        except Exception as e:
            return jsonify({"error": f"An error occurred during resume screening: {str(e)}"}), 500
    return jsonify({"error": "Something went wrong during file upload processing"}), 500


@app.route('/batch_screen', methods=['POST'])
def batch_screen_resumes():
    """
    API endpoint to screen multiple resumes in a batch.
    Expects multiple PDF files under 'resumes[]' and a 'job_description' string in the form data.
    Optional 'strictness' parameter.
    """
    if 'resumes[]' not in request.files:
        return jsonify({"error": "No resume files provided"}), 400
    if 'job_description' not in request.form:
        return jsonify({"error": "No job description provided"}), 400

    resume_files = request.files.getlist('resumes[]')
    job_description_prompt = request.form['job_description']
    strictness_level = request.form.get('strictness', 'medium')

    if not resume_files:
        return jsonify({"error": "No selected files"}), 400

    results = []
    for resume_file in resume_files:
        filename = secure_filename(resume_file.filename)
        try:
            resume_content = resume_file.read()
            score: ResumeScore = process_resume_from_bytes(job_description_prompt, resume_content, strictness_level)
            results.append({"filename": filename, "score": score.model_dump()})
        except ValidationError as e:
            results.append({"filename": filename, "error": "Data validation error from LLM output", "details": e.errors()})
        except Exception as e:
            results.append({"filename": filename, "error": f"Error processing resume: {str(e)}"})

    return jsonify(results), 200

@app.route('/recommend', methods=['POST'])
def recommend_candidates():
    """
    API endpoint to get recommendations from processed candidate scores.
    Expects a JSON body with 'candidate_scores' (list of dicts) and 'num_recommendations'.
    """
    data = request.json
    if not data or 'candidate_scores' not in data or 'num_recommendations' not in data:
        return jsonify({"error": "Invalid request body. Expected 'candidate_scores' and 'num_recommendations'."}), 400

    candidate_scores = data['candidate_scores']
    num_recommendations = data['num_recommendations']

    try:
        recommendations: RecommendationList = get_recommendations(candidate_scores, num_recommendations)
        return jsonify(recommendations.model_dump()), 200
    except ValidationError as e:
        return jsonify({"error": "Data validation error for recommendations", "details": e.errors()}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred during recommendation generation: {str(e)}"}), 500
