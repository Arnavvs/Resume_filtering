# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from resume_filter import process_resume, ResumeScore, get_recommendations, RecommendationList # Import new functions and models
from pydantic import ValidationError
import json # For JSON serialization of scores

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/ping', methods=['GET'])
def ping():
    return 'Server is alive!'

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
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            resume_file.save(filepath)
            result: ResumeScore = process_resume(job_description_prompt, filepath, strictness_level)
            return jsonify(result.model_dump()), 200 # Use .model_dump() for Pydantic v2+
        except ValidationError as e:
            return jsonify({"error": "Data validation error from LLM output", "details": e.errors()}), 500
        except FileNotFoundError:
            return jsonify({"error": "Resume file not found after upload. Check path."}), 500
        except Exception as e:
            return jsonify({"error": f"An error occurred during resume screening: {str(e)}"}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({"error": "Something went wrong during file upload processing"}), 500

@app.route('/batch_screen', methods=['POST'])
def batch_screen_resumes():
    """
    API endpoint to screen multiple resumes in a batch.
    Expects multiple PDF files under 'resumes[]' and a 'job_description' string in the form data.
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
        if resume_file.filename == '':
            continue # Skip empty files

        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            resume_file.save(filepath)
            score_result: ResumeScore = process_resume(job_description_prompt, filepath, strictness_level)
            results.append({"filename": filename, "score": score_result.model_dump()})
        except ValidationError as e:
            results.append({"filename": filename, "error": "Data validation error from LLM output", "details": e.errors()})
        except Exception as e:
            results.append({"filename": filename, "error": f"Error processing resume: {str(e)}"})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
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


if __name__ == '__main__':
    app.run(debug=True, port=5050, use_reloader=False)

