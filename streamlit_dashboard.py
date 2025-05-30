# streamlit_dashboard.py
import pdfplumber
import streamlit as st
import requests
import pandas as pd
import io
import json
from werkzeug.utils import secure_filename # Import secure_filename for consistency

# --- Configuration ---
# IMPORTANT: This URL will be updated after you deploy your Flask backend on Vercel
FLASK_BACKEND_URL = "YOUR_VERCEL_BACKEND_URL_HERE" # Replace with your actual Vercel URL!

st.set_page_config(layout="wide", page_title="AI-Powered Resume Screener Dashboard", page_icon="📝")

# --- Helper Functions ---
def call_batch_screen_api(job_description, uploaded_files, strictness):
    """Calls the Flask /batch_screen API."""
    files = [("resumes[]", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
    data = {
        "job_description": job_description,
        "strictness": strictness
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/batch_screen", files=files, data=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend or invalid response: {e}")
        st.error(f"Ensure the backend is running at {FLASK_BACKEND_URL} and accessible.")
        return None

def call_recommend_api(candidate_scores, num_recommendations):
    """Calls the Flask /recommend API."""
    headers = {'Content-Type': 'application/json'}
    data = {
        "candidate_scores": candidate_scores,
        "num_recommendations": num_recommendations
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/recommend", headers=headers, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to recommendation API or invalid response: {e}")
        return None

def process_single_resume(job_description, uploaded_file, strictness):
    """Calls the Flask /screen API for a single resume."""
    files = {"resume": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {
        "job_description": job_description,
        "strictness": strictness
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/screen", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing single resume: {e}")
        st.error(f"Ensure the backend is running at {FLASK_BACKEND_URL} and accessible.")
        return None

def display_results_table(results):
    """Displays batch processing results in a structured table."""
    data = []
    for item in results:
        row = {"Filename": item["filename"]}
        if "score" in item:
            score = item["score"]
            row["Name"] = score.get("name", "N/A")
            row["Aggregate Score"] = f"{score.get('aggregate_score', 0):.2f}"
            row["Technical Score"] = score.get("technical_score", 0)
            row["Soft Skills Score"] = score.get("softskills_score", 0)
            row["Extracurricular Score"] = score.get("extracurricular_score", 0)
            row["Client Need Score"] = score.get("client_need_score", 0)
            # Add details for expander
            row["Technical Reason"] = score.get("technical_reason", "N/A")
            row["Soft Skills Reason"] = score.get("softskills_reason", "N/A")
            row["Extracurricular Reason"] = score.get("extracurricular_reason", "N/A")
            row["Client Need Reason"] = score.get("client_need_reason", "N/A")
        elif "error" in item:
            row["Status"] = "Error"
            row["Details"] = item.get("error", "Unknown error")
        data.append(row)

    df = pd.DataFrame(data)

    if not df.empty:
        # Separate columns for display and expander details
        display_cols = ["Filename", "Name", "Aggregate Score", "Technical Score", "Soft Skills Score", "Extracurricular Score", "Client Need Score"]
        detail_cols = ["Technical Reason", "Soft Skills Reason", "Extracurricular Reason", "Client Need Reason"]

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        for i, row in df.iterrows():
            with st.expander(f"Details for {row['Filename']}: {row.get('Name', '')}"):
                if "Status" in row and row["Status"] == "Error":
                    st.error(f"Error: {row['Details']}")
                else:
                    st.write(f"**Technical:** Score {row['Technical Score']} - {row['Technical Reason']}")
                    st.write(f"**Soft Skills:** Score {row['Soft Skills Score']} - {row['Soft Skills Reason']}")
                    st.write(f"**Extracurricular:** Score {row['Extracurricular Score']} - {row['Extracurricular Reason']}")
                    st.write(f"**Client Need:** Score {row['Client Need Score']} - {row['Client Need Reason']}")
    else:
        st.info("No results to display yet.")


# --- Streamlit UI ---
st.title("AI-Powered Resume Screener Dashboard 🚀")

st.sidebar.header("Job Description & Settings")
job_description = st.sidebar.text_area(
    "Enter Job Description",
    "We are looking for a highly motivated Python Developer with experience in web frameworks like Flask or Django, database management (SQLAlchemy), and cloud platforms (AWS/GCP). Strong problem-solving skills, teamwork, and excellent communication are essential. Experience with machine learning libraries and a passion for open-source contributions are a plus."
)
strictness_level = st.sidebar.selectbox(
    "Screening Strictness",
    options=["low", "medium", "high", "very strict"],
    index=1, # Default to medium
    help="How strictly should the AI match resumes to the job description?"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Resumes")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Resumes (Single or Multiple)",
    type="pdf",
    accept_multiple_files=True
)

process_button_single = None
process_button_batch = None

if uploaded_files:
    if len(uploaded_files) == 1:
        st.sidebar.info("Detected a single resume. You can screen it individually.")
        process_button_single = st.sidebar.button("Process Single Resume")
    elif len(uploaded_files) > 1:
        st.sidebar.info(f"Detected {len(uploaded_files)} resumes. You can process them in batch.")
        process_button_batch = st.sidebar.button("Process Batch Resumes")

# --- Main Content Area ---
st.subheader("Screening Results")

if 'raw_results' not in st.session_state:
    st.session_state.raw_results = []

if process_button_single:
    with st.spinner("Processing single resume..."):
        single_result = process_single_resume(job_description, uploaded_files[0], strictness_level)
        if single_result:
            st.session_state.raw_results = [{"filename": uploaded_files[0].name, "score": single_result}]
            st.success("Single resume processed successfully!")
        else:
            st.error("Failed to process single resume.")

if process_button_batch:
    with st.spinner(f"Processing {len(uploaded_files)} resumes in batch..."):
        batch_results = call_batch_screen_api(job_description, uploaded_files, strictness_level)
        if batch_results:
            st.session_state.raw_results = batch_results
            st.success(f"Successfully processed {len(batch_results)} resumes!")
        else:
            st.error("Failed to process batch resumes.")

if st.session_state.raw_results:
    display_results_table(st.session_state.raw_results)

    st.markdown("---")
    st.subheader("Candidate Recommendations")

    if st.session_state.raw_results:
        # Filter out any error rows to only consider successfully processed candidates
        successful_processed_results = [
            item for item in st.session_state.raw_results if "score" in item
        ]
        if successful_processed_results:
            num_recommendations = st.slider(
                "Number of recommendations to generate:",
                min_value=1,
                max_value=len(successful_processed_results), # Max up to the number of successfully processed candidates
                value=min(3, len(successful_processed_results)), # Default to 3 or less if fewer candidates
                help="Select how many top candidates you want recommendations for."
            )
            recommend_button = st.button("Get Recommendations")

            if recommend_button:
                # Filter out error rows before sending only successful scores to recommendation API
                successful_candidates_for_recommendation = [
                    item["score"] for item in successful_processed_results
                ]
                if successful_candidates_for_recommendation:
                    with st.spinner(f"Getting top {num_recommendations} recommendations..."):
                        recommendations_json = call_recommend_api(successful_candidates_for_recommendation, num_recommendations)
                        if recommendations_json and "recommendations" in recommendations_json:
                            st.subheader(f"Top {num_recommendations} Recommended Candidates:")
                            for i, rec in enumerate(recommendations_json["recommendations"]):
                                st.markdown(f"**{i+1}. {rec['name']}**")
                                st.write(f"   Reason: {rec['reason']}")
                        else:
                            st.warning("Could not get recommendations. Check backend logs.")
                else:
                    st.warning("No successful resume scores to generate recommendations from.")
        else:
            st.info("No successful resume scores available to generate recommendations from.")
    else:
        st.info("Upload and process resumes first to enable recommendations.")

st.markdown("---")
st.info("Ensure your Flask backend (app.py) is deployed and accessible at the configured URL.")