# streamlit_dashboard.py
import pdfplumber
import streamlit as st
import requests
import pandas as pd
import io
import json
from werkzeug.utils import secure_filename # Import secure_filename for consistency

# --- Configuration ---
FLASK_BACKEND_URL = "http://127.0.0.1:5050" # Ensure Flask backend is running on this URL

st.set_page_config(layout="wide", page_title="AI-Powered Resume Screener Dashboard", page_icon="📝")

# --- Helper Functions ---
def call_batch_screen_api(job_description, uploaded_files, strictness):
    """Calls the Flask /batch_screen API."""
    # When sending files to requests.post, 'file.name' is used as the filename in the multipart form data.
    # The Flask backend will then apply secure_filename to this 'file.name'.
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
        st.error(f"Error connecting to backend: {e}")
        st.error("Please ensure the Flask backend (app.py) is running on " + FLASK_BACKEND_URL)
        return None

def call_recommend_api(candidate_scores, num_recommendations):
    """Calls the Flask /recommend API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "candidate_scores": candidate_scores,
        "num_recommendations": num_recommendations
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/recommend", headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting recommendations from backend: {e}")
        return None

# --- Dashboard UI ---
st.title("📝 AI-Powered Resume Screener Dashboard")
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("Job Details")
    job_description = st.text_area(
        "Enter Job Description Prompt",
        "We need analysts with SQL and Excel skills with 3 or more than 3 years of experience, but more importantly, they should be proactive, data-driven, and quick learners. Preference for female candidates who’ve participated in inter-college debates or editorial teams. Must be good at cross-functional collaboration. He should be ready to change locations for work.",
        height=200
    )
    strictness = st.radio(
        "Select Strictness Level",
        ("lenient", "medium", "very strict"),
        index=1 # 'medium' is default
    )

    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes (multiple files allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button("Process Resumes", use_container_width=True)

# Main content area
# Initialize session state variables if they don't exist
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = {}
if 'raw_results' not in st.session_state: # Also initialize raw_results for robustness
    st.session_state.raw_results = []

if process_button and uploaded_files:
    with st.spinner("Processing resumes... This may take a while for multiple files."):
        # Clear previous extracted texts when new files are uploaded/processed
        st.session_state.extracted_texts = {}
        st.session_state.processed_results = [] # Clear previous processed data too
        st.session_state.raw_results = [] # Clear previous raw data too

        # Dictionary to temporarily hold extracted text with the *secured filename* as key
        # This ensures consistency with the filenames returned by the Flask backend.
        temp_extracted_texts_map = {}

        # First, extract text from all uploaded PDFs and store them
        for file_data in uploaded_files:
            if file_data.name == '': # Check for empty filename
                continue

            # Generate the secured filename, just like the Flask backend does
            # This is crucial for matching keys later.
            secured_name = secure_filename(file_data.name)

            try:
                with pdfplumber.open(io.BytesIO(file_data.getvalue())) as pdf:
                    text_content = ""
                    for page in pdf.pages:
                        text_content += page.extract_text() or "" # Handle empty pages
                    temp_extracted_texts_map[secured_name] = text_content # Store text with secured name
            except Exception as e:
                st.warning(f"Could not extract text from {file_data.name}: {e}")
                # Store an error message under the secured name if extraction fails
                temp_extracted_texts_map[secured_name] = f"Error extracting text from {file_data.name}: {e}"

        # Now, move the temporarily extracted texts (with secured names) to session_state
        st.session_state.extracted_texts = temp_extracted_texts_map


        # Now call the backend API with the original uploaded files
        raw_results = call_batch_screen_api(job_description, uploaded_files, strictness)
        if raw_results:
            st.session_state.raw_results = raw_results  # Save for later use (e.g., recommendations)
            processed_data = []
            for item in raw_results:
                # Flask backend returns 'filename' as the secured filename.
                # This ensures consistency with the keys stored in st.session_state.extracted_texts.
                file_name_from_backend = item["filename"]

                if "score" in item:
                    score = item["score"]
                    processed_data.append({
                        "File Name": file_name_from_backend,
                        "Name": score.get("name", "N/A"),
                        "Technical Score": score.get("technical_score", 0),
                        "Technical Reason": score.get("technical_reason", ""),
                        "Soft Skills Score": score.get("softskills_score", 0),
                        "Soft Skills Reason": score.get("softskills_reason", ""),
                        "Extracurricular Score": score.get("extracurricular_score", 0),
                        "Extracurricular Reason": score.get("extracurricular_reason", ""),
                        "Client Need Score": score.get("client_need_score", 0),
                        "Client Need Reason": score.get("client_need_reason", ""),
                        "Aggregate Score": score.get("aggregate_score", 0.0)
                    })
                else:
                    processed_data.append({
                        "File Name": file_name_from_backend,
                        "Error": item.get("error", "Unknown error"),
                        "Details": item.get("details", "")
                    })
            st.session_state.processed_results = processed_data
            st.success(f"Successfully processed {len(processed_data)} resumes!")
        else:
            st.error("Failed to process resumes. Check backend logs.")

# Display results section only if there are processed results
if st.session_state.processed_results:
    st.header("📈 Resume Screening Results")

    results_df = pd.DataFrame(st.session_state.processed_results)

    # Convert scores to numeric if they are not already (important for sorting)
    numeric_cols = [col for col in results_df.columns if 'Score' in col]
    for col in numeric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0) # 'coerce' turns non-numeric into NaN, fillna(0) makes them 0

    # --- Feature 1: Resume Text Preview ---
    st.subheader("📄 Resume Text Preview (Click on File Name)")
    # The selectbox options come from the DataFrame, which uses the secured filenames.
    selected_file_name = st.selectbox("Select a resume to preview its extracted text:", results_df["File Name"].unique())
    if selected_file_name:
        # Retrieve extracted text using the secured filename from session state
        extracted_text = st.session_state.extracted_texts.get(selected_file_name, "Text not found or extraction error.")
        st.text_area(f"Extracted text from {selected_file_name}", extracted_text, height=300)


    # --- Display Detailed Scores and Sorting ---
    st.subheader("Detailed Candidate Scores")

    sort_column = st.selectbox(
        "Sort results by:",
        ["Aggregate Score", "Technical Score", "Soft Skills Score", "Extracurricular Score", "Client Need Score"],
        index=0 # Default sort by Aggregate Score
    )
    sort_order = st.radio("Sort order:", ("Descending", "Ascending"), index=0)

    # Sort the DataFrame based on user selection
    ascending = True if sort_order == "Ascending" else False
    if sort_column in results_df.columns:
        display_df = results_df.sort_values(by=sort_column, ascending=ascending)
    else:
        display_df = results_df # Fallback if column not found for some reason

    # Display the table, rounding scores for cleaner presentation
    st.dataframe(display_df[[
        "File Name", "Name", "Aggregate Score",
        "Technical Score", "Soft Skills Score", "Extracurricular Score", "Client Need Score",
        "Technical Reason", "Soft Skills Reason", "Extracurricular Reason", "Client Need Reason"
    ]].round(2))

    # --- Feature 2: Download Processed Data ---
    st.subheader("⬇️ Download Results")
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )
    json_data = display_df.to_json(orient="records", indent=4)
    st.download_button(
        label="Download Results as JSON",
        data=json_data,
        file_name="resume_screening_results.json",
        mime="application/json",
    )

    # --- Recommendation Section ---
    st.header("✨ AI-Powered Recommendations")
    # Only show recommendations if there's at least one successfully processed resume
    if any("Error" not in item for item in st.session_state.processed_results):
        num_recommendations = st.slider("How many top candidates would you like to recommend?", 1, len(st.session_state.processed_results), 3)
        recommend_button = st.button("Get Recommendations")

        if recommend_button:
            # Filter out error rows before sending only successful scores to recommendation API
            successful_candidates_for_recommendation = [
                item["score"] for item in st.session_state.get("raw_results", []) if "score" in item
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
        st.info("Upload and process resumes first to enable recommendations.")

st.markdown("---")
st.info("Ensure your Flask backend (app.py) is running before using the dashboard.")