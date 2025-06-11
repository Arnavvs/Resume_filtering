# streamlit_dashboard.py
import pdfplumber
import streamlit as st
import requests
import pandas as pd
import io
import json
from werkzeug.utils import secure_filename # Import secure_filename for consistency

# --- Configuration ---
FLASK_BACKEND_URL = "http://localhost:5000" # Set for local development

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
        response.raise_for_status()
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
        response.raise_for_status()
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

# NEW MODULE API CALLS
def call_red_flags_api(resume_content):
    headers = {'Content-Type': 'application/json'}
    data = {"resume_content": resume_content} # Pass content as string, backend will encode
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/module/red_flags", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Red Flags Detector: {e}")
        return None

def call_salary_estimation_api(job_description, resume_content):
    headers = {'Content-Type': 'application/json'}
    data = {"job_description": job_description, "resume_content": resume_content}
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/module/salary_estimation", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Salary Estimation: {e}")
        return None

def call_background_consistency_api(resume_content):
    headers = {'Content-Type': 'application/json'}
    data = {"resume_content": resume_content}
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/module/background_consistency", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Background Consistency Checker: {e}")
        return None

def call_candidate_fit_api(job_description, resume_content):
    headers = {'Content-Type': 'application/json'}
    data = {"job_description": job_description, "resume_content": resume_content}
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/module/candidate_fit", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Candidate Fit Score: {e}")
        return None


def display_results_table(results, selected_candidates_state, raw_results_state):
    """Displays batch processing results in a structured table with selection."""
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
        st.markdown("### Processed Resumes")
        
        # This list will hold the actual full items that are selected
        current_selected_ids = {f"{s['filename']}_{s['score']['name']}" for s in selected_candidates_state if 'score' in s}
        
        display_df = df.copy()
        # Initialize the 'Select' column to False by default and place it first
        display_df.insert(0, 'Select', False) # This line inserts the 'Select' column at index 0

        # Update the 'Select' column based on current_selected_ids
        for i, row in display_df.iterrows():
            unique_id = f"{row['Filename']}_{row.get('Name', 'NoName')}"
            if unique_id in current_selected_ids:
                display_df.at[i, 'Select'] = True

        # Streamlit's new way to handle dataframes with selection
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select candidates for module processing (Max 10)",
                    default=False,
                    width="small"
                )
            },
            hide_index=True,
            use_container_width=True,
            key="processed_resumes_data_editor"
        )
        
        # ... (rest of the function remains the same)
        
        # After editing, process the selections
        # We need to iterate through the original df to get the full original item
        # and then use the `edited_df` for the 'Select' status
        
        # Create a mapping from unique_id to the row in edited_df for easy lookup
        edited_df_map = {f"{row['Filename']}_{row.get('Name', 'NoName')}": row for i, row in edited_df.iterrows()}

        new_selection_state = []
        for original_item in raw_results_state:
            if "score" in original_item: # Only consider successfully scored items
                unique_id = f"{original_item['filename']}_{original_item['score'].get('name', 'NoName')}"
                if unique_id in edited_df_map:
                    if edited_df_map[unique_id]['Select']:
                        if len(new_selection_state) < 10:
                            new_selection_state.append(original_item)
                        else:
                            # If more than 10 are selected, a warning will appear,
                            # but we prevent adding more to the list.
                            # The UI checkbox might still show as selected if the user clicked it,
                            # but the logic prevents it from being added to the state.
                            pass
        
        # Update the session state with the new selection
        # Only update if there's a change to prevent infinite reruns when no actual selection change
        if sorted([f"{s['filename']}_{s['score']['name']}" for s in new_selection_state]) != \
           sorted([f"{s['filename']}_{s['score']['name']}" for s in st.session_state.selected_candidates]):
            st.session_state.selected_candidates = new_selection_state
            st.rerun() # Rerun to update the selection display
        
        # Display expander for details below the table
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

# Initialize session states
if 'raw_results' not in st.session_state:
    st.session_state.raw_results = []
if 'selected_candidates' not in st.session_state:
    st.session_state.selected_candidates = []
if 'show_modules' not in st.session_state:
    st.session_state.show_modules = False
if 'active_module' not in st.session_state:
    st.session_state.active_module = None

# Handle initial processing
if process_button_single:
    with st.spinner("Processing single resume..."):
        single_result = process_single_resume(job_description, uploaded_files[0], strictness_level)
        if single_result:
            # Store resume_content in raw_results for module processing
            resume_content_bytes = uploaded_files[0].getvalue()
            st.session_state.raw_results = [{"filename": uploaded_files[0].name, "score": single_result, "resume_content": resume_content_bytes.decode('latin-1')}]
            st.success("Single resume processed successfully!")
            st.session_state.selected_candidates = [] # Clear selected candidates on new processing
            st.session_state.show_modules = False
            st.session_state.active_module = None
        else:
            st.error("Failed to process single resume.")

if process_button_batch:
    with st.spinner(f"Processing {len(uploaded_files)} resumes in batch..."):
        batch_results = call_batch_screen_api(job_description, uploaded_files, strictness_level)
        if batch_results:
            st.session_state.raw_results = batch_results
            st.success(f"Successfully processed {len(batch_results)} resumes!")
            st.session_state.selected_candidates = [] # Clear selected candidates on new processing
            st.session_state.show_modules = False
            st.session_state.active_module = None
        else:
            st.error("Failed to process batch resumes.")

# Main content area for screening results
st.subheader("Screening Results")

if st.session_state.raw_results:
    display_results_table(st.session_state.raw_results, st.session_state.selected_candidates, st.session_state.raw_results)

    st.markdown("---")
    st.subheader("Candidate Recommendations")
    
    successful_processed_results_for_reco = [
        item for item in st.session_state.raw_results if "score" in item
    ]
    if successful_processed_results_for_reco:
        num_recommendations = st.slider(
            "Number of recommendations to generate:",
            min_value=1,
            max_value=len(successful_processed_results_for_reco),
            value=min(3, len(successful_processed_results_for_reco)),
            key="num_reco_slider"
        )
        recommend_button = st.button("Get Recommendations")

        if recommend_button:
            successful_candidates_for_recommendation = [
                item["score"] for item in successful_processed_results_for_reco
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

# Module Selection Button
if st.session_state.raw_results and st.session_state.selected_candidates:
    st.markdown("---")
    if st.button(f"Run Modules on {len(st.session_state.selected_candidates)} Selected Resumes"): # Changed button text
        st.session_state.show_modules = True
        st.session_state.active_module = None # Reset active module when entering this view
    else:
        st.info(f"Selected {len(st.session_state.selected_candidates)} candidate(s) for module processing.")
elif st.session_state.raw_results:
    st.info("Select candidates from the table above to enable module processing.")


if st.session_state.show_modules:
    st.markdown("---")
    st.header("Candidate Module Analysis")
    
    # Allow user to select which candidate to view modules for
    candidate_options = [
        f"{s['score']['name']} ({s['filename']})" for s in st.session_state.selected_candidates
    ]
    
    if candidate_options:
        selected_candidate_str = st.selectbox(
            "Select a candidate to run modules:",
            options=candidate_options,
            key="module_candidate_select"
        )

        selected_candidate_data = None
        if selected_candidate_str:
            # Find the full data for the selected candidate
            # Need to be careful with name/filename to ensure uniqueness
            parts = selected_candidate_str.rsplit(' (', 1) # Split from the right once
            name = parts[0]
            filename = parts[1][:-1] if len(parts) > 1 else "" # Remove the closing parenthesis
            
            selected_candidate_data = next(
                (s for s in st.session_state.selected_candidates if s['score']['name'] == name and s['filename'] == filename),
                None
            )

        if selected_candidate_data:
            st.markdown(f"### Analyzing: **{selected_candidate_str}**")
            
            st.write("Choose a module to run for the selected candidate:")
            
            # --- Module Buttons with improved layout ---
            module_cols = st.columns(4) # Create 4 columns for the buttons

            with module_cols[0]:
                if st.button("🚨 Red Flags Detector", use_container_width=True, key="btn_red_flags"):
                    st.session_state.active_module = "red_flags"
            with module_cols[1]:
                if st.button("💰 Salary Estimation", use_container_width=True, key="btn_salary"):
                    st.session_state.active_module = "salary_estimation"
            with module_cols[2]:
                if st.button("✅ Background Consistency", use_container_width=True, key="btn_consistency"):
                    st.session_state.active_module = "background_consistency"
            with module_cols[3]:
                if st.button("🎯 Candidate Fit Score", use_container_width=True, key="btn_fit_score"):
                    st.session_state.active_module = "candidate_fit_score"

            st.markdown("---")
            
            # --- Module Results Display ---
            if st.session_state.active_module:
                with st.spinner(f"Running {st.session_state.active_module.replace('_', ' ').title()} for {selected_candidate_str}..."):
                    resume_content_to_send = selected_candidate_data['resume_content'] # This is a string
                    
                    if st.session_state.active_module == "red_flags":
                        red_flags_result = call_red_flags_api(resume_content_to_send)
                        if red_flags_result:
                            st.markdown("#### 🚨 Red Flags Detector Results:")
                            if red_flags_result.get('red_flags_found'):
                                st.error(f"**Overall Assessment:** {red_flags_result.get('overall_assessment')}")
                                if red_flags_result.get('job_hopping'): st.warning(f"**Job Hopping:** {red_flags_result['job_hopping']}")
                                if red_flags_result.get('employment_gaps'): st.warning(f"**Employment Gaps:** {red_flags_result['employment_gaps']}")
                                if red_flags_result.get('buzzword_overuse'): st.warning(f"**Buzzword Overuse:** {red_flags_result['buzzword_overuse']}")
                                if red_flags_result.get('unrealistic_achievements'): st.warning(f"**Unrealistic Achievements:** {red_flags_result['unrealistic_achievements']}")
                                if red_flags_result.get('inconsistent_timelines'): st.warning(f"**Inconsistent Timelines:** {red_flags_result['inconsistent_timelines']}")
                            else:
                                st.success(f"**Overall Assessment:** {red_flags_result.get('overall_assessment', 'No significant red flags detected.')}")
                        else:
                            st.error("Failed to get red flags result.")

                    elif st.session_state.active_module == "salary_estimation":
                        salary_estimate_result = call_salary_estimation_api(job_description, resume_content_to_send)
                        if salary_estimate_result:
                            st.markdown("#### 💰 Salary Estimation Module Results:")
                            st.info(f"**Estimated Salary Range:** {salary_estimate_result.get('estimated_salary_range', 'N/A')}")
                            st.write(f"**Reasoning:** {salary_estimate_result.get('reasoning', 'N/A')}")
                        else:
                            st.error("Failed to get salary estimation result.")

                    elif st.session_state.active_module == "background_consistency":
                        consistency_result = call_background_consistency_api(resume_content_to_send)
                        if consistency_result:
                            st.markdown("#### ✅ Background Consistency Checker Results:")
                            if consistency_result.get('inconsistencies_found'):
                                st.error(f"**Overall Consistency Assessment:** {consistency_result.get('overall_consistency_assessment')}")
                                if consistency_result.get('education_consistency'): st.warning(f"**Education Consistency:** {consistency_result['education_consistency']}")
                                if consistency_result.get('job_title_vs_responsibilities'): st.warning(f"**Job Title vs. Responsibilities:** {consistency_result['job_title_vs_responsibilities']}")
                                if consistency_result.get('employment_date_consistency'): st.warning(f"**Employment Date Consistency:** {consistency_result['employment_date_consistency']}")
                                if consistency_result.get('known_issues'): st.warning(f"**Known Issues/Red Flags:** {consistency_result['known_issues']}")
                            else:
                                st.success(f"**Overall Consistency Assessment:** {consistency_result.get('overall_consistency_assessment', 'No significant inconsistencies found.')}")
                        else:
                            st.error("Failed to get background consistency result.")

                    elif st.session_state.active_module == "candidate_fit_score":
                        fit_score_result = call_candidate_fit_api(job_description, resume_content_to_send)
                        if fit_score_result:
                            st.markdown("#### 🎯 Candidate Fit Score Results:")
                            st.write(f"**Role Fit Score:** {fit_score_result.get('role_fit_score', 'N/A')}/10")
                            st.info(f"**Reasoning for Role Fit:** {fit_score_result.get('role_fit_reason', 'N/A')}")
                            st.write(f"**Culture Fit Score:** {fit_score_result.get('culture_fit_score', 'N/A')}/10")
                            st.info(f"**Reasoning for Culture Fit:** {fit_score_result.get('culture_fit_reason', 'N/A')}")
                            st.write(f"**Overall Fit Assessment:** {fit_score_result.get('overall_fit_assessment', 'N/A')}")
                        else:
                            st.error("Failed to get candidate fit score result.")
            else:
                st.info("Select a module to run for the chosen candidate.")
    else:
        st.info("No candidates selected for module analysis. Please select candidates from the table above.")


st.markdown("---")
st.info("Ensure your Flask backend (app.py) is deployed and accessible at the configured URL.")