import streamlit as st
import os
import shutil
import subprocess
import pandas as pd
import wikipedia
import sys

st.set_page_config(page_title="Bird Audio Species Detector", page_icon="🐦", layout="wide")

st.markdown("""
<style>
    /* Dark Mode Glassmorphism Aesthetic */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek gradient buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
    }
    
    /* Micro-interactions on file uploader */
    .stFileUploader {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        background: rgba(30, 41, 59, 0.5);
    }
    
    /* Make Alert/Success boxes look premium */
    div.stAlert {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-left: 5px solid;
    }
    
    /* Clean headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐦 Bird Audio Species Detector")
st.markdown("Upload a bird audio recording to run the local project pipeline. The file will be placed in the `input` directory and processed via `main.py` using NMF separation and BirdNET inference.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3, FLAC, OGG)", type=["wav", "mp3", "flac", "ogg"])

with col2:
    st.markdown("### Process Settings")
    n_sources = st.slider("Number of Sources for NMF Separation", min_value=1, max_value=5, value=2, step=1)
    
    st.markdown("### Location Filter (Crucial for Accuracy)")
    st.info("Limit predictions to birds native to a region (e.g., Himalayas) to eliminate impossible global guesses.")
    enable_location = st.checkbox("Enable Geographic Filtering", value=False)
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.59, disabled=not enable_location, help="Default: Himalayas") 
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=83.94, disabled=not enable_location)
    week = st.slider("Week of Year (-1 for all year)", min_value=-1, max_value=48, value=-1, disabled=not enable_location)

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_wikipedia_info(species_name):
    import wikipedia
    # Improve search: Try exact scientific name first, then fallback to adding "bird"
    search_queries = [species_name, species_name + " bird"]
    
    for query in search_queries:
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                continue
                
            page = wikipedia.page(search_results[0], auto_suggest=False)
            summary = wikipedia.summary(search_results[0], sentences=3)
            
            # Find a valid image (ignore SVGs and generic wiki icons)
            image_url = None
            if hasattr(page, 'images') and page.images:
                for img in page.images:
                    img_lower = img.lower()
                    if img_lower.endswith(('.jpg', '.jpeg', '.png')) and "commons-logo" not in img_lower and "wikiquote" not in img_lower:
                        image_url = img
                        break
                    
            return {
                "summary": summary,
                "url": page.url,
                "image_url": image_url,
                "title": page.title
            }
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                summary = wikipedia.summary(e.options[0], sentences=3)
                image_url = None
                if hasattr(page, 'images') and page.images:
                    for img in page.images:
                        img_lower = img.lower()
                        if img_lower.endswith(('.jpg', '.jpeg', '.png')) and "commons-logo" not in img_lower and "wikiquote" not in img_lower:
                            image_url = img
                            break
                return {
                    "summary": summary,
                    "url": page.url,
                    "image_url": image_url,
                    "title": page.title
                }
            except:
                pass
        except Exception:
            pass
            
    return None

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    st.markdown("### 📊 Acoustic Data Profile")
    with st.expander("Explore Neural Spectrogram", expanded=True):
        try:
            import librosa
            import librosa.display
            import matplotlib.pyplot as plt
            import numpy as np
            
            uploaded_file.seek(0)
            y, sr = librosa.load(uploaded_file, sr=22050)
            
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, cmap='magma')
            
            ax.set_title("Vocal Frequency Sweep (Neural Input)", color='white', fontsize=12)
            ax.tick_params(colors='white', labelsize=8)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
            cbar.ax.tick_params(colors='white')
            
            st.pyplot(fig)
            uploaded_file.seek(0)
        except Exception as e:
            st.warning(f"⚠️ Could not render visual spectrogram: {e}")
            uploaded_file.seek(0)
            
    if st.button("Trigger Neural Inference 🚀"):
        try:
            with st.spinner("Preparing directories and saving file..."):
                input_dir = "input"
                processed_dir = "processed"
                
                # Clear directories to ensure a fresh run
                clear_directory(input_dir)
                clear_directory(processed_dir)
                
                # Create a generic species folder for the upload
                species_folder = os.path.join(input_dir, "Unknown_Bird")
                os.makedirs(species_folder, exist_ok=True)
                
                # Save the uploaded file
                upload_path = os.path.join(species_folder, uploaded_file.name)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
            with st.spinner("Running main backend pipeline (`main.py`)..."):
                import uuid
                current_run_id = str(uuid.uuid4())
                
                # Run the backend process with the selected n_sources and location parameters
                cmd = [sys.executable, "main.py", "--n_sources", str(n_sources), "--run_id", current_run_id]
                if enable_location:
                    cmd.extend(["--lat", str(lat), "--lon", str(lon), "--week", str(week)])
                
                process = subprocess.run(
                    cmd,
                    cwd=os.getcwd(),
                    capture_output=True,
                    text=True
                )
                
                if process.returncode != 0:
                    st.error("Backend process failed!")
                    with st.expander("Show Console Error Output"):
                        st.code(process.stderr)
                    st.stop()
                    
            with st.spinner("Parsing results..."):
                results_file = "prebuilt_birdnet_evaluation.csv"
                if not os.path.exists(results_file):
                    st.error("The backend finished but did not produce the expected CSV results file.")
                    st.stop()
                    
                df = pd.read_csv(results_file)
                
                if len(df) == 0:
                    st.warning("Analysis completed, but no high-confidence predictions were found.")
                else:
                    current_base_name = os.path.splitext(uploaded_file.name)[0]
                    
                    if "Run_ID" in df.columns:
                        df_current = df[df["Run_ID"] == current_run_id]
                    else:
                        # Fallback for old CSV rows
                        df_current = df[df["Filename"] == current_base_name]
                        
                    if len(df_current) == 0:
                        # CRITICAL ERROR: The backend silently bypassed predictions!
                        st.error("The backend executed, but absolutely no results were returned for this specific run. Ensure the file contains actual audio or isn't corrupted.")
                        st.stop()
                        
                    import json
                    
                    # We are now guaranteed to ONLY have the exact rows generated by the UUID we just passed to the backend.
                    # This fundamentally murders any caching/ghost data from previous runs!
                    top_row = df_current.iloc[-1]
                    
                    # Native Multi-Species Support: Parse JSON array of dictionaries
                    if "Top_Predictions_JSON" in top_row and isinstance(top_row["Top_Predictions_JSON"], str):
                        try:
                            predictions = json.loads(top_row["Top_Predictions_JSON"])
                        except Exception:
                            predictions = [{"species": top_row.get("Top_Predicted_Species", "Unknown"), "confidence": top_row.get("Top_Prediction_Confidence", top_row.get("Ensemble_Score", 0.0))}]
                    else:
                        predictions = [{"species": top_row.get("Top_Predicted_Species", "Unknown"), "confidence": top_row.get("Top_Prediction_Confidence", top_row.get("Ensemble_Score", 0.0))}]
                        
                    st.markdown("## 🔎 Audio Detection Results")
                    
                    # Generate dynamic result cards for every detected bird
                    for i, pred in enumerate(predictions):
                        species = pred.get("species", "Unknown")
                        conf = float(pred.get("confidence", 0.0))
                        
                        if conf >= 0.40:
                            st.success(f"**#{i+1}** Probable Species: **{species}** (Confidence: **{conf:.3f}**)")
                        else:
                            st.warning(f"**#{i+1}** Candidate: **{species}** (Confidence: **{conf:.3f}**) - *Needs cleaner audio for certainty*")
                            
                    st.markdown("---")
                    
                    col_info, col_table = st.columns([1, 1])
                    
                    with col_info:
                        # Generate dynamic Wikipedia sections for every detected bird
                        for pred in predictions:
                            species = pred.get("species", "Unknown")
                            if species in ["None", "Unknown"]:
                                continue
                                
                            st.subheader(f"📖 About the {species}")
                            with st.spinner(f"Fetching {species}..."):
                                wiki_data = fetch_wikipedia_info(species)
                                
                                if wiki_data:
                                    # Output the image if found
                                    if wiki_data.get("image_url"):
                                        st.image(wiki_data["image_url"], caption=wiki_data.get("title", species), use_container_width=True)
                                        
                                    st.write(wiki_data["summary"])
                                    st.markdown(f"**[Read more on Wikipedia]({wiki_data['url']})**")
                                else:
                                    st.info(f"No detailed Wikipedia information found for {species}.")
                            st.markdown("---")
                                
                    with col_table:
                        st.subheader("📊 Analysis Matrix")
                        st.dataframe(df, use_container_width=True)
                        
                        st.subheader("Backend Execution Log")
                        with st.expander("View AI Diagnostics"):
                            st.text(process.stdout)

                        # Generate Interactive Download Component
                        st.markdown("<br>", unsafe_allow_html=True)
                        report_content = f"🐦 BIODIVERSITY SURVEY REPORT\n"
                        report_content += "=" * 40 + "\n"
                        report_content += f"Acoustic File: {uploaded_file.name}\n"
                        report_content += f"Geolocation Matrix: Lat {lat}, Lon {lon}\n"
                        report_content += f"Sensors Verified: {n_sources} Source Arrays\n\n"
                        report_content += "DETECTED SPECIES SIGNATURES\n"
                        report_content += "-" * 40 + "\n"
                        for i, pred in enumerate(predictions):
                            species = pred.get('species', 'Unknown')
                            conf = float(pred.get('confidence', 0.0))
                            report_content += f"[{i+1}] {species.upper()} | Confidence Index: {conf:.3f}\n"
                        
                        st.download_button(
                            label="📥 Download Secure Biodiversity Report (.txt)",
                            data=report_content,
                            file_name=f"{current_base_name}_survey.txt",
                            mime="text/plain",
                            use_container_width=True,
                            help="Download a certified text document of these AI findings."
                        )

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Built with Streamlit • Native Pipeline Handler</div>", unsafe_allow_html=True)
