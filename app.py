import streamlit as st
import os
import shutil
import subprocess
import pandas as pd
import wikipedia
import sys
import uuid
import json
import librosa
import numpy as np

# Setup full width layout and default sidebar state
st.set_page_config(page_title="Bird Audio Species Detector", page_icon="🐦", layout="wide", initial_sidebar_state="collapsed")

# 1. Custom Theme & Glassmorphism Injection (Plotly + CSS)
st.markdown("""
<style>
    /* Premium background mimicking the dark forest aesthetic with Deep Indigo wash */
    .stApp {
        background: linear-gradient(rgba(10, 14, 26, 0.85), rgba(10, 14, 26, 0.95)), 
                    url("https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?q=80&w=2674&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    
    header { visibility: hidden; }

    /* The "div:has" Glassmorphism hack from Stitch to encapsulate main columns */
    div[data-testid="stVerticalBlock"] > div:has(div.glass-card) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Result Cards */
    .result-banner {
        background: rgba(204, 255, 0, 0.08);
        border-left: 5px solid #CCFF00;
        border-radius: 12px;
        padding: 15px 20px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #CCFF00;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .secondary-card {
        background: rgba(79, 70, 229, 0.1);
        border-left: 3px solid #4F46E5;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }
    
    /* Wikipedia Card Override */
    .wiki-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Primary Run Action Button styling */
    div.stButton > button {
        background: linear-gradient(45deg, #4F46E5, #CCFF00);
        color: #0A0E1A !important;
        font-weight: 900;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        margin-top: 20px;
        width: 100%;
        box-shadow: 0 0 15px rgba(204, 255, 0, 0.3), inset 0 0 10px rgba(255,255,255,0.2);
    }
    div.stButton > button:hover {
        box-shadow: 0 0 25px rgba(204, 255, 0, 0.7);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.title("Bird Species Detector Dashboard")

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_wikipedia_info(species_name):
    import wikipedia
    search_queries = [species_name, species_name + " bird"]
    
    for query in search_queries:
        try:
            search_results = wikipedia.search(query)
            if not search_results: continue
                
            page = wikipedia.page(search_results[0], auto_suggest=False)
            summary = wikipedia.summary(search_results[0], sentences=3)
            
            image_url = None
            if hasattr(page, 'images') and page.images:
                for img in page.images:
                    img_lower = img.lower()
                    if img_lower.endswith(('.jpg', '.jpeg', '.png')) and "commons-logo" not in img_lower and "wikiquote" not in img_lower:
                        image_url = img
                        break
                    
            return {"summary": summary, "url": page.url, "image_url": image_url, "title": page.title}
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
                return {"summary": summary, "url": page.url, "image_url": image_url, "title": page.title}
            except:
                pass
        except Exception:
            pass
            
    return None

def clear_directory(dir_path):
    if os.path.exists(dir_path): shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# 3. Layout Structure
col_left, col_right = st.columns([1, 2.5], gap="large")

with col_left:
    st.markdown('<div class="glass-card"></div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Project Settings")
    uploaded_file = st.file_uploader("Upload Audio File (WAV, MP3)", type=["wav", "mp3", "flac", "ogg"])
    
    # CRITICAL FIX: Explicitly rename back to Sources, not Intensity! A value of 1 skips separation leading to dropping the prediction accuracy!
    n_sources = st.slider("Number of NMF Sources", min_value=1, max_value=5, value=2, step=1)
    st.caption("Splits multi-bird calls. Set to 2 or higher for best accuracy!")
    
    st.markdown("### 🌍 Geographic Targeting")
    enable_location = st.checkbox("Enable Filtering", value=False)
    
    c_lat, c_lon = st.columns(2)
    with c_lat: lat = st.number_input("Latitude", value=28.59, disabled=not enable_location)
    with c_lon: lon = st.number_input("Longitude", value=83.94, disabled=not enable_location)
    week = st.slider("Week of Year (-1 for all)", min_value=-1, max_value=48, value=-1, disabled=not enable_location)
    
    run_clicked = st.button("🚀 RUN ANALYSIS", use_container_width=True)

with col_right:
    st.markdown('<div class="glass-card"></div>', unsafe_allow_html=True)
    st.subheader("Acoustic Data Profile")
    
    if uploaded_file is not None:
        try:
            import plotly.graph_objects as go
            
            uploaded_file.seek(0)
            y, sr = librosa.load(uploaded_file, sr=22050)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            if D.shape[1] > 1200: D = D[:, ::D.shape[1]//1200]
            if D.shape[0] > 400: D = D[::D.shape[0]//400, :]
                
            colorscale = [[0.0, '#0A0E1A'], [0.2, '#1e1b4b'], [0.5, '#4F46E5'], [0.8, '#a3e635'], [1.0, '#CCFF00']]
            
            fig = go.Figure(data=go.Heatmap(z=D, colorscale=colorscale, showscale=False, hoverinfo='skip'))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=250, yaxis=dict(showgrid=False, zeroline=False, visible=False),
                xaxis=dict(showgrid=False, zeroline=False, visible=False)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            uploaded_file.seek(0)
            st.audio(uploaded_file, format='audio/wav')
                
        except Exception as e:
            st.warning(f"⚠️ Spectrogram generation skipped or failed: {e}")
            
    else:
        st.info("Upload an audio file on the left to seamlessly render the interactive Plotly Spectrogram.")
        
    st.markdown("<br>", unsafe_allow_html=True)

    if run_clicked and uploaded_file is not None:
        diagnostics = [f"[INFO] Audio Loaded: {uploaded_file.name}"]
        
        try:
            with st.spinner("Processing deep induction..."):
                input_dir = "input"
                processed_dir = "processed"
                
                clear_directory(input_dir)
                clear_directory(processed_dir)
                os.makedirs(os.path.join(input_dir, "Unknown_Bird"), exist_ok=True)
                
                upload_path = os.path.join(input_dir, "Unknown_Bird", uploaded_file.name)
                uploaded_file.seek(0)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                diagnostics.append(f"[PROCESS] Running NMF Separation (Sources: {n_sources}) ...")
                
                current_run_id = str(uuid.uuid4())
                cmd = [sys.executable, "main.py", "--n_sources", str(n_sources), "--run_id", current_run_id]
                if enable_location:
                    cmd.extend(["--lat", str(lat), "--lon", str(lon), "--week", str(week)])
                
                import time
                t0 = time.time()
                process = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
                elapsed = time.time() - t0
                
                if process.returncode != 0:
                    st.error("Backend process failed!")
                    st.code(process.stderr)
                    st.stop()
                    
                diagnostics.append(f"[SUCCESS] Native inference loop finished in {elapsed:.2f}s")
                    
                results_file = "prebuilt_birdnet_evaluation.csv"
                df = pd.read_csv(results_file)
                current_base_name = os.path.splitext(uploaded_file.name)[0]
                
                if "Run_ID" in df.columns:
                    df_current = df[df["Run_ID"] == current_run_id]
                else:
                    df_current = df[df["Filename"] == current_base_name]
                    
                if len(df_current) == 0:
                    st.error("No predictions returned.")
                    st.stop()
                    
                top_row = df_current.iloc[-1]
                
                predictions = []
                if "Top_Predictions_JSON" in top_row and isinstance(top_row["Top_Predictions_JSON"], str):
                    try:
                        predictions = json.loads(top_row["Top_Predictions_JSON"])
                    except Exception:
                        pass
                if not predictions:
                     predictions = [{"species": top_row.get("Top_Predicted_Species", "Unknown"), "confidence": top_row.get("Top_Prediction_Confidence", 0.0)}]
                
                diagnostics.append(f"[MODEL] Classifying top candidate: {predictions[0].get('species', 'Unknown')}")
                
                with st.expander("Show Backend Diagnostics", expanded=False):
                    st.code("\n".join(diagnostics), language="bash")
                
                lead_pred = predictions[0]
                lead_species = lead_pred.get("species", "Unknown")
                lead_conf = float(lead_pred.get("confidence", 0.0))
                
                st.markdown('<div class="glass-card"></div>', unsafe_allow_html=True)
                st.subheader("Analysis Validation Matrix")
                
                st.markdown(f'''
                <div class="result-banner">
                    <span style="display:flex; flex-direction:column;">
                        <span style="font-size:0.8rem; color:#A1A1AA; font-weight:normal;">Probable Species:</span>
                        <span>{lead_species}</span>
                    </span>
                    <span>Confidence: {(lead_conf*100):.1f}%</span>
                </div>
                ''', unsafe_allow_html=True)
                
                if len(predictions) > 1:
                    st.markdown("##### Secondary Signatures Detected")
                    for i, pred in enumerate(predictions[1:], start=2):
                        sec_species = pred.get("species", "Unknown")
                        sec_conf = float(pred.get("confidence", 0.0))
                        
                        st.markdown(f'''
                        <div class="secondary-card">
                            <span style="color: #818CF8; font-weight: 700; margin-right: 10px;">#{i} Secondary Species:</span> 
                            <span style="color: #FFFFFF;">{sec_species}</span> 
                            <span style="color: #9CA3AF; font-size: 0.9em; float: right;">Confidence: {(sec_conf*100):.1f}%</span>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown("---")
                col_info, col_table = st.columns([1.5, 1])
                with col_info:
                    st.markdown('<div class="glass-card"></div>', unsafe_allow_html=True)
                    for pred in predictions:
                        species = pred.get("species", "Unknown")
                        if species in ["None", "Unknown"]: continue
                        
                        wiki_data = fetch_wikipedia_info(species)
                        if wiki_data:
                            st.markdown(f'''
                            <div class="wiki-card">
                                <h4>{species}</h4>
                                <p style="font-size: 0.9em; color: #cbd5e1;">{wiki_data["summary"]}</p>
                                <a href="{wiki_data['url']}" target="_blank" style="color: #CCFF00; text-decoration: none;">[Read Wikipedia Article]</a>
                            </div>
                            ''', unsafe_allow_html=True)
                            if wiki_data.get("image_url"):
                                st.image(wiki_data["image_url"], use_container_width=True)
                                
                with col_table:
                    st.markdown('<div class="glass-card"></div>', unsafe_allow_html=True)
                    st.subheader("Analysis Matrix")
                    st.dataframe(df_current, use_container_width=True)
                    
                    report_content = f"🐦 BIODIVERSITY SURVEY REPORT\n"
                    report_content += "=" * 40 + "\n"
                    report_content += f"Acoustic File: {uploaded_file.name}\n"
                    for i, pred in enumerate(predictions):
                        report_content += f"[{i+1}] {pred.get('species')} | {float(pred.get('confidence', 0.0)):.3f}\n"
                    
                    st.download_button(
                        label="📥 Download Biodiversity Report",
                        data=report_content,
                        file_name=f"survey_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"Pipeline Interrupted: {e}")
