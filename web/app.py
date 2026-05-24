# web/app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import requests
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import zipfile
import io

st.set_page_config(
    page_title="Behavior Detection System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Modern CSS with clean design
st.markdown("""
<style>
    /* Main layout */
    .main { padding-top: 0.5rem; }
    .uploadedFile { display: none; }
    
    /* Colors */
    :root {
        --primary: #2E7D32;
        --danger: #D32F2F;
        --warning: #F57C00;
        --info: #1976D2;
        --bg-light: #FAFAFA;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #2E7D32;
        background-image: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
    }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #FAFAFA;
    }
    
    /* Header styling */
    h1 { color: #2E7D32; font-weight: 600; }
    h2 { color: #333; font-weight: 600; }
    h3 { color: #555; font-weight: 500; }
    
    /* Metrics card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []
if 'single_result' not in st.session_state:
    st.session_state.single_result = None

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/video.png", width=80)
    st.title("Behavior Detection")
    
    page = st.radio(
        "Navigation",
        ["Analyze Video", "Batch Processing", "Dashboard", "History"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System Status
    st.subheader("System Status", divider="green")
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            response = requests.get(f"{API_URL}/api/health/", timeout=2)
            if response.status_code == 200:
                st.success("API", icon="✅")
            else:
                st.error("API", icon="❌")
        except:
            st.error("API", icon="❌")
    
    with col2:
        import torch
        if torch.cuda.is_available():
            st.success("GPU", icon="✅")
        else:
            st.info("CPU", icon="ℹ️")
    
    # Quick Stats
    st.markdown("---")
    st.subheader("Quick Stats", divider="green")
    
    videos_processed = len(st.session_state.processed_videos)
    if videos_processed > 0:
        total_duration = sum([v['result']['info']['duration'] 
                            for v in st.session_state.processed_videos])
        st.metric("Videos Analyzed", videos_processed)
        st.metric("Total Duration", f"{total_duration:.0f}s")
    else:
        st.info("No videos processed yet", icon="ℹ️")

# Main Content
if page == "Analyze Video":
    st.header("🎥 Analyze Video", divider="green")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            # File info
            st.divider()
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.text(f"📁 Name: {uploaded_file.name}")
            with col_info2:
                st.text(f"📦 Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            st.divider()
            
            if st.button("▶️ Analyze", type="primary", use_container_width=True, key="analyze_btn"):
                with st.spinner("Analyzing..."):
                    progress = st.progress(0)
                    
                    try:
                        progress.progress(30)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        
                        progress.progress(60)
                        response = requests.post(
                            f"{API_URL}/api/video/process",
                            files=files,
                            timeout=300
                        )
                        
                        progress.progress(90)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result["success"]:
                                progress.progress(100)
                                st.session_state.single_result = result
                                
                                st.session_state.processed_videos.append({
                                    'name': uploaded_file.name,
                                    'time': datetime.now(),
                                    'result': result
                                })
                                
                                st.success("✅ Analysis complete!")
                                st.balloons()
                            else:
                                st.error(f"Error: {result.get('error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        progress.empty()
    
    with col2:
        st.subheader("Results")
        
        if st.session_state.single_result:
            result = st.session_state.single_result
            info = result['info']
            
            # Metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Duration", f"{info['duration']}s")
                st.metric("Resolution", info['resolution'])
            with metric_col2:
                st.metric("FPS", f"{info['fps']:.1f}")
                st.metric("Frames", f"{info['frames']:,}")
            
            st.divider()
            
            # Download processed video
            try:
                video_url = f"{API_URL}{result['output_url']}"
                video_response = requests.get(video_url, timeout=30)
                
                if video_response.status_code == 200:
                    st.video(video_response.content)
                    
                    st.download_button(
                        "⬇️ Download Result",
                        data=video_response.content,
                        file_name=f"analyzed_{uploaded_file.name}",
                        mime="video/mp4",
                        use_container_width=True
                    )
            except:
                st.info("Video processing completed but output not yet available")
        else:
            st.info("👈 Upload and analyze a video to see results")

elif page == "Batch Processing":
    st.header("📦 Batch Processing", divider="green")
    
    st.markdown("Upload multiple videos to process them sequentially.")
    
    uploaded_files = st.file_uploader(
        "Choose videos",
        type=['mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader(f"Selected {len(uploaded_files)} Files")
        
        # File list
        file_data = []
        total_size = 0
        for file in uploaded_files:
            size_mb = file.size / 1024 / 1024
            total_size += size_mb
            file_data.append({
                'File': file.name,
                'Size (MB)': f"{size_mb:.2f}"
            })
        
        df_files = pd.DataFrame(file_data)
        st.dataframe(df_files, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(uploaded_files))
        with col2:
            st.metric("Total Size", f"{total_size:.2f} MB")
        with col3:
            estimated_time = len(uploaded_files) * 30
            st.metric("Est. Time", f"{estimated_time//60}m")
        
        st.divider()
        
        if st.button("▶️ Start Processing", type="primary", use_container_width=True):
            st.subheader("Processing Progress")
            
            overall_progress = st.progress(0)
            status_text = st.empty()
            batch_results = []
            
            for idx, file in enumerate(uploaded_files):
                overall_progress.progress((idx) / len(uploaded_files))
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                
                with st.expander(f"📹 {file.name}", expanded=True):
                    file_progress = st.progress(0)
                    file_status = st.empty()
                    
                    try:
                        file_progress.progress(30)
                        file_status.text("📤 Uploading...")
                        
                        files = {"file": (file.name, file.getvalue())}
                        
                        file_progress.progress(60)
                        file_status.text("🔍 Detecting...")
                        
                        response = requests.post(
                            f"{API_URL}/api/video/process",
                            files=files,
                            timeout=300
                        )
                        
                        file_progress.progress(90)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result["success"]:
                                file_progress.progress(100)
                                file_status.text("✅ Done")
                                
                                batch_results.append({
                                    'file': file.name,
                                    'status': 'Success',
                                    'result': result
                                })
                                
                                st.session_state.processed_videos.append({
                                    'name': file.name,
                                    'time': datetime.now(),
                                    'result': result
                                })
                            else:
                                file_status.text("❌ Failed")
                                batch_results.append({
                                    'file': file.name,
                                    'status': 'Failed'
                                })
                    
                    except Exception as e:
                        file_status.text(f"❌ Error")
                        batch_results.append({
                            'file': file.name,
                            'status': 'Error'
                        })
            
            overall_progress.progress(100)
            status_text.text("✅ Batch complete!")
            
            st.divider()
            st.subheader("📊 Summary")
            
            success_count = len([r for r in batch_results if r['status'] == 'Success'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Success", success_count)
            with col2:
                st.metric("❌ Failed", len(batch_results) - success_count)
            with col3:
                rate = (success_count/len(batch_results)*100) if batch_results else 0
                st.metric("Success Rate", f"{rate:.0f}%")
            
            # Download results
            if success_count > 0:
                st.divider()
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for result in batch_results:
                        if result['status'] == 'Success':
                            video_url = f"{API_URL}{result['result']['output_url']}"
                            try:
                                video_response = requests.get(video_url, timeout=10)
                                if video_response.status_code == 200:
                                    zip_file.writestr(
                                        f"analyzed_{result['file']}", 
                                        video_response.content
                                    )
                            except:
                                pass
                
                st.download_button(
                    "⬇️ Download All Results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"batch_results_{datetime.now():%Y%m%d_%H%M%S}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.balloons()

elif page == "Dashboard":
    st.header("📊 Dashboard", divider="green")
    
    if len(st.session_state.processed_videos) == 0:
        st.info("No data available yet. Analyze some videos to see statistics.", icon="ℹ️")
    else:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_videos = len(st.session_state.processed_videos)
        total_duration = sum([v['result']['info']['duration'] 
                            for v in st.session_state.processed_videos])
        avg_fps = sum([v['result']['info']['fps'] 
                     for v in st.session_state.processed_videos]) / total_videos if total_videos > 0 else 0
        
        with col1:
            st.metric("Videos Analyzed", total_videos)
        with col2:
            st.metric("Total Duration", f"{total_duration:.0f}s")
        with col3:
            st.metric("Average FPS", f"{avg_fps:.1f}")
        with col4:
            st.metric("Last Processed", 
                     st.session_state.processed_videos[-1]['time'].strftime("%H:%M"))
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processing Timeline")
            
            timeline_data = []
            for v in st.session_state.processed_videos:
                timeline_data.append({
                    'Time': v['time'],
                    'Video': v['name'][:15] + '...' if len(v['name']) > 15 else v['name'],
                    'Duration': v['result']['info']['duration']
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            fig = px.scatter(df_timeline, x='Time', y='Duration', 
                           hover_data=['Video'],
                           title="Videos Processed Over Time")
            fig.update_traces(marker=dict(size=10, color='#2E7D32'))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Resolution Distribution")
            
            resolutions = {}
            for v in st.session_state.processed_videos:
                res = v['result']['info']['resolution']
                resolutions[res] = resolutions.get(res, 0) + 1
            
            if resolutions:
                fig = px.pie(
                    values=list(resolutions.values()),
                    names=list(resolutions.keys()),
                    title="Video Resolutions"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Recent Videos")
        
        history_data = []
        for v in st.session_state.processed_videos[-10:]:
            history_data.append({
                'Time': v['time'].strftime("%H:%M:%S"),
                'File': v['name'],
                'Duration (s)': v['result']['info']['duration'],
                'Resolution': v['result']['info']['resolution'],
                'FPS': f"{v['result']['info']['fps']:.1f}"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

elif page == "History":
    st.header("📋 History", divider="green")
    
    if len(st.session_state.processed_videos) == 0:
        st.info("No processing history yet.", icon="ℹ️")
    else:
        # Filter and sort options
        col1, col2 = st.columns(2)
        
        with col1:
            sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Duration"])
        
        with col2:
            limit = st.slider("Show last N videos", 5, len(st.session_state.processed_videos), 
                            len(st.session_state.processed_videos))
        
        st.divider()
        
        # Prepare data
        history_data = []
        for v in st.session_state.processed_videos:
            history_data.append({
                'Time': v['time'],
                'File': v['name'],
                'Duration (s)': v['result']['info']['duration'],
                'Resolution': v['result']['info']['resolution'],
                'FPS': f"{v['result']['info']['fps']:.1f}",
                'Frames': v['result']['info']['frames']
            })
        
        df = pd.DataFrame(history_data)
        
        # Sort
        if sort_by == "Date (Oldest)":
            df = df.sort_values('Time', ascending=True)
        elif sort_by == "Duration":
            df = df.sort_values('Duration (s)', ascending=False)
        else:  # Date (Newest)
            df = df.sort_values('Time', ascending=False)
        
        # Limit
        df_display = df.head(limit).copy()
        df_display['Time'] = df_display['Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Videos", len(df))
            st.metric("Total Duration", f"{df['Duration (s)'].sum():.0f}s")
        
        with col2:
            st.metric("Average Duration", f"{df['Duration (s)'].mean():.0f}s")
            st.metric("Average FPS", f"{df['FPS'].str.rstrip('%').astype(float).mean():.1f}")
        
        with col3:
            st.metric("Most Common Resolution", df['Resolution'].mode()[0] if len(df) > 0 else '-')
            st.metric("Total Frames", f"{df['Frames'].sum():,}")
        
        st.divider()
        
        # Export
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                "📊 Export CSV",
                data=csv,
                file_name=f"history_{datetime.now():%Y%m%d}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Clear history
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.processed_videos = []
                st.rerun()


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 1rem 0; color: #999; font-size: 0.9rem;'>
    CS338 Computer Vision - Behavior Detection System
</div>
""", unsafe_allow_html=True)
