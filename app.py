import streamlit as st
import cv2
import numpy as np
import time
import threading
from scipy.spatial import distance
import mediapipe as mp

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Driver Drowsiness System", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# GORGEOUS GLASSMORPHISM & CLEAN UI CSS
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Base */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e1e2f); /* Deep elegant slate background */
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Elegant Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Streamlit Containers & Layout spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Login Centered Glass Card */
    .login-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        max-width: 450px;
        margin: auto;
    }
    
    /* Modern Dashboard Status Card */
    .status-card {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: white;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .status-active {
        background: linear-gradient(135deg, #059669, #10b981); /* Clean emerald green */
    }
    .status-warning {
        background: linear-gradient(135deg, #e11d48, #f43f5e); /* Clean crimson red */
        animation: pulse-border 1.5s infinite;
    }
    .status-idle {
        background: linear-gradient(135deg, #334155, #475569); /* Soft gray */
    }
    
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.5); transform: scale(1); }
        50% { box-shadow: 0 0 25px 15px rgba(244, 63, 94, 0); transform: scale(1.02); }
        100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); transform: scale(1); }
    }
    
    /* Emergency Rest locator card */
    .rest-stop-card {
        background: rgba(15, 23, 42, 0.8); border: 1px solid #38bdf8;
        padding: 20px; border-radius: 16px; margin-top: 20px;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    .emergency-btn {
        background: #ef4444; color: white; width: 100%; border: none;
        padding: 15px; border-radius: 10px; font-size: 1.2rem; font-weight: bold;
        cursor: pointer; margin-top: 10px; display: block; text-align: center;
    }
    
    /* Clean Streamlit Metrics */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important; /* Soft sky blue */
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important; /* Cool gray */
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.9rem !important;
    }
    
    /* Soft Button Aesthetics */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid rgba(255,255,255,0.1);
        color: #f8fafc;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: rgba(71, 85, 105, 0.9);
        border-color: #38bdf8;
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.2);
        color: #38bdf8;
    }
    
    hr {
        border-color: rgba(255,255,255,0.08);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)





# ==========================================
# AUDIO & CORE LOGIC
# ==========================================
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def async_beep():
    try:
        import winsound
        winsound.Beep(1500, 500)
    except Exception:
        print('\a') 

def play_alert_sound():
    t = threading.Thread(target=async_beep)
    t.start()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================================
# AI ENGINE (face_mesh — cloud compatible)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ==========================================
# MAIN DASHBOARD CONTROLLER
# ==========================================
def main_app():
    detector = load_ai_engine()
    
    # ------------- SYSTEM SIDEBAR -------------
    st.sidebar.markdown(f"<h2 style='text-align: center; color: #38bdf8;'>🚗 AI Guardian </h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin-top:0.5rem;'>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚙️ Video Source")
    app_mode = st.sidebar.radio("Select Source", ["Live Webcam", "Upload Video"], label_visibility="collapsed")
    
    source = 0
    if app_mode == "Upload Video":
        uploaded_video = st.sidebar.file_uploader("Upload a Video file", type=['mp4', 'avi', 'mov'])
        if uploaded_video is not None:
            with open("temp_vid.mp4", "wb") as tfile:
                tfile.write(uploaded_video.read())
            source = "temp_vid.mp4"
        else:
            source = None

    st.sidebar.markdown("<br>### 🎛️ Sensitivity Parameters", unsafe_allow_html=True)
    ear_thresh = st.sidebar.slider("EAR Threshold (Lower = stricter)", min_value=0.15, max_value=0.35, value=0.25, step=0.01)
    frame_thresh = st.sidebar.slider("Warning Frames Trigger", min_value=5, max_value=60, value=20)
    
    st.sidebar.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # ------------- DASHBOARD LAYOUT -------------
    st.title("Driver Drowsiness Monitor")
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem;'>Real-time cognitive tracking using high-speed facial landmarks.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    main_col1, main_col2 = st.columns([2.5, 1], gap="large")
    
    with main_col1:
        stframe = st.empty()
        
        # Elegant controls
        c1, c2, c3, c4 = st.columns(4)
        with c2:
            start_btn = st.button("▶️ START SCAN")
        with c3:
            stop_btn = st.button("⏹️ STANDBY")
            
    with main_col2:
        status_placeholder = st.empty()
        status_placeholder.markdown("<div class='status-card status-idle'><h2 style='margin:0'>💤 STANDBY</h2><p style='margin:0; opacity:0.8'>Click Start</p></div>", unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            ear_metric = st.empty()
            blink_metric = st.empty()
        with m_col2:
            alert_metric = st.empty()
            fps_metric = st.empty()
            
        rest_stop_placeholder = st.empty()
            
        def update_metrics(e=0.00, b=0, a=0, f=0.0):
            with ear_metric.container():
                st.metric(label="Live EAR", value=f"{e:.3f}")
            with blink_metric.container():
                st.metric(label="Blinks", value=b)
            with alert_metric.container():
                st.metric(label="Alerts", value=a)
            with fps_metric.container():
                st.metric(label="FPS", value=f"{f:.1f}")
                
        update_metrics()

    # ------------- PROCESSING LOGIC -------------
    if 'run_system' not in st.session_state:
        st.session_state.run_system = False
        
    if start_btn:
        st.session_state.run_system = True
    if stop_btn:
        st.session_state.run_system = False
        
    if st.session_state.run_system and source is not None:
        cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            st.error("Cannot access camera. Please check operating system permissions.")
            return
            
        counter = 0
        blink_count = 0
        drowsy_score = 0
        prev_time = time.time()
        
        critical_start_time = None
        
        while st.session_state.run_system:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            if len(frame.shape) < 2 or frame.shape[0] < 10:
                continue
                
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # FPS Tracker
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # AI Inference
            detection_result = detector.process(rgb)
            
            current_status = "No Face Found"
            current_ear = 0.0
            
            if detection_result.multi_face_landmarks:
                current_status = "Active"
                face_landmarks = detection_result.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                left_coords = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_EYE]
                right_coords = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_EYE]
                
                left_EAR = calculate_EAR(left_coords)
                right_EAR = calculate_EAR(right_coords)
                ear = (left_EAR + right_EAR) / 2.0
                current_ear = ear
                
                # Draw sleek annotations
                left_hull = cv2.convexHull(np.array(left_coords))
                right_hull = cv2.convexHull(np.array(right_coords))
                cv2.drawContours(frame, [left_hull], -1, (0, 255, 100), 1) # Soft green outline
                cv2.drawContours(frame, [right_hull], -1, (0, 255, 100), 1)
                
                if ear < ear_thresh:
                    counter += 1
                    cv2.putText(frame, "EYES CLOSED", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
                    
                    if counter >= frame_thresh:
                        current_status = "Drowsy"
                        cv2.putText(frame, "CRITICAL: WAKE UP!", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        
                        if counter == frame_thresh:
                            drowsy_score += 1
                            play_alert_sound()
                else:
                    if 2 <= counter < frame_thresh:
                        blink_count += 1
                    counter = 0
                    current_status = "Active"
                        
            
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if frame_rgb is not None and frame_rgb.size > 0:
    stframe.image(
        frame_rgb,
        channels="RGB",
        use_container_width=True
    )
else:
    st.error("Could not display the video frame.")
    break

            
            # Status & Metric rendering
            if current_status == "Active":
                html = "<div class='status-card status-active'><h2 style='margin:0'>✅ SECURE</h2><p style='margin:0; opacity:0.9'>Monitoring active</p></div>"
                critical_start_time = None
                rest_stop_placeholder.empty()
            elif current_status == "Drowsy":
                html = "<div class='status-card status-warning'><h2 style='margin:0'>🚨 CRITICAL</h2><p style='margin:0; opacity:0.9'>Wake up immediately!</p></div>"
                if critical_start_time is None:
                    critical_start_time = time.time()
                elif (time.time() - critical_start_time) > 3.0:
                    rest_stop_placeholder.markdown("""
                        <div class='rest-stop-card'>
                            <h3 style='margin-top:0; color:#38bdf8;'>⚠️ Nearest Safe Stops</h3>
                            <ul style='color:#f8fafc; padding-left:20px; margin-bottom:10px;'>
                                <li>☕ Costo Coffee - 1.2 miles</li>
                                <li>⛽ Shell Service Station - 2.5 miles</li>
                                <li>🛏️ Rest Area Southbound - 4.1 miles</li>
                            </ul>
                            <button class='emergency-btn'>📞 Call Emergency Contact</button>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                html = "<div class='status-card' style='background: linear-gradient(135deg, #1e293b, #334155);'><h2 style='margin:0; color:#94a3b8;'>👤 SCANNING</h2><p style='margin:0; opacity:0.6'>Position face...</p></div>"
                critical_start_time = None
                rest_stop_placeholder.empty()
                
            status_placeholder.markdown(html, unsafe_allow_html=True)
            update_metrics(current_ear, blink_count, drowsy_score, fps)

        cap.release()
        rest_stop_placeholder.empty()
        status_placeholder.markdown("<div class='status-card status-idle'><h2 style='margin:0'>💤 STANDBY</h2><p style='margin:0;opacity:0.8'>Process inactive</p></div>", unsafe_allow_html=True)
        update_metrics()


main_app()
