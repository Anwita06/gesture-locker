import cv2
import mediapipe as mp
import json
import numpy as np
from deepface import DeepFace
import os
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

DB_FILE = "db.json"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,      # Faster model for CPU/real-time streaming
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

def get_transparent_canvas(canvas):
    if canvas is None:
        return ""
    bgra = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    mask = np.max(canvas, axis=2) <= 10
    bgra[mask, 3] = 0
    _, buffer = cv2.imencode('.png', bgra)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

# Database Operations
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

# Helper Functions
def count_fingers(lm, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in range(21)]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = 0

    # Robust thumb detection based on hand horizontal orientation (index MCP vs pinky MCP)
    if pts[5][0] < pts[17][0]:  # Thumb should extend to the left (e.g. palm-up right hand or palm-down left hand)
        if pts[4][0] < pts[3][0]:
            up += 1
    else:  # Thumb should extend to the right (e.g. palm-up left hand or palm-down right hand)
        if pts[4][0] > pts[3][0]:
            up += 1

    for tip, pip in zip(tips, pips):
        if pts[tip][1] < pts[pip][1]:
            up += 1
    return up

def cosine_similarity(a, b):
    a = np.array(a); b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def decode_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + base64_str

# Global Active Session States
class ActiveSession:
    def __init__(self):
        self.reset_registration()
        self.reset_login()

    def reset_registration(self):
        self.reg_face_encoding = None
        self.reg_static_gestures = []
        self.reg_dynamic_canvas = None
        self.reg_dynamic_prev = (0, 0)
        self.reg_latest_landmarks = None

    def reset_login(self):
        self.login_face_ok = False
        self.login_static_ok = False
        self.login_dynamic_ok = False
        self.login_static_index = 0
        self.login_dynamic_canvas = None
        self.login_dynamic_prev = (0, 0)
        self.login_latest_landmarks = None

session_state = ActiveSession()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    base64_str = data.get('image')
    action = data.get('action') # 'idle', 'draw_register', 'draw_login'
    
    if not base64_str:
        return jsonify({'error': 'No image data'}), 400

    frame = decode_image(base64_str)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400
        
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    annotated_frame = frame.copy()
    status_msg = ""
    fingers_count = -1
    
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        # Draw hand landmarks in original MediaPipe styling
        mp_draw.draw_landmarks(annotated_frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        # Count fingers
        fingers_count = count_fingers(lm, w, h)
        status_msg = f"Hand detected: {fingers_count} fingers up"
        
        # Store for dynamic capturing
        pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in range(21)]
        tip_index = pts[8]
        tip_middle = pts[12]
        
        index_up = pts[8][1] < pts[6][1]
        middle_up = pts[12][1] < pts[10][1]
        ring_up = pts[16][1] < pts[14][1]
        pinky_up = pts[20][1] < pts[18][1]
        
        # Handle Dynamic registration drawing
        if action == 'draw_register':
            session_state.reg_latest_landmarks = lm
            if session_state.reg_dynamic_canvas is None or session_state.reg_dynamic_canvas.shape[:2] != (h, w):
                session_state.reg_dynamic_canvas = np.zeros((h, w, 3), np.uint8)
            
            canvas = session_state.reg_dynamic_canvas
            prev = session_state.reg_dynamic_prev
            
            if index_up and not middle_up:
                cv2.circle(canvas, tip_index, 10, (0, 0, 255), -1)
                if prev != (0, 0):
                    cv2.line(canvas, prev, tip_index, (0, 0, 255), 6)
                session_state.reg_dynamic_prev = tip_index
                status_msg = "Drawing..."
            elif index_up and middle_up and not ring_up:
                cv2.circle(canvas, tip_middle, 40, (0, 0, 0), -1)
                session_state.reg_dynamic_prev = (0, 0)
                status_msg = "Erasing..."
            elif index_up and middle_up and ring_up:
                canvas[:] = 0
                session_state.reg_dynamic_prev = (0, 0)
                status_msg = "Cleared Canvas"
            else:
                session_state.reg_dynamic_prev = (0, 0)
                status_msg = "Drawing paused"
                
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, canvas, 0.3, 0)
            
        # Handle Dynamic login drawing
        elif action == 'draw_login':
            session_state.login_latest_landmarks = lm
            if session_state.login_dynamic_canvas is None or session_state.login_dynamic_canvas.shape[:2] != (h, w):
                session_state.login_dynamic_canvas = np.zeros((h, w, 3), np.uint8)
                
            canvas = session_state.login_dynamic_canvas
            prev = session_state.login_dynamic_prev
            
            if index_up and not middle_up:
                cv2.circle(canvas, tip_index, 10, (0, 0, 255), -1)
                if prev != (0, 0):
                    cv2.line(canvas, prev, tip_index, (0, 0, 255), 6)
                session_state.login_dynamic_prev = tip_index
                status_msg = "Drawing..."
            elif index_up and middle_up and not ring_up:
                cv2.circle(canvas, tip_middle, 40, (0, 0, 0), -1)
                session_state.login_dynamic_prev = (0, 0)
                status_msg = "Erasing..."
            elif index_up and middle_up and ring_up:
                canvas[:] = 0
                session_state.login_dynamic_prev = (0, 0)
                status_msg = "Cleared Canvas"
            else:
                session_state.login_dynamic_prev = (0, 0)
                status_msg = "Drawing paused"
                
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, canvas, 0.3, 0)
    else:
        status_msg = "No hand detected"
        if action == 'draw_register':
            session_state.reg_dynamic_prev = (0, 0)
            if session_state.reg_dynamic_canvas is not None:
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, session_state.reg_dynamic_canvas, 0.3, 0)
        elif action == 'draw_login':
            session_state.login_dynamic_prev = (0, 0)
            if session_state.login_dynamic_canvas is not None:
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, session_state.login_dynamic_canvas, 0.3, 0)

    # Encode back to base64
    out_b64 = encode_image(annotated_frame)
    frame_index = data.get('frame_index')
    
    return jsonify({
        'status': 'success',
        'message': status_msg,
        'fingers': fingers_count,
        'image': out_b64,
        'frame_index': frame_index
    })

# Discrete Registration Handlers
@app.route('/register_face', methods=['POST'])
def register_face():
    data = request.json
    base64_str = data.get('image')
    if not base64_str:
        return jsonify({'status': 'fail', 'message': 'No image provided'}), 400
        
    frame = decode_image(base64_str)
    try:
        enc = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        session_state.reg_face_encoding = enc
        return jsonify({'status': 'success', 'message': 'Face encoding captured successfully!'})
    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'Face registration error: {str(e)}'}), 400

@app.route('/register_static', methods=['POST'])
def register_static():
    if len(session_state.reg_static_gestures) >= 3:
        return jsonify({'status': 'fail', 'message': 'Already registered 3 static gestures. Please save or reset.'}), 400
        
    data = request.json
    base64_str = data.get('image')
    if not base64_str:
        return jsonify({'status': 'fail', 'message': 'No image provided'}), 400
        
    frame = decode_image(base64_str)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        fingers = count_fingers(lm, w, h)
        if fingers in [1, 2, 3, 4, 5]:
            session_state.reg_static_gestures.append(fingers)
            return jsonify({
                'status': 'success', 
                'message': f'Gesture registered: {fingers} fingers.',
                'count': len(session_state.reg_static_gestures)
            })
        else:
            return jsonify({'status': 'fail', 'message': 'Please show 1-5 fingers clearly.'}), 400
    else:
        return jsonify({'status': 'fail', 'message': 'No hand detected in the camera.'}), 400

@app.route('/register_dynamic_start', methods=['POST'])
def register_dynamic_start():
    session_state.reg_dynamic_canvas = None
    session_state.reg_dynamic_prev = (0, 0)
    session_state.reg_latest_landmarks = None
    return jsonify({'status': 'success', 'message': 'Dynamic drawing started.'})

@app.route('/register_dynamic_stop', methods=['POST'])
def register_dynamic_stop():
    canvas = session_state.reg_dynamic_canvas
    lm = session_state.reg_latest_landmarks
    
    if canvas is None:
        return jsonify({'status': 'fail', 'message': 'No drawing canvas detected.'}), 400
        
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 10)
    
    if len(xs) > 50 and lm is not None:
        pts_path = np.column_stack([xs, ys])[::15]
        # Store dynamic path
        session_state.reg_dynamic_path = pts_path.flatten().tolist()
        
        # Store 15-point pose data
        key_landmarks = [4, 3, 2, 8, 7, 6, 12, 11, 10, 16, 15, 14, 20, 19, 18] 
        pose_data = [lm[i].x for i in key_landmarks] + [lm[i].y for i in key_landmarks]
        session_state.reg_dynamic_pose = pose_data
        
        return jsonify({'status': 'success', 'message': 'Dynamic gesture captured successfully!'})
    else:
        return jsonify({'status': 'fail', 'message': 'Drawing was too short or no hand detected.'}), 400

@app.route('/save_registration', methods=['POST'])
def save_registration():
    data = request.json
    pin = data.get('pin')
    
    if not pin or len(pin) < 4:
        return jsonify({'status': 'fail', 'message': 'Invalid PIN. Must be at least 4 digits.'}), 400
        
    if not session_state.reg_face_encoding:
        return jsonify({'status': 'fail', 'message': 'Face registration incomplete.'}), 400
        
    if len(session_state.reg_static_gestures) != 3:
        return jsonify({'status': 'fail', 'message': 'Must register exactly 3 static gestures.'}), 400
        
    if not getattr(session_state, 'reg_dynamic_path', None):
        return jsonify({'status': 'fail', 'message': 'Dynamic gesture incomplete.'}), 400
        
    db = {
        "admin": {
            "face_encoding": session_state.reg_face_encoding,
            "static_gestures": session_state.reg_static_gestures,
            "dynamic_path": session_state.reg_dynamic_path,
            "dynamic_pose": session_state.reg_dynamic_pose,
            "pin": pin
        }
    }
    save_db(db)
    session_state.reset_registration()
    return jsonify({'status': 'success', 'message': 'Registration saved and complete!'})


# Discrete Login/Verification Handlers
@app.route('/verify_face', methods=['POST'])
def verify_face():
    db_admin = load_db().get("admin")
    if not db_admin or "face_encoding" not in db_admin:
        return jsonify({'status': 'fail', 'message': 'No user registered yet.'}), 400
        
    data = request.json
    base64_str = data.get('image')
    if not base64_str:
        return jsonify({'status': 'fail', 'message': 'No image provided'}), 400
        
    frame = decode_image(base64_str)
    try:
        enc = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        sim = cosine_similarity(enc, db_admin["face_encoding"])
        matched = sim > 0.6
        session_state.login_face_ok = matched
        if matched:
            return jsonify({'status': 'success', 'message': 'Face verification successful!'})
        else:
            return jsonify({'status': 'fail', 'message': 'Face verification failed.'}), 401
    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'Face verify error: {str(e)}'}), 400

@app.route('/verify_static', methods=['POST'])
def verify_static():
    db_admin = load_db().get("admin")
    if not db_admin or "static_gestures" not in db_admin:
        return jsonify({'status': 'fail', 'message': 'No user registered yet.'}), 400
        
    data = request.json
    base64_str = data.get('image')
    if not base64_str:
        return jsonify({'status': 'fail', 'message': 'No image provided'}), 400
        
    frame = decode_image(base64_str)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        fingers = count_fingers(lm, w, h)
        
        required_gestures = db_admin["static_gestures"]
        curr_index = session_state.login_static_index
        expected = required_gestures[curr_index]
        
        if fingers == expected:
            session_state.login_static_index += 1
            if session_state.login_static_index == len(required_gestures):
                session_state.login_static_ok = True
                return jsonify({'status': 'success', 'message': 'All static gestures verified!', 'complete': True})
            else:
                return jsonify({
                    'status': 'success', 
                    'message': f'Gesture step {session_state.login_static_index} matched!',
                    'complete': False,
                    'next_step': session_state.login_static_index + 1
                })
        else:
            session_state.login_static_index = 0  # reset sequence on fail
            return jsonify({'status': 'fail', 'message': 'Wrong gesture! Sequence reset. Start from step 1.'}), 401
    else:
        return jsonify({'status': 'fail', 'message': 'No hand detected in the camera.'}), 400

@app.route('/verify_dynamic_start', methods=['POST'])
def verify_dynamic_start():
    session_state.login_dynamic_canvas = None
    session_state.login_dynamic_prev = (0, 0)
    session_state.login_latest_landmarks = None
    return jsonify({'status': 'success', 'message': 'Dynamic drawing started.'})

@app.route('/verify_dynamic_stop', methods=['POST'])
def verify_dynamic_stop():
    db_admin = load_db().get("admin")
    if not db_admin or "dynamic_pose" not in db_admin:
        return jsonify({'status': 'fail', 'message': 'No user registered yet.'}), 400
        
    canvas = session_state.login_dynamic_canvas
    lm = session_state.login_latest_landmarks
    
    if canvas is None:
        return jsonify({'status': 'fail', 'message': 'No drawing canvas detected.'}), 400
        
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 10)
    
    if len(xs) > 50 and lm is not None:
        # Check dynamic path
        pts_live = np.column_stack([xs, ys])[::15]
        path_ok = len(pts_live) > 5
        
        # Check hand pose
        key_landmarks = [4, 3, 2, 8, 7, 6, 12, 11, 10, 16, 15, 14, 20, 19, 18]
        live_pose = np.array([lm[i].x for i in key_landmarks] + [lm[i].y for i in key_landmarks])
        saved_pose = np.array(db_admin.get("dynamic_pose", [0.0] * 30))
        
        pose_distance = np.linalg.norm(live_pose - saved_pose)
        pose_ok = pose_distance < 0.5
        
        matched = path_ok and pose_ok
        session_state.login_dynamic_ok = matched
        
        if matched:
            return jsonify({'status': 'success', 'message': 'Dynamic gesture verified successfully!'})
        else:
            return jsonify({
                'status': 'fail', 
                'message': f'Dynamic gesture match failed (Distance: {pose_distance:.3f})'
            }), 401
    else:
        return jsonify({'status': 'fail', 'message': 'Drawing was too short or no hand detected.'}), 400

@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    db_admin = load_db().get("admin")
    if not db_admin or "pin" not in db_admin:
        return jsonify({'status': 'fail', 'message': 'No user registered yet.'}), 400
        
    data = request.json
    pin = data.get('pin')
    
    if pin == db_admin["pin"]:
        # Verify other steps
        steps = {
            'face': session_state.login_face_ok,
            'static': session_state.login_static_ok,
            'dynamic': session_state.login_dynamic_ok
        }
        
        if all(steps.values()):
            session_state.reset_login()
            return jsonify({
                'status': 'success', 
                'message': '🎉 ACCESS GRANTED! LOCKER OPENED!', 
                'unlocked': True
            })
        else:
            failed_steps = [k for k, v in steps.items() if not v]
            return jsonify({
                'status': 'fail', 
                'message': f'Locker Locked! Unverified steps: {", ".join(failed_steps)}',
                'unlocked': False
            }), 403
    else:
        return jsonify({'status': 'fail', 'message': 'Incorrect PIN.'}), 401

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_state.reset_registration()
    session_state.reset_login()
    return jsonify({'status': 'success', 'message': 'Session states reset.'})

@app.route('/get_db_status', methods=['GET'])
def get_db_status():
    db_admin = load_db().get("admin")
    return jsonify({
        'registered': db_admin is not None,
        'has_face': db_admin is not None and "face_encoding" in db_admin,
        'has_static': db_admin is not None and "static_gestures" in db_admin,
        'has_dynamic': db_admin is not None and "dynamic_pose" in db_admin
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
