from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class TrafficSignalSystem:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.vehicle_detectors = [VehicleDetectionTracker() for _ in range(4)]
        self.vehicle_counts = [0] * 4
        self.average_speeds = [0] * 4
        self.signal_states = ['red'] * 4
        self.current_green = 0
        self.last_signal_change = time.time()
        self.frames = [None] * 4
        self.min_green_time = 10
        self.max_green_time = 60
        self.detected_vehicles = [list() for _ in range(4)]

        # Open all video captures with reduced resolution
        self.caps = [cv2.VideoCapture(path) for path in self.video_paths]
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Start signal timing thread
        threading.Thread(target=self.manage_signals, daemon=True).start()

    def manage_signals(self):
        """Manage traffic signal changes based on vehicle counts."""
        while True:
            current_time = time.time()
            if current_time - self.last_signal_change > self.get_green_time(self.vehicle_counts[self.current_green]):
                self.signal_states[self.current_green] = 'red'
                self.current_green = (self.current_green + 1) % 4
                self.signal_states[self.current_green] = 'green'
                self.last_signal_change = current_time
            time.sleep(1)  # Check every second

    def get_green_time(self, vehicle_count):
        """Calculate green time based on vehicle count."""
        if vehicle_count < 5:
            return self.min_green_time
        elif vehicle_count < 15:
            return (self.max_green_time + self.min_green_time) // 2  # Mid range
        else:
            return self.max_green_time

    def process_lane_result(self, result, lane_idx):
        """Process detection results for a lane."""
        self.vehicle_counts[lane_idx] = result["number_of_vehicles_detected"]
        speeds = [v["speed_info"]["kph"] for v in result["detected_vehicles"] if v["speed_info"]["kph"] is not None]
        self.average_speeds[lane_idx] = np.mean(speeds) if speeds else 0
        self.detected_vehicles[lane_idx] = result["detected_vehicles"]

    def detect_vehicles(self, lane_idx):
        """Process frames from a lane for vehicle detection."""
        cap = self.caps[lane_idx]
        frame_skip = 3  # Process every 3rd frame
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if ret:
                if frame_count % frame_skip == 0:  # Only process every nth frame
                    timestamp = datetime.now()
                    result = self.vehicle_detectors[lane_idx].process_frame(frame, timestamp)
                    self.frames[lane_idx] = frame
                    self.process_lane_result(result, lane_idx)
                frame_count += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def generate_frame(self, lane_idx):
        """Generate frames for display in the web interface."""
        while True:
            if self.frames[lane_idx] is not None:
                frame = self.frames[lane_idx].copy()
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Initialize the traffic signal system with video paths
traffic_system = TrafficSignalSystem([
    "https://videos.pexels.com/video-files/854745/854745-hd_1280_720_50fps.mp4",
    "https://videos.pexels.com/video-files/854745/854745-hd_1280_720_50fps.mp4",
    "video.mp4",
    "video.mp4"
])

# Start vehicle detection threads
for i in range(4):
    threading.Thread(target=traffic_system.detect_vehicles, args=(i,), daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:lane_idx>')
def video_feed(lane_idx):
    return Response(traffic_system.generate_frame(lane_idx),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signal_data')
def signal_data():
    """API to send current signal data for each lane."""
    signal_info = []
    for i in range(4):
        signal_info.append({
            "lane": i + 1,
            "vehicle_count": traffic_system.vehicle_counts[i],
            "signal": traffic_system.signal_states[i]
        })
    return jsonify(signal_info)

@app.route('/vehicle_logs', methods=['GET'])
def vehicle_logs():
    """API to send vehicle detection data for the React dashboard."""
    logs = {
        "detected_vehicles": [vehicle for lane in traffic_system.detected_vehicles for vehicle in lane]
    }
    return jsonify(logs)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
