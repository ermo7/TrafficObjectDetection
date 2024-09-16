from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import math

app = Flask(__name__)

# Defining folders for uploads and processed videos
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed'

# Create the directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Declare the tracker variable globally.. That resets tracker for each upload.
tracker = None

# he color map for different vehicle types. For each model.
label_colors = {
    'ambulance': (255, 0, 0),  # Red
    'army vehicle': (0, 128, 0),  # Dark Green
    'auto rickshaw': (255, 255, 0),  # Yellow
    'bicycle': (0, 0, 255),  # Blue
    'bus': (255, 165, 0),  # Orange
    'car': (0, 255, 0),  # Green
    'minibus': (173, 216, 230),  # Light Blue
    'minivan': (75, 0, 130),  # Indigo
    'motorbike': (255, 153, 255),  # Light Pink
    'motorcycle': (255, 153, 255),  # Light Pink
    'pickup': (0, 255, 255),  # Cyan
    'policecar': (0, 0, 0),  # Black
    'rickshaw': (0, 255, 127),  # Spring Green
    'scooter': (255, 105, 180),  # Hot Pink
    'suv': (128, 0, 128),  # Purple
    'three wheelers -CNG-': (139, 69, 19),  # Brown
    'truck': (128, 0, 0),  # Maroon
    'van': (0, 191, 255),  # Deep Sky Blue
    'wheelbarrow': (218, 165, 32),  # Goldenrod
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global tracker  #  the global tracker variable
    logs = []  # LOG LIST

    if request.method == 'POST':
        try:
            video_file = request.files['video']
            model_choice = request.form['model_choice']
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            logs.append(f"Uploaded video: {video_file.filename}")

            # Reset the SORT tracker for each new video
            tracker = Sort(max_age=7, min_hits=1, iou_threshold=0.4)  # Reinitialize the tracker
            logs.append("Tracker reset for the new video.")

            # Update model paths to match the actual file names in your directory
            model_path = get_model_path(model_choice, logs)

            if not model_path:
                return redirect(url_for('upload_file'))

            # Load the YOLO model
            model = YOLO(model_path)
            logs.append("Model loaded successfully.")

            # Process the video using the selected model
            processed_video_path, report = process_video(video_path, model, logs)
            if not processed_video_path:
                logs.append("Failed to process video.")
                flash("Failed to process video.", "danger")
                return redirect(url_for('upload_file'))

            return render_template('processed.html', filename=os.path.basename(processed_video_path), report=report, logs=logs)

        except Exception as e:
            logs.append(f"Error: {e}")
            print(f"Error: {e}")
            flash("An error occurred while processing the video.", "danger")
            return redirect(url_for('upload_file'))

    # For GET requests, render the upload form
    return render_template('upload.html')


def get_model_path(model_choice, logs):
    """Helper function to determine the model path based on the model choice."""
    if model_choice == 'YOLOv8n_PreTrained_Model':
        return "YOLOv8n_PreTrained_Model.pt"
    elif model_choice == 'YOLOv8n_Custom_Trained_Model':
        return "YOLOv8n_Custom_Trained_Model.pt"
    elif model_choice == 'YOLOv8m':
        return "YOLOv8m.pt"
    elif model_choice == 'YOLOv8l':
        return "YOLOv8l.pt"
    elif model_choice == 'YOLOv8n_Custom_One Class_Trained_Model':
        return "YOLOv8n_Custom_One Class_Trained_Model.pt"
    else:
        logs.append("Error: Invalid model choice")
        flash("Invalid model choice.", "danger")
        return None


def calculate_traffic_density_level(traffic_density):
    # Adjust these thresholds based on the expected number of vehicles in each segment
    if traffic_density > 50:  # Example: More than 30 vehicles in 10 seconds is "Super Heavy Traffic"
        return {"level": "Super Heavy Traffic", "color": "red"}
    elif traffic_density > 25:  # Example: 10-20 vehicles in 10 seconds is "Heavy Traffic"
        return {"level": "Heavy Traffic", "color": "orange"}
    elif traffic_density > 15:  # Example: 10-20 vehicles in 10 seconds is "Moderate Traffic"
        return {"level": "Moderate Traffic", "color": "yellow"}
    else:  # Less than or equal to 10 vehicles in 10 seconds is "Light Traffic"
        return {"level": "Light Traffic", "color": "green"}

    #Traffic Density Report: The report summarizes the traffic conditions (light, moderate, heavy)
    # and provides details on the types and number of vehicles detected in each segment.

def process_video(video_path, model, logs):
    global tracker  # Use the global tracker variable
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logs.append("Error: Could not open video.")
        return None, {}

    logs.append("Video opened successfully.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the number of frames in a 10-second segment
    frames_per_segment = int(fps * 10)  # Number of frames in each 10-second segment

    #  Manually measure the pixel length of a known object in your video
    reference_object_real_length = 5
    #reference_object_real_length- > This is a manually defined value representing the real-world length of a known object (in meters).
    # For example, in this code, it's set to 5 meters, which could be the length of a car.
    reference_object_pixel_length = 40  # Measure this in pixels in your video
    #reference_object_pixel_length: This is the corresponding
    # length of that object in pixels as measured in the video (40 pixels).

    #Pixels_to_meters: The conversion factor is then calculated as the ratio of the real-world length to the pixel length.
    # This means that each pixel in the video corresponds to 0.125 meters in the real world
    # (since 5 meters / 40 pixels = 0.125 meters per pixel).


    #  the pixel-to-meter conversion factor
    pixels_to_meters = reference_object_real_length / reference_object_pixel_length
    logs.append(f"Pixels to meters conversion factor: {pixels_to_meters}")

    output_video_path = os.path.join(PROCESSED_FOLDER, 'Annotated_' + os.path.basename(video_path))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    logs.append("VideoWriter initialized.")

    frame_count = 0
    segment_metrics = []  # List to store metrics for each 10-second segment

    tracked_ids = {}  # Dictionary to keep track of labels for each unique ID
    unique_ids = set()  # Set to store unique object IDs

    previous_positions_times = {}  # Store the previous position and time of each object
    current_segment = initialize_segment_metrics(frame_count)  # Initialize current segment metrics

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps  # Calculation of current time in seconds based on frame count

        # Running inference on the current frame with the selected model and a confidence threshold
        results = model(frame, conf=0.3)  # Setting a confidence threshold of 0.3

        detections = []
        labels_list = []  # To store labels corresponding to detections

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            label_indexes = result.boxes.cls.cpu().numpy().astype(int)  # Class indexes

            for box, confidence, label_idx in zip(boxes, confidences, label_indexes):
                if confidence >= 0.3:  # Adjust the threshold if necessary
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([x1, y1, x2, y2, confidence])
                    labels_list.append(result.names[label_idx])  # Store the label

        # Update the tracker with the current frame detection...
        if len(detections) > 0:
            tracked_objects = tracker.update(np.array(detections))
        else:
            tracked_objects = []

        for idx, obj in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = map(int, obj[:5])

            # Calculate the center of the bounding box, it helps to calculate the distance/meter values for speed
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Add the ID to the set of unique IDs for the current segment
            current_segment['unique_ids'].add(obj_id)

            # Check if we have a previous position and time for this ID
            if obj_id in previous_positions_times:
                prev_x, prev_y, prev_time = previous_positions_times[obj_id]

                # Calculate the distance traveled in pixels
                distance_pixels = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

                # Convert the distance to meters
                distance_meters = distance_pixels * pixels_to_meters

                # Calculate the time elapsed since the object was last seen
                time_elapsed = current_time - prev_time

                if time_elapsed > 0:
                    # Calculate speed in meters per second
                    speed_mps = distance_meters / time_elapsed

                    # Convert speed to km/h
                    speed_kmph = speed_mps * 4

                    # Store the speed for the current segment
                    current_segment['speeds'].append(speed_kmph)

            # Update the previous position and time with the current center and time
            previous_positions_times[obj_id] = (center_x, center_y, current_time)

            # Assign the label to the tracked ID
            if obj_id not in tracked_ids:
                tracked_ids[obj_id] = labels_list[idx]  # Assign the first seen label to the ID

                # Increment the vehicle type count for the first occurrence of this ID in the current segment
                label = tracked_ids[obj_id]
                if label in current_segment['vehicle_types']:
                    current_segment['vehicle_types'][label] += 1
                else:
                    current_segment['vehicle_types'][label] = 1

            label = tracked_ids[obj_id]  # Retrieve the assigned label

            # Get the corresponding color for the label
            color = label_colors.get(label, (255, 255, 255))  # Default to white if label not found

            # Draw the bounding box with the corresponding color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker box with thickness 3

            # Convert confidence to percentage and annotate the label
            confidence_percentage = int(detections[idx][4] * 100)
            text = f'{label} ID:{obj_id} ({confidence_percentage}%)'
            font_scale = 1.0  # Font scale for readability
            thickness = 2  # Thick text for visibility
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw a filled rectangle for the text background
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # Put the label text above the bounding box
            cv2.putText(frame, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Write the annotated frame to the output video
        out.write(frame)
        frame_count += 1

        # Check if we've reached the end of a 10-second segment
        if frame_count % frames_per_segment == 0:
            finalize_segment_metrics(current_segment, frame_count, frames_per_segment, segment_metrics)
            logs.append(f"Segment {len(segment_metrics)} processed.")

            # Reset metrics for the next segment
            current_segment = initialize_segment_metrics(frame_count)

    cap.release()
    out.release()

    # If there are remaining frames that didn't fill a full 10-second segment, finalize that segment
    if frame_count % frames_per_segment != 0:
        finalize_segment_metrics(current_segment, frame_count, frame_count % frames_per_segment, segment_metrics)
        logs.append(f"Final segment processed.")

    logs.append("Video processing completed.")
    return output_video_path, segment_metrics


def initialize_segment_metrics(start_frame):
    """Initialize and return a new dictionary for segment metrics."""
    return {
        'start_frame': start_frame,
        'traffic_density': 0,
        'total_vehicles': 0,
        'avg_speed': 0,
        'vehicle_types': {},
        'speeds': [],
        'unique_ids': set()
    }

def finalize_segment_metrics(segment, frame_count, frames_per_segment, segment_metrics):
    """Finalize and store metrics for a segment."""
    segment['total_vehicles'] = len(segment['unique_ids'])

    if segment['speeds']:
        segment['avg_speed'] = sum(segment['speeds']) / len(segment['speeds'])
    else:
        segment['avg_speed'] = 0

    # Total vehicles directly as traffic density instead of dividing by frames_per_segment
    segment['traffic_density'] = segment['total_vehicles']  # No division by frames_per_segment


    # Speed Calculation
    # Distance and Speed Calculations:
	# Pixel-to-Meter Conversion: This is calculated using a reference object of known length in the video.
    # Explain that each pixel corresponds to a certain real-world distance (e.g., 0.125 meters per pixel).
	# Speed Estimation: Speed is calculated based on the distance a vehicle travels across frames (converted to meters)
    # and the time elapsed (in seconds). Speeds are stored and averaged for each segment of the video.


# Calculate traffic density level and color based on total vehicles
    density_info = calculate_traffic_density_level(segment['traffic_density'])
    segment['traffic_density_level'] = density_info['level']
    segment['traffic_density_color'] = density_info['color']

    # Store this segment's metrics
    segment_metrics.append(segment)


if __name__ == '__main__':
    app.run(debug=True)
