import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Function to process the first frame of a video and play the video for a specific duration
def process_and_play_video(window_name, video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        # Resize the frame to 480x480
        frame_resized = cv2.resize(frame, (480, 480))

        # Use YOLO to detect objects in the first frame
        results = model.predict(frame_resized)

        # Extract and print the vehicle count
        vehicle_count = 0
        for result in results:
            for box in result.boxes.data:
                # Assuming class IDs for vehicles (cars, trucks, buses) are 2, 3, 5, 7
                class_id = int(box[5])
                if class_id in [2, 3, 5, 7]:
                    vehicle_count += 1

        # Determine the time duration to play the video based on vehicle count
        if vehicle_count <= 8:
            playback_duration = 4  # seconds
            reason = "Vehicle count <= 8, so playing the video for 4 seconds."
        elif vehicle_count > 8 and vehicle_count <= 12:
            playback_duration = 5  # seconds
            reason = "Vehicle count > 8 and <= 12, so playing the video for 5 seconds."
        else:
            playback_duration = None  # Full video
            reason = "Vehicle count > 12, so playing the full video."

        # Print the video timing and reason
        if playback_duration is not None:
            print(f"Vehicle count in {video_path}: {vehicle_count}")
            print(f"Allocated time for video: {playback_duration} seconds.")
        else:
            print(f"Vehicle count in {video_path}: {vehicle_count}")
            print(f"Allocated time for video: Full video.")

        print(f"Reason: {reason}\n")

        # Display the first frame
        cv2.imshow(window_name, frame_resized)
        cv2.waitKey(0)  # Wait for a key press before continuing

    # Continue playing the rest of the video for the determined duration
    if playback_duration is not None:
        # Calculate the number of frames to display based on the playback duration
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = fps * playback_duration
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to 480x480
            frame_resized = cv2.resize(frame, (480, 480))

            cv2.imshow(window_name, frame_resized)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            frame_count += 1
    else:
        # Play the full video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to 480x480
            frame_resized = cv2.resize(frame, (480, 480))

            cv2.imshow(window_name, frame_resized)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()

# Paths to your video files
video1_path = 'videos/video1.mp4'
video2_path = 'videos/video2.mp4'
video3_path = 'videos/video3.mp4'
video4_path = 'videos/video4.mp4'

# Create windows first so they all appear simultaneously
cv2.namedWindow('Window 1')
cv2.namedWindow('Window 2')
cv2.namedWindow('Window 3')
cv2.namedWindow('Window 4')

# Adjust the positions and sizes to create 4 square boxes
cv2.resizeWindow('Window 1', 480, 480)
cv2.resizeWindow('Window 2', 480, 480)
cv2.resizeWindow('Window 3', 480, 480)
cv2.resizeWindow('Window 4', 480, 480)

# Position windows in a 2x2 grid
cv2.moveWindow('Window 1', 0, 0)  # Top-left corner
cv2.moveWindow('Window 2', 480, 0)  # Top-right corner
cv2.moveWindow('Window 3', 0, 480)  # Bottom-left corner
cv2.moveWindow('Window 4', 480, 480)  # Bottom-right corner

# Process the first frame and play each video
process_and_play_video('Window 1', video1_path)
process_and_play_video('Window 2', video2_path)
process_and_play_video('Window 3', video3_path)
process_and_play_video('Window 4', video4_path)

# Close all OpenCV windows
cv2.destroyAllWindows()


