import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:/Users/91774/Downloads/weights/best.pt")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def detect_video(video_path, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Use actual FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection with optimized settings
        results = model(frame, imgsz=640, conf=0.3, iou=0.3, device=device)  

        # Use the built-in YOLO visualization
        frame = results[0].plot()

        out.write(frame)
        cv2.imshow("YOLOv8 Video Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_path}")

if __name__ == "__main__":
    
    video_path = r"C:\Users\91774\Downloads\VID-20250219-WA0004.mp4"
    # video_path = r"C:\Users\91774\Downloads\WhatsApp Video 2025-02-13 at 4.24.54 PM.mp4"
    detect_video(video_path)

# import torch
# print("Torch version:", torch.__version__)
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Version:", torch.version.cuda)
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

