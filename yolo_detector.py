from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2


# Load YOLO model
model = YOLO("yolov8n.pt")


def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf
        if class_name == "cat":
            annotator.box_label(
            box=coordinator,
            label=class_name,
            color=colors(0, True)
        )
    # Draw bounding box
    # annotator.box_label(
    #     box=coordinator, label=class_name, color=colors(0, True)
    # )

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame)
    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    # video_writer = cv.VideoWriter(
    #     video_path + "_demo.avi", cv.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)
    # )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()
        
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX 
  
            # Use putText() method for 
            # inserting text on video 
            text = 'Patcharamon-Clicknext-Internship-2024'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            color = (0, 0, 255)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = frame.shape[1] - text_width - 10
            y = 30
            cv2.putText(frame,  
                        text,  
                        (x, y),  
                        font,  
                        font_scale,  
                        color,  
                        thickness,  
                        cv2.LINE_AA)
            
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)

            # Write result to video
            # video_writer.write(frame_result)

            # Show result
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Video", frame_result)
            cv2.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    # video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
