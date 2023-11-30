import os
import cv2
import dlib
import time
import matplotlib.pyplot as plt

def process_image(image_path, output_folder, detector):
    detected_faces = []
    print(f"Processing image: {image_path}")
    start_time = time.time()

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    detected_faces.append(len(faces))

    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    # Extracting file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))

    # Saving the processed image to the output folder
    output_path = os.path.join(output_folder, f"{file_name}_detected{file_extension}")
    cv2.imwrite(output_path, img)
    
    # Ask the user for the actual number of faces
    user_input = int(input("Enter the actual number of faces in the image: "))
    

    elapsed_time = time.time() - start_time
    print(f"Output generated: {output_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds\n")
    
    return detected_faces, user_input, True  # Indicate that an image was processed

def process_video(video_path, output_folder, detector):
    frame_count = 0
    detected_faces = []
    print(f"Processing video: {video_path}")
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)

    # Extracting file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(video_path))
    
    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Creating VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, f"{file_name}_detected{file_extension}")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        detected_faces.append(len(faces))

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    print(f"Output generated: {output_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds\n")
    
    return list(range(1, frame_count + 1)), detected_faces, True  # Indicate that a video was processed

def main():
    detected_faces_img = []
    user_input_img = []  # Initialize outside the loop
    image_processed = False
    video_processed = False
    
    # Path to the folder containing images and videos
    folder_path = "D:\Github\FaceRocgnition\dlib\InputFolderForImageAndVideoFIles"

    # Output folder to save processed images and videos
    output_folder = "D:\Github\FaceRocgnition\dlib\OutputFolder"
    os.makedirs(output_folder, exist_ok=True)

    # Create a face detector
    detector = dlib.get_frontal_face_detector()

    # Process images and videos in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            detected_faces, user_input, image_processed = process_image(file_path, output_folder, detector)
            user_input_img.append(user_input)
            detected_faces_img.extend(detected_faces)
            
        elif filename.endswith(".mp4"):
            total_frames, detected_faces_vid, video_processed = process_video(file_path, output_folder, detector)
            # print(len(total_frames), len(detected_faces_vid))
            # Ask the user for the actual number of faces
            user_input_vid = int(input("Enter the total actual number of faces in the video: "))
    
    # Plot the number of detected faces and the actual number over frames for images
    if image_processed:
        plt.plot(detected_faces_img, label='Detected Faces (Images)')
        plt.scatter(range(len(user_input_img)), user_input_img, color='r', label='Actual Faces (Images)')
        plt.xlabel('Image Number')
        plt.ylabel('Number of Faces')
        plt.title('Face Detection in Images')
        plt.legend()
        plt.show()
    
    # Plot the number of detected faces and the actual number over frames for videos
    if video_processed:
        plt.plot(total_frames, detected_faces_vid, label='Detected Faces (Videos)')
        plt.axhline(y=user_input_vid, color='r', linestyle='--', label='Actual Faces (Videos)')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Faces')
        plt.title('Face Detection in Videos')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
