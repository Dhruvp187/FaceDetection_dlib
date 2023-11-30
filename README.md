# Face Detection/Recognition with dlib

Welcome to the Face Detection/Recognition repository! This project utilizes the dlib library to detect and recognize faces in images and videos.

## Repository Structure:

- **InputFolderForImageAndVideoFiles:** This folder contains the input images and videos on which you want to perform face detection and recognition.

- **OutputFolder:** The output of the code will be saved in this folder. It includes annotated images/videos with detected faces.

- **FaceRecognition_dlib.py:** The main Python script that implements face detection and recognition using the dlib model.

## Dependencies:

Make sure to install the required dependencies before running the code.

```bash
pip install opencv-python dlib matplotlib
```

## Instructions to Run:

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/face-detection-recognition.git
cd face-detection-recognition
```

2. **Run the Code:**

```bash
python FaceRecognition_dlib.py
```

The script will process the images and videos in the input folder and save the results in the output folder.

## Scope for Use:

This repository can be used for various applications, including:

1. **Security Systems:** Implement face detection and recognition for access control systems.

2. **Image and Video Annotation:** Automatically annotate faces in images and videos.

3. **Social Media Analysis:** Analyze and count faces in social media posts or videos.

4. **Human-Computer Interaction:** Use face recognition for user authentication.

## Code Functionality:

The provided Python script, `FaceRecognition_dlib.py`, is designed to perform face detection and recognition using the dlib library. Here's an overview of how the code operates:

### Image Processing:

1. **Image Loading:**
   - The script reads images from the specified `InputFolderForImageAndVideoFiles` directory.

2. **Face Detection:**
   - Utilizing the dlib `get_frontal_face_detector` function, faces are detected in the grayscale version of the image.

3. **Annotating Detected Faces:**
   - Detected faces are outlined with rectangles and saved in the `OutputFolder` directory.

4. **User Input:**
   - The user is prompted to enter the actual number of faces in the image for later comparison.

5. **Comparison and Logging:**
   - The script compares the detected faces with the user-provided actual count.
   - The results, including the detected faces, actual faces, and differences, are logged.

### Video Processing:

1. **Video Loading:**
   - Similarly, the script reads videos from the specified `InputFolderForImageAndVideoFiles` directory.

2. **Face Detection in Frames:**
   - Frames from the video are processed individually for face detection.

3. **Annotating Detected Faces in Frames:**
   - Detected faces in each frame are outlined, and the processed frames are saved in a new video file.

4. **User Input for Video:**
   - The user is prompted to enter the total actual number of faces in the video.

5. **Comparison and Logging for Video:**
   - The script compares the detected faces in each frame with the user-provided total count.
   - The results, including the detected faces, actual faces, and differences over frames, are logged.

### Plotting Results:

1. **Image Results Plotting:**
   - If image processing occurs, a plot is generated showing the number of detected faces and the actual number of faces for each processed image.

2. **Video Results Plotting:**
   - If video processing occurs, a plot is generated showing the number of detected faces in each frame and the total actual number of faces in the video.

## Additional Expansion:

1. **Using Different dlib Models:**
   Extend the capabilities of face recognition by utilizing various dlib models. For instance, you can incorporate the `shape_predictor_68_face_landmarks` model to identify and analyze facial landmarks. This model provides 68 specific points on the face, opening up possibilities for more detailed analysis and applications such as emotion recognition or facial expression analysis.

   To implement facial landmarks detection, consider modifying the code to incorporate the `shape_predictor_68_face_landmarks` model. This can enhance the precision and functionality of the face recognition system.

2. **Integrating Deep Learning Models:**
   Explore and integrate deep learning models for more accurate face recognition.

3. **GUI Implementation:**
   Develop a graphical user interface for easy interaction.

4. **Real-Time Processing:**
   Modify the code to perform face detection and recognition in real-time using a webcam.

## Improving Accuracy with MMOD Model:

For those seeking enhanced accuracy at the cost of processing time, the dlib library offers the `mmod` (max-margin object detection) model. The `mmod` model is known for its superior accuracy in face detection compared to the traditional models, albeit at a higher computational cost.

To improve accuracy using the `mmod` model, consider the following steps:

- **Download the Pre-trained Model:**
  Obtain the pre-trained `mmod_human_face_detector.dat` model from the dlib model zoo.

- **Update the Code:**
  Modify the existing code to load and use the `mmod` model for face detection. Update the `detector` initialization to utilize the `mmod` model.

```python
# Create a face detector using the mmod model
detector = dlib.cnn_face_detection_model_v1("path/to/mmod_human_face_detector.dat")
```

- **Runtime Considerations:**
  Be aware that using the `mmod` model may significantly increase the processing time, especially on large images or videos.

## Documentation Improvement:

Enhance the documentation to guide users on incorporating different dlib models for face recognition. Provide clear instructions on how to download and use the `shape_predictor_68_face_landmarks` model and the `mmod` model for improved accuracy.

We welcome contributions and suggestions to make these additions more versatile and useful! If you encounter any issues or have ideas for enhancement, please open an issue or submit a pull request.

Happy coding! 🚀
