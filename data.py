# Preprocess data
# Extract frames, faces, and cropped features (eyes, nose, mouth) from videos
# Run download-FaceForensics.py
# then use command python3 data.py to process data
import cv2
from pathlib import Path
from deepface import DeepFace
import dlib
from tqdm import tqdm

def frameExtraction(vidPath, imgPath):
    vidFiles = list(vidPath.glob('*.mp4'))
    print("Found " + str(len(vidFiles)) + " MP4 file(s)")
    FrameInterval = 30
    frameCount = 0
    for file in tqdm(vidFiles, desc="Processing videos"):
        vid = cv2.VideoCapture(str(file))
        if not vid.isOpened():
            print(f"Error: Could not open {file.name}")
            continue
        vidOutputDir = imgPath
        vidOutputDir.mkdir(parents=True, exist_ok=True)
        saveCount = 0
        while True:
            success, image = vid.read()
            if not success:
                break
            if frameCount % FrameInterval == 0:
                frame_path = vidOutputDir / f"frame{frameCount}.jpg"
                cv2.imwrite(str(frame_path), image)
                saveCount+=1
            frameCount += 1
        vid.release()
    print("Frame Extraction Complete")

def faceCrop(frameInput, frameOutput):
    # Read the input image
    imgFiles = list(frameInput.glob('*.jpg'))
    print("Found " + str(len(imgFiles)) + " jpg file(s)")
    faceOutputDir = frameOutput
    faceOutputDir.mkdir(parents=True, exist_ok=True)
    totalFaces = 0
    for currImgFile in tqdm(imgFiles, desc="Cropping faces"):
        # Detect faces
        img = cv2.imread(str(currImgFile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = DeepFace.extract_faces(
            img_path = str(currImgFile),
            detector_backend='retinaface',
            enforce_detection=False,
            align=True
        )
        for faceIdx, faceObj in enumerate(faces):
            if faceObj['confidence'] <= 0.97:  # check confidence
                continue
            faceArea = faceObj['facial_area']
            x,y,w,h=faceArea['x'], faceArea['y'], faceArea['w'], faceArea['h']
            if w < 50 or h < 50: # face too small
                continue
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                continue
            faceCrop = img[y:y + h, x:x + w]
            original_name = currImgFile.stem
            video_folder = currImgFile.parent.name
            face_filename = f"{video_folder}_{original_name}_face{faceIdx}.jpg"
            face_path = faceOutputDir / face_filename
            cv2.imwrite(str(face_path), faceCrop)
            totalFaces+=1
    print("Face Crop Complete - Extracted " + str(totalFaces) + " faces from " + str(len(imgFiles)))

def featuresCrop(frameInput, frameOutput):
    imgFiles = list(frameInput.glob('*.jpg'))
    print("Found " + str(len(imgFiles)) + " jpg file(s)")

    featureOutputDirLeft = frameOutput/"leftEye"
    featureOutputDirLeft.mkdir(parents=True, exist_ok=True)
    featureOutputDirRight = frameOutput/"rightEye"
    featureOutputDirRight.mkdir(parents=True, exist_ok=True)
    featureOutputDirNose = frameOutput/"nose"
    featureOutputDirNose.mkdir(parents=True, exist_ok=True)
    featureOutputDirMouth = frameOutput/"mouth"
    featureOutputDirMouth.mkdir(parents=True, exist_ok=True)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    totalEyes = 0
    totalNoses = 0
    totalMouths = 0
    for currImgFile in tqdm(imgFiles, desc="Cropping features"):
        img = cv2.imread(str(currImgFile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    
        original_name = currImgFile.stem
        video_folder = currImgFile.parent.name

        faces = detector(gray, 1)
        for face_idx, face in enumerate(faces):
            landmarks = predictor(gray, face)
            # Left Eye Detection
            left_eye_points = []
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye_points.append((x, y))
            if left_eye_points:
                x_coords = [p[0] for p in left_eye_points]
                y_coords = [p[1] for p in left_eye_points]
                padding = 10
                x_min = max(0, min(x_coords) - padding)
                y_min = max(0, min(y_coords) - padding)
                x_max = min(img.shape[1], max(x_coords) + padding)
                y_max = min(img.shape[0], max(y_coords) + padding)
                left_eye_crop = img[y_min:y_max, x_min:x_max]
                if left_eye_crop.size > 0:
                    eye_filename = f"{video_folder}_{original_name}_face{face_idx}_left_eye.jpg"
                    cv2.imwrite(str(featureOutputDirLeft / eye_filename), left_eye_crop)
                    totalEyes += 1
            
            # Right Eye Detection
            right_eye_points = []
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye_points.append((x, y))
            if right_eye_points:
                x_coords = [p[0] for p in right_eye_points]
                y_coords = [p[1] for p in right_eye_points]
                padding = 10
                x_min = max(0, min(x_coords) - padding)
                y_min = max(0, min(y_coords) - padding)
                x_max = min(img.shape[1], max(x_coords) + padding)
                y_max = min(img.shape[0], max(y_coords) + padding)
                right_eye_crop = img[y_min:y_max, x_min:x_max]
                if right_eye_crop.size > 0:
                    eye_filename = f"{video_folder}_{original_name}_face{face_idx}_right_eye.jpg"
                    cv2.imwrite(str(featureOutputDirRight / eye_filename), right_eye_crop)
                    totalEyes += 1

            # Nose Detection
            nose_points = []
            for n in range(27, 36):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                nose_points.append((x, y))
            if nose_points:
                x_coords = [p[0] for p in nose_points]
                y_coords = [p[1] for p in nose_points]
                padding = 15
                x_min = max(0, min(x_coords) - padding)
                y_min = max(0, min(y_coords) - padding)
                x_max = min(img.shape[1], max(x_coords) + padding)
                y_max = min(img.shape[0], max(y_coords) + padding)
                nose_crop = img[y_min:y_max, x_min:x_max]
                if nose_crop.size > 0:
                    nose_filename = f"{video_folder}_{original_name}_face{face_idx}_nose.jpg"
                    cv2.imwrite(str(featureOutputDirNose / nose_filename), nose_crop)
                    totalNoses += 1

            # Mouth Detection
            mouth_points = []
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                mouth_points.append((x, y))
            if mouth_points:
                x_coords = [p[0] for p in mouth_points]
                y_coords = [p[1] for p in mouth_points]
                padding = 10
                x_min = max(0, min(x_coords) - padding)
                y_min = max(0, min(y_coords) - padding)
                x_max = min(img.shape[1], max(x_coords) + padding)
                y_max = min(img.shape[0], max(y_coords) + padding)
                mouth_crop = img[y_min:y_max, x_min:x_max]
                if mouth_crop.size > 0:
                    mouth_filename = f"{video_folder}_{original_name}_face{face_idx}_mouth.jpg"
                    cv2.imwrite(str(featureOutputDirMouth / mouth_filename), mouth_crop)
                    totalMouths += 1
    print(f"Features Crop Complete - Extracted {totalEyes} eyes, {totalNoses} noses, and {totalMouths} mouths from {len(imgFiles)}")

origVidPath = Path("FFData/original_sequences/youtube/c23/videos")
origImgPath = Path("Frames/original")
frameExtraction(origVidPath, origImgPath)
manipVidPath = Path("FFData/manipulated_sequences/Deepfakes/c23/videos")
manipImgPath = Path("Frames/manipulated")
frameExtraction(manipVidPath, manipImgPath)

origFramePath = Path("Frames/original")
origFrameOut = Path("CroppedFaces/original")
faceCrop(origFramePath, origFrameOut)
manipFramePath = Path("Frames/manipulated")
manipFrameOut = Path("CroppedFaces/manipulated")
faceCrop(manipFramePath, manipFrameOut)

origFramePath = Path("CroppedFaces/original")
origFrameOut = Path("Features/original")
featuresCrop(origFramePath, origFrameOut)
manipFramePath = Path("CroppedFaces/manipulated")
manipFrameOut = Path("Features/manipulated")
featuresCrop(manipFramePath, manipFrameOut)
