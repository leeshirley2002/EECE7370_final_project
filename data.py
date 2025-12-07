import cv2
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm

def frameExtraction(vidPath, imgPath):
    print(f"Path exists: {vidPath.exists()}")
    vidFiles = list(vidPath.glob('*.mp4'))
    print("Found " + str(len(vidFiles)) + " MP4 file(s)")
    FrameInterval = 30

    for file in tqdm(vidFiles, desc="Processing videos"):
        vid = cv2.VideoCapture(str(file))
        if not vid.isOpened():
            print(f"Error: Could not open {file.name}")
            continue
        vidName = file.stem
        vidOutputDir = imgPath / vidName
        vidOutputDir.mkdir(parents=True, exist_ok=True)
        frameCount = 0
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
    imgFiles = list(frameInput.rglob('*.jpg'))
    print("Found " + str(len(imgFiles)) + " jpg file(s)")
    faceOutputDir = frameOutput
    faceOutputDir.mkdir(parents=True, exist_ok=True)
    totalFaces = 0
    for currImgFile in tqdm(imgFiles, desc="Cropping faces"):
        # Detect faces
        faces = DeepFace.extract_faces(
            img_path = str(currImgFile),
            detector_backend='retinaface',
            enforce_detection=False,
            align=True
        )
        img = cv2.imread(str(currImgFile))
        for faceIdx, faceObj in enumerate(faces):
            if faceObj['confidence'] < 0.97:
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
    imgFiles = list(frameInput.rglob('*.jpg'))
    print("Found " + str(len(imgFiles)) + " jpg file(s)")

    featuresOutputDirLeft = frameOutput/"leftEye"
    featuresOutputDirLeft.mkdir(parents=True, exist_ok=True)
    featuresOutputDirRight = frameOutput/"rightEye"
    featuresOutputDirRight.mkdir(parents=True, exist_ok=True)
    featuresOutputDirNose = frameOutput/"nose"
    featuresOutputDirNose.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    totalEyes = 0
    totalNoses = 0
    for currImgFile in tqdm(imgFiles, desc="Cropping features"):
        img = cv2.imread(str(currImgFile))
        # ENHANCEMENT: Upscale low-resolution images
        h, w = img.shape[:2]
        if w < 200 or h < 200:
            scale = max(200/w, 200/h)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ENHANCEMENT: Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # ENHANCEMENT: Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # Smaller steps for better detection
            minNeighbors=3, 
            minSize=(30, 30)  # Lower minimum for small faces
        )
        
        for face_idx, (fx, fy, fw, fh) in enumerate(faces):
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            roi_color = img[fy:fy+fh, fx:fx+fw]
            
            # Try multiple detection passes with different parameters
            eyes_detected = []
            
            # Pass 1: Aggressive
            eyes1 = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.02, minNeighbors=2, minSize=(10, 10))
            eyes_detected.extend(eyes1)
            
            # Pass 2: Standard
            eyes2 = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
            eyes_detected.extend(eyes2)
            
            # Remove duplicates (eyes detected in both passes)
            unique_eyes = []
            for (ex, ey, ew, eh) in eyes_detected:
                is_duplicate = False
                for (ux, uy, uw, uh) in unique_eyes:
                    # Check if eyes overlap significantly
                    if abs(ex - ux) < 20 and abs(ey - uy) < 20:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_eyes.append((ex, ey, ew, eh))
            # Filter eyes
            valid_eyes = []
            for (ex, ey, ew, eh) in unique_eyes:
                # Eyes must be in upper 70% of face
                if ey > fh * 0.7:
                    continue
                # Eyes should have reasonable aspect ratio (wider than tall or roughly square)
                eye_aspect_ratio = ew / eh
                if eye_aspect_ratio < 0.3 or eye_aspect_ratio > 5.0:
                    continue
                # Eyes should not be too large relative to face
                if ew > fw * 0.6 or eh > fh * 0.5:
                    continue
                valid_eyes.append((ex, ey, ew, eh))
            
            # Sort left to right and take only the 2 leftmost
            valid_eyes = sorted(valid_eyes, key=lambda e: e[0])[:2]
            
            original_name = currImgFile.stem
            video_folder = currImgFile.parent.name
            eye_side="n"
            for eye_idx, (ex, ey, ew, eh) in enumerate(valid_eyes[:2]):
                eye_crop = roi_color[ey:ey+eh, ex:ex+ew]
                if eye_crop.size == 0:
                    continue
                if eye_idx == 0:
                    eye_side = "left"
                else:
                    eye_side = "right"
                eye_filename = f"{video_folder}_{original_name}_face{face_idx}_{eye_side}_eye.jpg"
                if eye_side=="left":
                    output_dir=featuresOutputDirLeft
                elif eye_side=="right":
                    output_dir=featuresOutputDirRight
                else:
                    print("ERROR")
                cv2.imwrite(str(output_dir / eye_filename), eye_crop)
                totalEyes += 1

            # DETECT MOUTH
            # Look in lower half of face only
            mouth_roi_y = int(fh * 0.55)
            mouth_roi = roi_gray[mouth_roi_y:, :]
            mouthDetected = []
            mouths1 = mouth_cascade.detectMultiScale(
                mouth_roi,
                scaleFactor=1.1,
                minNeighbors=15,  # High to avoid false positives
                minSize=(25, 12)
            )
            mouthDetected.extend(mouths1)
            mouths2 = mouth_cascade.detectMultiScale(
                mouth_roi,
                scaleFactor=1.05,
                minNeighbors=7,
                minSize=(20, 10)
            )
            mouthDetected.extend(mouths2)
            # Find the most likely mouth (lowest in face, reasonably centered)
            valid_mouth = None
            if len(mouthDetected) > 0:
                # Filter mouths - should be in lower portion and centered
                for (mx, my, mw, mh) in mouthDetected:
                    # Adjust y coordinate back to full face ROI
                    actual_my = my + mouth_roi_y
                    # Mouth should be in lower 45% of face
                    if actual_my < fh * 0.55:
                        continue
                    # Mouth should be reasonably centered
                    mouth_center_x = mx + mw/2
                    if mouth_center_x < fw * 0.2 or mouth_center_x > fw * 0.8:
                        continue
                    # Take the lowest mouth (most likely to be actual mouth)
                    if valid_mouth is None or actual_my > valid_mouth[1]:
                        valid_mouth = (mx, actual_my, mw, mh)

            # DETECT NOSE
            # Use fixed position if no eyes found
            if len(valid_eyes) >=1:
                if len(valid_eyes) >= 2:
                    avg_eye_y = (valid_eyes[0][1] + valid_eyes[1][1]) // 2
                    avg_eye_h = (valid_eyes[0][3] + valid_eyes[1][3]) // 2
                    nose_top = avg_eye_y + int(avg_eye_h * 1.0)
                else:
                    nose_top = valid_eyes[0][1] + int(valid_eyes[0][3] * 1.2)
                # Determine bottom boundary from mouth
                if valid_mouth is not None:
                    # Nose ends just above mouth (leave small gap)
                    nose_bottom = valid_mouth[1] - int(fh * 0.03)
                else:
                    # No mouth detected - use conservative estimate
                    nose_bottom = int(fh * 0.68)
                # Calculate nose height
                nose_h = nose_bottom - nose_top
                
                # Ensure minimum height
                if nose_h < int(fh * 0.20):
                    # If too small, extend down more
                    nose_bottom = nose_top + int(fh * 0.28)
                    nose_h = nose_bottom - nose_top
                
                # Ensure nose doesn't extend too far down
                if nose_bottom > fh * 0.75:
                    nose_bottom = int(fh * 0.68)
                    nose_h = nose_bottom - nose_top
            else:
                continue
            
            nose_x = int(fw * 0.28)
            nose_w = int(fw * 0.44)
            
            nose_y_end = min(nose_top + nose_h, fh)
            nose_x_end = min(nose_x + nose_w, fw)

            # Validate
            if nose_top < 0 or nose_top >= nose_y_end:
                continue
            if nose_top > fh * 0.55:
                continue

            if nose_x < nose_x_end:
                nose_crop = roi_color[nose_top:nose_y_end, nose_x:nose_x_end]
                if nose_crop.size > 0:
                    nose_filename = f"{video_folder}_{original_name}_face{face_idx}_nose.jpg"
                    cv2.imwrite(str(featuresOutputDirNose / nose_filename), nose_crop)
                    totalNoses += 1
    print("Features Crop Complete - Extracted " + str(totalEyes) + " eyes and " + str(totalNoses) + " noses from " + str(len(imgFiles)))

origVidPath = Path("Archive/FFData/original_sequences/youtube/c23/videos")
origImgPath = Path("Frames/original")
frameExtraction(origVidPath, origImgPath)
manipVidPath = Path("Archive/FFData/manipulated_sequences/Deepfakes/c23/videos")
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

origFramePath = Path("CroppedFaces/original")
origFrameOut = Path("Features/original")
featuresCrop(origFramePath, origFrameOut)
manipFramePath = Path("CroppedFaces/manipulated")
manipFrameOut = Path("Features/manipulated")
featuresCrop(manipFramePath, manipFrameOut)
# 1520 eyes, 620 deleted, 900 eyes remain
# 157 noses, 25 deleted, 132 noses remain
