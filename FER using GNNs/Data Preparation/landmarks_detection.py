import os
import cv2
import math
import numpy as np
import csv
import mediapipe as mp

train_dir = "E:\\GNN\\datasets\\train"
validation_dir = "E:\\GNN\\datasets\\validation"

dst_train_dir = "E:\\GNN\\datasets\\landmark\\train"
dst_validation_dir = "E:\\GNN\\datasets\\landmark\\validation"

dst_train_label_path = "E:\\GNN\\datasets\\label\\train"
dst_validation_label_path = "E:\\GNN\\datasets\\label\\validation"

classes = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

mp_face_mesh = mp.solutions.face_mesh


def get_landmarks(IMAGE_FILE, class_i, dst_file_path, label_path):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for idx, file in enumerate(IMAGE_FILE):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = face_landmarks.landmark[i]
                    x = pt1.x
                    y = pt1.y
                    z = pt1.z

                    data = [x, y, z]
                    with open(
                        f"{dst_file_path}\\image{idx}_{class_i}.csv",
                        "a",
                        newline="",
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

        label = [class_i]
        with open(f"{label_path}\\label_{class_i}.csv", "w") as fl:
            writer = csv.writer(fl)
            writer.writerow(label)


def landmarks(paths, dst_paths, dst_label):
    for class_i in os.listdir(paths):
        file_path = os.path.join(train_dir, class_i)
        dst_file_path = os.path.join(dst_paths, class_i)
        label_path = os.path.join(dst_label, class_i)
        file_paths = []
        for filenames in os.listdir(file_path):
            filename_path = os.path.join(file_path, filenames)
            file_paths.append(filename_path)

        get_landmarks(file_paths, classes[class_i], dst_file_path, label_path)
    print(f"\n\nFinished processing {paths}\n\n")


landmarks(train_dir, dst_train_dir, dst_train_label_path)
print(f"successfully completed train set.")
landmarks(validation_dir, dst_validation_dir, dst_validation_label_path)
print(f"successfully completed validation set.")

