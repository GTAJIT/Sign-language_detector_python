import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = input("Enter the class name: ")
dataset_size = int(input(f"How many images do you want to collect for {classes} class? (default: 100): "))
if dataset_size == '':
    dataset_size = 100

cap = cv2.VideoCapture(0)

# Wait until user presses 's' to start
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    mirrored_frame = cv2.flip(frame, 1)
    cv2.putText(mirrored_frame, f'Class {classes} - Press "s" to start saving', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Mirrored Webcam', mirrored_frame)
    if cv2.waitKey(25) & 0xFF == ord('s'):
        break

class_dir = os.path.join(DATA_DIR, str(classes))
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

print(f'Collecting images for class {classes}')
counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    if not ret:
        continue
    mirrored_frame = cv2.flip(frame, 1)
    cv2.putText(mirrored_frame, f'{counter}', (600,450), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imshow('Mirrored Webcam', mirrored_frame)
    cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), mirrored_frame)
    counter += 1

    # Delay to avoid saving too quickly (optional)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
print(f'done for class {classes}')
    

print('Images collected successfully!')
cap.release()
cv2.destroyAllWindows()