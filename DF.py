import os
import cv2
from deepface import DeepFace


def find_available_cameras(max_cameras=10):
    """Pronalazi dostupne kamere."""
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras


def analyze_images_from_folder(folder_path):
    """Analiza slika u mapi."""
    if not os.path.exists(folder_path):
        print(f"Datoteka '{folder_path}' nije pronađena!")
        return

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Analiza {file_name}...")
            try:
                analysis = DeepFace.analyze(img_path=file_path, actions=['age', 'gender', 'emotion'])
                gender = analysis[0]['gender']
                age = int(analysis[0]['age'])
                emotion = analysis[0]['dominant_emotion']
                print(f"{file_name}: Gender: {gender}, Age: {age}, Emotion: {emotion}")
            except Exception as e:
                print(f"Pogreška {file_name}: {e}")
        else:
            print(f"Preskakanje datoteka koje nisu slike: {file_name}")

def analyze_live_feed(camera_index=0):
    """Analiza lica na live feedu."""
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        print(f"Error: Nemože se spojiti na kameru {camera_index}.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detekcija lica
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Analiza lica
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_img)

            try:
                analysis = DeepFace.analyze(img_path=temp_face_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                
                gender = max(analysis[0]['gender'], key=analysis[0]['gender'].get)
                age = int(analysis[0]['age'])
                emotion = analysis[0]['dominant_emotion']
                label = f"{gender}, {age}, {emotion}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
            except Exception as e:
                print(f"Error analize: {e}")


        cv2.imshow("Live Feed - Age, Gender, Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    print("Odabir:")
    print("1: Analiza slika iz mape")
    print("2: Analiza live feeda")
    choice = input("Odabir (1 or 2): ")

    if choice == "1":
        folder_path = "test"  # Promjeni naziv mape za testne slike!!!!
        analyze_images_from_folder(folder_path)
    elif choice == "2":
        print("Traženje dostupnih kamera")
        available_cameras = find_available_cameras()
        if not available_cameras:
            print("Nema dostupnih kamera. Izlaz.")
            return

        print("Dostupne kamere:")
        for cam_index in available_cameras:
            print(f"Kamera {cam_index}")

        try:
            camera_index = int(input("Unesite indeks kamere: "))
            if camera_index not in available_cameras:
                print("Pogrešan indeks kamere. Izlaz.")
                return
        except ValueError:
            print("Pogrešan ulaz. Izlaz.")
            return

        analyze_live_feed(camera_index=camera_index)
    else:
        print("Pogrešan odabir! Izlaz.")

if __name__ == "__main__":
    main()
