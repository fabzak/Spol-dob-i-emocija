# Face_detection
# Spol-dob-i-emocija
Napravljena su tri koda: 
-DF.py je kod koji koristi DeepFace model za prepoznavanje spola, dobi i emocija te se koristi OpenCV za detekciju lica, može se obrađivati 'live feed' ili slike iz mape test (slike       formata .png i .jpg)
-DF_RGB.py je kod koji je isti kao i DF.py uz dodatak promjene boja u RGB spektru na 'live feedu'
-ONNX.py je kod koji koristi mediapipe za detekciju slike i ONNX modele za prepoznavanje spola i dobi, nisu točni uopče, ali 'live feed' je u znatno večim fps-ima nego DeepFace kod, također ima RGB mod

-kod prvog pokretanja koda s DeepFace će se vjerojatno morati instalirati DF modeli pa to može potrajati, također postoji mogućnost da neće raditi od prve, to jest morati će se pokrenut dva tri puta kod kako bi proradilo (ne pitajte kako sam to doznao :)

# Kreiranje virtualnog okruženja
python3 -m venv env
source env/bin/activate  # Na Windowsu: env\Scripts\activate

# PRVO UVIJEK AŽURIRATI PIP!!!
pip install --upgrade pip

# Potrebni paketi
pip install opencv-python opencv-python-headless deepface mediapipe onnxruntime

# Ako se ne instalira s deepface-om treba i tensorflow
pip install tensorflow tf-keras

# Ako nemate NVIDIA GPU za optimizirano pokretanje instalirati i tensorflow za CPU
pip install tensorflow-cpu
