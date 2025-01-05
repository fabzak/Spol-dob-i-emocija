# Face_detection

# Kreiranje virtualnog okruženja
python3 -m venv env
source env/bin/activate  # Na Windowsu: env\Scripts\activate

# PRVO UVIJEK AŽURIRATI PIP!!!
pip install --upgrade pip

# Potrebni paketi
pip install opencv-python opencv-python-headless deepface

# Ako se ne instalira s deepface-om treba i tensorflow
pip install tensorflow tf-keras

# Ako nemate NVIDIA GPU za optimizirano pokretanje instalirati i tensorflow za CPU
pip install tensorflow-cpu