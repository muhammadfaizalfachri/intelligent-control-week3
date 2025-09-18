import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model yang telah dilatih
model = load_model('cnn_model.h5')  # Pastikan file cnn_model.h5 adalah hasil training dari langkah 3

# Load label kelas
# Ganti dengan daftar label kelas sesuai model Anda, misal:
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Ganti dengan label kelas hasil training Anda

# Cek jumlah output model dan jumlah label
num_classes = model.output_shape[-1]
if len(class_labels) != num_classes:
    raise ValueError(f"Jumlah label ({len(class_labels)}) tidak sama dengan output model ({num_classes}).")

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)
    
    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi kelas
    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100  # Persentase akurasi
    if idx >= len(class_labels):
        label = "Unknown"
    else:
        label = class_labels[idx]
    
    # Tampilkan hasil
    cv2.putText(frame, f'Class: {label} ({confidence:.2f}%)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()