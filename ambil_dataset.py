import cv2
import os

# nama orang
nama = "gilang"

# folder dataset
path = "dataset/" + nama

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Ambil Dataset", frame)

    # tekan spasi untuk ambil foto
    key = cv2.waitKey(1)

    if key == 32:  # tombol spasi
        count += 1
        filename = path + "/" + str(count) + ".jpg"
        cv2.imwrite(filename, gray)
        print("Foto disimpan:", filename)

    if key == 27:  # ESC keluar
        break

cap.release()
cv2.destroyAllWindows()