import cv2
import cv2 as c
import numpy as np
import tensorflow as tf
import os
from pyzbar.pyzbar import decode as dec
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def sendmailto(rec,name):
    sender_email = "23am004@kpriet.ac.in"
    receiver_email = rec
    password = "nled erec zvhy oqam"
    subject = "Defect detected"
    body = f"object {name} is found defective"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
    print("Email sent successfully!")
# 1. Barcode detection to identify the product
def bardet():
    print("inside code detection")
    cap = c.VideoCapture(1)
    a = 0
    while a != 1:
        _, img = cap.read()
        for i in dec(img):
            a = 1
            types = i.type
            data = i.data.decode()
    return data

# 2. Load old images of the product in both "working" and "off" states
def load_old_images(file,name):
    s=file+"\\"+name+"\\working"
    t=file+"\\"+name+"\\off"
    l,l1=[],[]
    try:
        for i in os.listdir(s):
            l.append(s+"\\"+i)
    except:
        pass
    try:
        for i in os.listdir(t):
            l1.append(i)
    except:
        pass
    return l,l1

# 3. CNN model for defect detection (using TensorFlow)
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output binary result: Defective or not
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def pictake():
    cap=cv2.VideoCapture(1)
    _,img=cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

# 4. Compare current image with working and off state images and predict defects
def compare_images(current_image,image_path,model):
    current_image_resized = cv2.resize(current_image, (128, 128))
    current_image_resized = np.expand_dims(current_image_resized, axis=-1)  # Add channel dimension (128, 128, 1)
    current_image_resized = np.expand_dims(current_image_resized, axis=0)   # Add batch dimension (1, 128, 128, 1)
    working_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    if working_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    working_image_resized = cv2.resize(working_image, (128, 128))
    working_image_resized = np.expand_dims(working_image_resized, axis=-1)  # Add channel dimension
    working_image_resized = np.expand_dims(working_image_resized, axis=0)  # Add batch dimension
    working_diff = np.mean(np.abs(current_image_resized - working_image_resized))
    return working_diff

# Main function
def detect_defects(cur, image_folder, model,product_name):
        working_images = load_old_images(image_folder,product_name)[0]
        ml=[]
        for i in working_images:
            val = compare_images(cur, i, model)
            ml.append(val)
        print(ml)
        p = np.array(ml)
        print("Avg diff: ",np.mean(p))
        return np.mean(p)


from time import sleep
while True:
    file='C:\\Users\\abhiv\\abhi 1\\abhi\\defect'
    name=bardet()
    print(name)
    sleep(3)
    model = create_model()
    if name:
        cur=pictake()
        thrh_val=detect_defects(cur,file,model,name)
        thrsh=120
        if thrh_val<thrsh:
            sendmailto("abhicoder39@gmail.com",name)
            print('Defect found , message sent....')
        else:
            print("No Defect....")

    sleep(3)


