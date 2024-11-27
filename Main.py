import time
import tkinter
from tkinter import *
from tkinter import  ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

main = tkinter.Tk()
main.title("Density Based Smart Traffic Control System")
main.geometry("1270x650")

background_image = Image.open("bgg.jpg")  # Replace with your image path
background_image = background_image.resize((1270,550), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)

# Create a canvas
canvas = Canvas(main, width=1270, height=550)
canvas.pack(fill="both", expand=True)

# Set the background image
image_x = 0
image_y = 70
canvas.create_image(image_x, image_y, image=background_photo, anchor="nw")

global filename
global refrence_pixels
global sample_pixels
global ans

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
def uploadTrafficImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    pathlabel.config(text=filename)

def yol():
    global ans
    ans = 0
    try:
        image = cv2.imread(filename)
        (H, W) = image.shape[:2]
        print("[INFO]: Image loaded successfully")
    except Exception as e:
        messagebox.showinfo('[INFO]: Got Error while loading image:', str(e))
        raise SystemExit()
    # Load YOLO model
    net = cv2.dnn.readNet("yolo custom/yolov3_custom_last.weights","yolo custom/yolov3_custom.cfg")

    # Load class labels
    with open("yolo custom\classes.names", 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]

    # Generate random colors for each label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Get output layer names
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO]: YOLO took {:.6f} seconds".format(end - start))

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.4:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check for a specific class ID
            if classIDs[i] == 0:  # Adjust this ID based on your specific need
                ans = 1
                print(ans)

    # Save and display the output image
    if ans==1:
        cv2.imwrite("output/output.jpg", image)
        cv2.imshow("YOLO", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        confidence = 0.5
        threshold = 0.3
        try:
            image= cv2.imread(filename)
            (H, W) = image.shape[:2]
            print("[INFO]: Image loaded successfully")
        except Exception as e:
            messagebox.showinfo('[INFO]: Got Error while loading image:', str(e))
            raise SystemExit()
    # Load the COCO class labels our YOLO model was trained on
        labelsPath = 'yolo-coco/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")

    # Initialize a list of colors to represent each possible class label
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Paths to the YOLO weights and model configuration
        weightsPath = 'yolo-coco/yolov3.weights'
        configPath = 'yolo-coco/yolov3.cfg'

    # Load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Determine only the *output* layer names that we need from YOLO
        layer_names = net.getLayerNames()
        try:
            unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
            ln = [layer_names[i - 1] for i in unconnected_out_layers]
        except AttributeError:
            ln = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

    # Initialize our lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []

    # Loop over each of the layer outputs
        for output in layer_outputs:
        # Loop over each of the detections
            for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence_value = scores[classID]

            # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
                if confidence_value > confidence:
                # Scale the bounding box coordinates back relative to the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence_value))
                    classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    # Ensure at least one detection exists
        if len(idxs) > 0:
        # Loop over the indexes we are keeping
            for i in idxs.flatten():
            # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output image
        cv2.imshow("YOLO", image)
        cv2.waitKey(0)
def visualize(imgs, format=None,gray=False):
    j = 0
    plt.figure(figsize=(10, 20))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        if j == 0:
            plt.title('Sample Image')
            plt.imshow(img, format)
            j = j + 1
        elif j > 0:
            plt.title('Reference Image')
            plt.imshow(img, format)

    plt.show()
from CannyEdgeDetector import CannyEdgeDetector

def applyCanny():
    #messagebox.showinfo("IMage is")
    imgs = []
    img = mpimg.imread(filename)
    img = rgb2gray(img)
    imgs.append(img)
    edge = CannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    imgs = edge.detect()
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
    cv2.imwrite("gray/test.png",img)
    temp = []
    img1 = mpimg.imread('gray/test.png')
    img2 = mpimg.imread('gray/refrence1.png')
    temp.append(img1)
    temp.append(img2)
    visualize(temp)

def pixelcount():
    global refrence_pixels
    global sample_pixels
    img = cv2.imread('gray/test.png', cv2.IMREAD_GRAYSCALE)
    sample_pixels = np.sum(img == 255)
    print(sample_pixels)
    img = cv2.imread('gray/refrence1.png', cv2.IMREAD_GRAYSCALE)
    refrence_pixels = np.sum(img == 255)
    print(refrence_pixels)
    messagebox.showinfo("Pixel Counts", "Total sample White Pixels Count : "+str(sample_pixels)+"\nTotal Reference White Pixels Count : "+str(refrence_pixels))


def timeAllocation():
    global ans
    if ans == 1:
        messagebox.showinfo("Green Signal Allocation Time","Traffic found ambulance green signal will be in time : 10 secs")
    elif ans == 0:
        avg = (sample_pixels-refrence_pixels)/100
        if avg >= 90:
            messagebox.showinfo("Green Signal Allocation Time","Traffic is very high allocation green signal time : 60 secs")
        if avg > 85 and avg < 90:
            messagebox.showinfo("Green Signal Allocation Time","Traffic is high allocation green signal time : 50 secs")
        if avg > 65 and avg <= 85:
            messagebox.showinfo("Green Signal Allocation Time","Traffic is moderate green signal time : 40 secs")
        if avg > 50 and avg <= 65:
            messagebox.showinfo("Green Signal Allocation Time","Traffic is low allocation green signal time : 30 secs")
        if avg <= 50:
            messagebox.showinfo("Green Signal Allocation Time","Traffic is very low allocation green signal time : 20 secs")


def exit():
    main.destroy()



font = ('times', 16, 'bold')
title = Label(main, text='      Density Based Smart Traffic Control System Using Canny Edge Detection Algorithm for Congregating Traffic Information',anchor=W, justify=CENTER)
title.config(bg='aquamarine', fg='black')
title.config(font=font)
title.config(height=3, width=110)
title.place(x=0,y=0)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Traffic Image", command=uploadTrafficImage)
upload.place(x=200,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=200,y=150)

process1 = Button(main, text="Detect Image", command=yol)
process1.place(x=200,y=200)
process1.config(font=font1)


process = Button(main, text="Image Preprocessing Using Canny Edge Detection", command=applyCanny)
process.place(x=200,y=250)
process.config(font=font1)

count = Button(main, text="White Pixel Count", command=pixelcount)
count.place(x=200,y=300)
count.config(font=font1)

count = Button(main, text="Calculate Green Signal Time Allocation", command=timeAllocation)
count.place(x=200,y=350)
count.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=200,y=400)
exitButton.config(font=font1)


main.config(bg='lavender')
main.mainloop()
