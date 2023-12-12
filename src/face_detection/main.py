import cv2

def detect_faces_in_image(image_path, cascade_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image!")
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to black and white, is less color to simplefy the process

    # Load the Haar cascade for face detection
    haar_cascade = cv2.CascadeClassifier(cascade_path) # cascadeClassifier
    if haar_cascade.empty():
        print("Failed to load the Haar cascade.")
        return

    # Detect faces in the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces_rect) == 0:
        print("No faces detected!")
        return

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define paths to your image and Haar cascade XML file
image_path = ''
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # output form a machine learning
# algorithm/ what im doing here is using the machine learinig to recognize a face

# Call the function
detect_faces_in_image(image_path, cascade_path)

