import imutils
from imutils.video import WebcamVideoStream
import tensorflow as tf
import cv2
import numpy as np


def add_sunglasses(image, right_eye_corner, left_eye_corner):
    # read the image including the alpha-channel (done by adding the IMREAD_UNCHANGED flag)
    sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

    # set the width to be 50% more than from eye to eye and resize the image
    width = int(abs(right_eye_corner[0] - left_eye_corner[0]) * 1.5)
    sunglasses = imutils.resize(sunglasses, width=width)
    rows, cols, _ = sunglasses.shape

    # the starting point to insert the image is the on the right eye's outer corner.
    # since it is mirrored, left is actually right, so we move a little bit up and left from the coordinates
    start_x = int(right_eye_corner[0] - width/6)
    start_y = int(right_eye_corner[1] - rows / 2.5)

    # the image is the first three channels
    overlay_image = sunglasses[..., :3]
    # the alpha channel is the fourth
    mask = sunglasses[..., 3:] / 255.0

    result = image.copy()
    # using the mask to include the pixels from the webcam and remove the 'transparent' pixels from the sunglasses
    # see also https://stackoverflow.com/a/54058766
    result[start_y:start_y + rows, start_x:start_x + cols] = ((1.0 - mask) * result[start_y:start_y + rows,
                                                              start_x:start_x + cols] + mask * overlay_image)

    return result


def run_detection(verbose=False):
    # load model architecture and weights
    with open("facial-keypoints-detection.json", "r") as file:
        model = tf.keras.models.model_from_json(file.read())
    model.load_weights('facial-keypoints-detection.h5')

    webcam = WebcamVideoStream(0).start()

    # load OpenCV face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # read frame from webcam
        frame = webcam.read()
        image = imutils.resize(frame, height=480)

        # the model was trained on grayscale, so convert
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        faces = None

        for (x, y, w, h) in detected_faces:
            if verbose: cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cut out the face and save in a stack of faces for prediction
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (96, 96))

            face = face / 255

            face = face.reshape(1, 96, 96, 1)
            faces = face if faces is None else np.vstack([faces, face])

        if len(detected_faces) > 0:
            # predict all of the faces
            predictions = model.predict(faces)

            for (f_x, f_y, f_w, f_h), points in zip(detected_faces, predictions):
                # find the corners (convert to image coordinates from detected coordinates)
                right_eye_corner = points[10] * f_w + f_x, points[11] * f_h + f_y
                left_eye_corner = points[6] * f_w + f_x, points[7] * f_h + f_y

                # add sunglasses
                image = add_sunglasses(image, right_eye_corner, left_eye_corner)

                if verbose:
                    for x, y in zip(points[0::2], points[1::2]):
                        x = int(x*f_w) + f_x
                        y = int(y*f_h) + f_y

                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # show the image
        cv2.imshow("Webcam", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    run_detection()