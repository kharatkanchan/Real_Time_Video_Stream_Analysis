package com.example;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class VideoProcessor {
    private CascadeClassifier faceDetector;
    private VideoCapture capture;

    public VideoProcessor() {
        faceDetector = new CascadeClassifier("src/main/resources/haarcascade_frontalface_alt.xml");
        capture = new VideoCapture(0); // Use 0 for the default camera
    }

    public void processStream() {
        if (!capture.isOpened()) {
            System.out.println("Error: Camera not detected");
            return;
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0));
            }

            // Here you would send the frame to the UI for display
        }
        capture.release();
    }
}
