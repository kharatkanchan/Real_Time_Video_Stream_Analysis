package com.example;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class VideoStreamUI extends JFrame {
    private JLabel imageView;
    private CascadeClassifier faceDetector;
    private Net ageNet;
    private Net genderNet;
    private String[] ageLabels = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
    private String[] genderLabels = {"Male", "Female"};
    private boolean isCameraRunning = false;
    private boolean isRecognitionEnabled = true;
    private VideoCapture capture;

    public VideoStreamUI() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load models
        ageNet = Dnn.readNetFromCaffe("src/main/resources/deploy_age.prototxt", "src/main/resources/age_net.caffemodel");
        genderNet = Dnn.readNetFromCaffe("src/main/resources/deploy_gender.prototxt", "src/main/resources/gender_net.caffemodel");

        faceDetector = new CascadeClassifier("src/main/resources/haarcascade_frontalface_alt.xml");

        imageView = new JLabel();
        add(imageView, BorderLayout.CENTER);
        setTitle("Real-Time Video Stream Analysis");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel controlPanel = new JPanel();
        JButton startStopCameraButton = new JButton("Start Camera");
        JButton toggleRecognitionButton = new JButton("Toggle Recognition");

        startStopCameraButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (isCameraRunning) {
                    stopCamera();
                    startStopCameraButton.setText("Start Camera");
                } else {
                    startCamera();
                    startStopCameraButton.setText("Stop Camera");
                }
            }
        });

        toggleRecognitionButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                isRecognitionEnabled = !isRecognitionEnabled;
                toggleRecognitionButton.setText(isRecognitionEnabled ? "Disable Recognition" : "Enable Recognition");
            }
        });

        controlPanel.add(startStopCameraButton);
        controlPanel.add(toggleRecognitionButton);
        add(controlPanel, BorderLayout.SOUTH);

        setVisible(true);
    }

    private void startCamera() {
        isCameraRunning = true;
        capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println("Error: Camera not detected");
            return;
        }

        Thread cameraThread = new Thread(new Runnable() {
            @Override
            public void run() {
                Mat frame = new Mat();
                while (isCameraRunning && capture.read(frame)) {
                    if (!frame.empty()) {
                        if (isRecognitionEnabled) {
                            processFrame(frame);
                        }

                        // Resize frame to fit the JLabel size
                        Imgproc.resize(frame, frame, new Size(imageView.getWidth(), imageView.getHeight()));

                        // Convert and display the frame
                        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB); // Convert the frame to RGB
                        ImageIcon image = new ImageIcon(matToBufferedImage(frame));
                        imageView.setIcon(image);
                        imageView.repaint();
                    } else {
                        System.out.println("No captured frame -- Break!");
                        break;
                    }
                }
            }
        });
        cameraThread.start();
    }

    private void stopCamera() {
        isCameraRunning = false;
        if (capture != null) {
            capture.release();
        }
    }

    private void processFrame(Mat frame) {
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(frame, faces);

        for (Rect rect : faces.toArray()) {
            // Draw rectangles around detected faces
            Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0), 2);

            // Predict age and gender
            Mat faceROI = new Mat(frame, rect);
            String age = predictAge(faceROI);
            String gender = predictGender(faceROI);

            // Display age and gender next to the rectangle
            String label = gender + ", " + age;
            int baseLine[] = new int[1];
            Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, 1, baseLine);

            // Draw the label background rectangle
            Imgproc.rectangle(frame, new Point(rect.x, rect.y - labelSize.height - baseLine[0]),
                    new Point(rect.x + labelSize.width, rect.y), new Scalar(255, 255, 255), Imgproc.FILLED);
            // Draw the label text
            Imgproc.putText(frame, label, new Point(rect.x, rect.y - baseLine[0]),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 0), 2);
        }
    }

    private String predictAge(Mat face) {
        Mat blob = Dnn.blobFromImage(face, 1.0, new Size(227, 227),
                new Scalar(78.4263377603, 87.7689143744, 114.895847746), false);
        ageNet.setInput(blob);
        Mat agePreds = ageNet.forward();
        int classId = getMaxClassId(agePreds);
        return ageLabels[classId];
    }

    private String predictGender(Mat face) {
        Mat blob = Dnn.blobFromImage(face, 1.0, new Size(227, 227),
                new Scalar(78.4263377603, 87.7689143744, 114.895847746), false);
        genderNet.setInput(blob);
        Mat genderPreds = genderNet.forward();
        int classId = getMaxClassId(genderPreds);
        return genderLabels[classId];
    }

    private int getMaxClassId(Mat preds) {
        Core.MinMaxLocResult result = Core.minMaxLoc(preds);
        return (int) result.maxLoc.x;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer); // Get all the pixels
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }

    public static void main(String[] args) {
        new VideoStreamUI();
    }
}
