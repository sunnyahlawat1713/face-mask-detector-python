# face-mask-detector-python
In this project, I have built a program in python to detect people's faces in a static image/video stream and predict whether the person is wearing a mask or not.

# Face detection:
For face detection, I have used OpenCV's readNet(cv2.dnn.readNet()) model which takes in two arguments, a binary file containing trained weights, and a text file containing network configuration. Both of these files can be found in the face_detector folder of this repository. 

# Mask detection: 
For mask prediction(on detected faces), I have used the MobileNetV2 model trained on ImageNet weights with the top of the model excluded and the weights of the remaining layers have been retained. The top(head) of the model has been re-defined to suit the specific classification problem at hand(mask-detection).
