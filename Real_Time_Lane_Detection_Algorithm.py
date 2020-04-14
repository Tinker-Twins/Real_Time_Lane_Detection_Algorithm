# Lane Detection Algorithm

# Import Libraries

import cv2 as cv
import numpy as np


# Optimization

def make_coordinates(image, line):
    slope, intercept = line
    y1 = int(image.shape[0]) # Initial Y-Coordinate to Draw Lines (Bottom of Image)
    y2 = int(y1*3/5)         # Final Y-Coordinate to Draw Lines (Slightly Lower than Middle)
    x1 = int((y1-intercept)/slope) # y=mx+c ==> x=(y-c)/m
    x2 = int((y2-intercept)/slope) # y=mx+c ==> x=(y-c)/m
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = [] # This List will Contain Coordinates of Left Lane Line
    right_fit = [] # This List will Contain Coordinates of Right Lane Line
    if lines is None:
        return None
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1) # Fits a Polynomial of Degree 1 (Last Argument) to Given Pair of Points (x1,x2),(y1,y2)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:                           # Deciding Lines Based on Slope (-ve-->Left & +ve-->Right)
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average  = np.average(left_fit, axis=0) # Average Left Line Array Vertically (Axis=0) to get Average Slopes and Intercepts
        right_fit_average = np.average(right_fit, axis=0) # Average Right Line Array Vertically (Axis=0) to get Average Slopes and Intercepts
        left_line  = make_coordinates(image, left_fit_average) # Draw Left Line Based on Coordinates, Slope and Intercept
        right_line = make_coordinates(image, right_fit_average) # Draw Right Line Based on Coordinates, Slope and Intercept
        averaged_lines = [left_line, right_line]
        return averaged_lines


# Image Processing

def gradient(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY) # Convert to Grayscale
    smooth_image = cv.GaussianBlur(gray_image,(5,5),0) # Gaussian Blur (Smoothening)
    gradient_image = cv.Canny(gray_image,50,150)       # Canny Edge Detection (Gradients)
    return gradient_image


# Lane Detection from Gradient Image (Line Detection using Hough Transform)

def display_detected_lines(image,lines):
    line_image = np.zeros_like(image) # Black Image
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(0,255,0),10) # Arguments: Image, First Point Coordinate to Draw Line, Second Point Coordinate to Draw Line, Color of the Line [BGR], Line Thickness
    return line_image


# Region of Interest Selection

def region_of_interest(gradient_image):
    height = gradient_image.shape[0]
    width = gradient_image.shape[1]
    ROI = np.array([[(200, height),(550, 250),(1100, height),]]) # Defining Triangular ROI
    mask = np.zeros_like(gradient_image) # Create Mask (Black Image) of Same Dimension as Image
    cv.fillPoly(mask,ROI,255) # Fill Mask with White ROI
    masked_image = cv.bitwise_and(mask,gradient_image) # Bitwise AND Source Image and Mask to Obtain Masked Image
    return masked_image

# Import Video Stream

video = cv.VideoCapture("Input Video Stream.mp4")

# Settings to Save the Processed Video

outputVideo = cv.VideoWriter("Real Time Lane Detection.avi", cv.VideoWriter_fourcc('M', 'P', 'E', 'G'), 50, (1280, 720))

# Process Video Stream

while(video.isOpened()):
    ret,frame = video.read() # "Ret" will Return Boolean Value Regarding Getting the Frame (TRUE or FALSE) | "Frame" will Get Next Frame
    if ret:
        gradient_image = gradient(frame)
        ROI = region_of_interest(gradient_image)
        lines = cv.HoughLinesP(ROI,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=100) # Arguments:
                                                                                                 # Image for Line Detection = Region of Interest Image
                                                                                                 # Rho Accuracy = 2 Pixel
                                                                                                 # Theta Accuracy = pi/180 = (1 Degree)
                                                                                                 # Threshold (Minimum Number of Votes to Consider a Line) = 100
                                                                                                 # Placeholder Argument = Empty Array
                                                                                                 # Threshold Length of Detected Line = 40 Pixels
                                                                                                 # Maximum Distance between 2 Lines to Consider them as a Single Line = 100 Pixels
                                                                                                 # Returns: Lines as 2D Arrays of Points
        averaged_lines = average_slope_intercept(frame,lines)
        line_image = display_detected_lines(frame,averaged_lines)
        lanes_detected_image = cv.addWeighted(frame,0.8,line_image,1,0) # Arguments: Image Array 1, Weight 1, Image Array 2 (Same Size as Image Array 1), Weight 2, Gamma = Scalar Added to Each Sum
        outputVideo.write(lanes_detected_image) # Save the Processed Video
        cv.imshow("Real-Time Lane Detection", lanes_detected_image) # Display the Processed Video
        cv.waitKey(1) # Wait For 'ESC' Key to be Pressed
        if cv.waitKey(1) & 0xFF == ord(' '):
            break
    else:
        print('\nEND OF INPUT VIDEO STREAM.')
        print('\nSAVING PROCESSED VIDEO STREAM...')
        print('\nDONE!')

        break
video.release()
cv.destroyAllWindows()
