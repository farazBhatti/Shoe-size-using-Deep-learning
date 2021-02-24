# import the necessary packages
import argparse
import cv2
import imutils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image




""" Step 1: Edge Detection """
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # get the grayscale image
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
#gray = cv2.GaussianBlur(gray, (3, 3), 0) # with a bit of blurring
#BasicImage(gray).show()

# automatic Canny edge detection thredhold computation


# zero-parameter automatic Canny edge detection (method 2)
# Vary the percentage thresholds that are determined (in practice 0.33 tends to give good approx. results)
# A lower value of sigma  indicates a tighter threshold, whereas a larger value of sigma  gives a wider threshold.
 


#BasicImage(edged).show()

# since some of the outlines are not exactly clear, we construct
# and apply a closing kernel to close the gaps b/w white pixels
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)





gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 20, 40)


#
#high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#low_thresh = high_thresh / 2.0
#
#
#print('high threshold = ',high_thresh)
#print('low threshold = ',low_thresh)
#
#
#edged = cv2.Canny(gray, low_thresh, high_thresh) # detect edges (outlines) of the objects



# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) != 0:
    # find the biggest countour (c) by the area
    c = max(cnts, key = cv2.contourArea)
    cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
    x,y,w,h = cv2.boundingRect(c)
    
# show the contour (outline) of the piece of paper
#print('Approx = ',len(approx))
print("STEP 2: Find contours of paper")
#print("screen count = ", screenCnt)
#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()