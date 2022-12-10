import cv2
import numpy as np


def get_contour_labels(imgfile):
  image = cv2.imread(imgfile)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Convert the grayscale image to binary
  ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

  # To detect object contours, we want a black background and a white 
  # foreground, so we invert the image (i.e. 255 - pixel value)
  inverted_binary = ~binary

  # Find the contours on the inverted binary image, and store them in a list
  # Contours are drawn around white blobs.
  # hierarchy variable contains info on the relationship between the contours
  contours, hierarchy = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

  #This is inmtermediate contour image having red contours plotted along the letters
  with_contours_int = cv2.drawContours(image, contours, -1,(0,0,255),2)
 
  #We again perform binarization of above image inorder to find contours again 
  gray_contour = cv2.cvtColor(with_contours_int, cv2.COLOR_BGR2GRAY)

  ret, binary_contour = cv2.threshold(gray_contour, 100, 255, 
    cv2.THRESH_OTSU)
  inverted_contour = ~binary_contour

  # We find contours again of this inverted binary map so that word boundaries are detected
  contours, hierarchy = cv2.findContours(inverted_contour,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

  #New contours are blue in colour to identify word bounds
  with_contours = cv2.drawContours(with_contours_int, contours, -1,(255,0,0),1)

  origimage = cv2.imread(imgfile)
  bboxes = []
  # Draw a bounding box around all contours
  for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      # Make sure contour area is large enough
      if (cv2.contourArea(c)) > 20:
          #cv2.rectangle(origimage,(x,y), (x+w,y+h), (0,0,0), cv2.FILLED)
          bboxes.append([x, y, w, h])
          
  final_img = np.zeros((1024, 1024, 3), dtype = np.uint8)
  for b in bboxes:
      x = b[0]
      y = b[1]
      w = b[2]
      h = b[3]
      cv2.rectangle(final_img,(x,y), (x+w,y+h), (255, 255, 255), cv2.FILLED)

  return final_img

imgfile = 'cropped.jpg'
final_img = get_contour_labels(imgfile)
final_img = ~final_img
cv2.imwrite('final.jpg', final_img)