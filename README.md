# Spear Text Detection

Data Programming for Text Detection in Documents using Spear

# IDEA

The work includes usage of CAGE from SPEAR to detect text within documents accurately, which can be used in creation of large benchmark datasets for Text detection task for any down stream tasks




# Methodology

1. Passing images to the CAGE model, which has several labeling functions which generate weak labels of pixel level information of Image data describing Textual or Non-textual information of the corresponding pixel
2. Usage of effective post processing steps to generate bounding boxes for the corresponding detected word level text
3. This method can be applied to variou
# Labeling Functions

- Pretrained Models based Labelling Functions
- 1. DocTR 
- 2. Tesseract
- Image Processing based Labelling Functions
- 1. Convex hull Labeling Function
- 2. Edges based Labeling Function
- 3. Contour based Labeling Function
- 4. Segmentation based Labeling Function
- 5. Mask Region based Labeling Function

# Results

# References

1. SPEAR
2. Scikit-Image
3. Docbank dataset