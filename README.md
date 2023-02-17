# TEXTRON : Improving Multi-Lingual Text Detection through Data Programming

Data Programming for Text Detection in Documents using [CAGE](https://arxiv.org/abs/1911.09860)

# IDEA

The work includes usage of CAGE from [SPEAR](https://github.com/decile-team/spear) to detect text within documents accurately, which can be used in creation of large benchmark datasets for Text detection task for any down stream tasks.


# Methodology

1. Passing images to the CAGE model, which has several labeling functions which generate weak labels of pixel level information of Image data describing Textual or Non-textual information of the corresponding pixel
2. Usage of effective post processing steps to generate bounding boxes for the corresponding detected word level text
3. This method can be applied to documents of various settings and provides SOTA and near to SOTA results
4. Labeling functions could be used as a plug and play model to analyze results of different configurations


# Labeling Functions

- Pretrained Models based Labelling Functions
- 1. DocTR 
- Image Processing based Labelling Functions
- 1. Convex hull Labeling Function
- 2. Edges based Labeling Function
- 3. Contour based Labeling Function
- 4. Segmentation based Labeling Function
- 5. Mask Region based Labeling Function
- 6. Tesseract Model for Text Detection

# Results


| |Tesseract | DBNet | TEXTRON|
| :---:   | :---: | :---: | :---:   | :---: | :---: | :---:   | :---: | :---: | :---: |
| Threshold | P | R | F1        | P              | R              | F1             | P              | R              | F1             |
| :---:   | :---: | :---: | :---:   | :---: | :---: | :---:   | :---: | :---: | :---: |
| 0.5 | 40.49 | 74.03  | 52.35  | **90.49** | 80.00          | 84.92          | 87.23          | **84.46** | **85.82** |
| 0.6 | 29.63 | 54.16  | 38.30  | 76.45          | 67.59          | 71.75          | **79.97** | **77.43** | **78.68** |
| 0.7 | 13.21 | 24.15  | 17.08  | 43.56          | 49.27          | 46.24          | **64.42** | **62.38** | **63.38** |
| 0.8 | 04.62 | 08.44  | 05.97  | 18.82          | 16.64          | 17.67          | **33.63** | **32.56** | **33.09** |
| 0.9 | 00.36 | 00.65  | 00.46  | 03.11          | 02.75          | 02.92          | **33.63** | **32.56** | **33.09** |

TEXTRON yields a better overall performance and also shows significant improvement in detecting classes like equations and footers as compared to DBNet

# References

1. [SPEAR]((https://github.com/decile-team/spear))
2. [Scikit-Image](https://scikit-image.org/)
3. [Docbank dataset](https://github.com/doc-analysis/DocBank)
4. [DocTR](https://github.com/mindee/doctr)


# Installation and Implementation

1. Run pip install -r requriements.txt
2. Make the configurations as stated in config.py
3. Run the main.py code to get the predictions in the results folder defined in config.py