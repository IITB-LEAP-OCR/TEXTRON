<div align="center">

<h3 align="center">TEXTRON: Weakly Supervised Multilingual Text Detection through Data Programming</h3>
</div>

<p align="center">
  
  [![arXiv](https://img.shields.io/badge/arXiv-2402.09811-b31b1b.svg)](https://arxiv.org/abs/2402.09811) &nbsp;
   [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Data-yellow)](https://huggingface.co/datasets/badrivishalk/TEXTRON_INDIC_DATASETS)
</p>

 



Data Programming for Text Detection in Documents using [CAGE](https://arxiv.org/abs/1911.09860). The work includes usage of CAGE from [SPEAR](https://github.com/decile-team/spear) to detect text within documents accurately, which can be used in creation of large benchmark datasets for Text detection task for any down stream tasks.

## ABSTRACT

Several recent deep learning (DL) based techniques perform considerably well on image-based multilingual text detection. However, their performance relies heavily on the availability and quality of training data. There are numerous types of page-level document images consisting of information in several modalities, languages, fonts, and layouts. This makes text detection a challenging problem in the field of computer vision (CV), especially for low-resource or handwritten languages. Furthermore, there is a scarcity of word-level labeled data for text detection, especially for multilingual settings and Indian scripts that incorporate both printed and handwritten text. Conventionally, Indian script text detection requires training a DL model on plenty of labeled data, but to the best of our knowledge, no relevant datasets are available. Manual annotation of such data requires a lot of time, effort, and expertise. In order to solve this problem, we propose \Textron, a {\em Data Programming-based approach}, where users can plug various text detection methods into a weak supervision-based learning framework. One can view this approach to multilingual text detection as an ensemble of different CV-based techniques and DL approaches. **TEXTRON** can leverage the predictions of DL models pre-trained on a significant amount of language data in conjunction with CV-based methods to improve text detection in other languages. We demonstrate that **TEXTRON** can improve the detection performance for documents written in Indian languages, despite the absence of corresponding labeled data. Further, through extensive experimentation, we show improvement brought about by our approach over the current State-of-the-art (SOTA) models, especially for handwritten Devanagari text.


## Citation

If you use this paper or the accompanying code/data in your research, please cite it as:

```
@InProceedings{TEXTRON,
    author    = {Dhruv Kudale and Badri Vishal Kasuba and Venkatapathy Subramanian and Parag Chaudhuri and Ganesh Ramakrishnan},
    title     = {TEXTRON: Weakly Supervised Multilingual Text Detection Through Data Programming},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2871-2880},
    url       = {https://arxiv.org/abs/2402.09811}
}
```





## Getting Started

### Installation and Implementation

1. Run ```pip install -r requriements.txt``
2. Make the configurations as stated in **config.py**
   1. Create a directory outside the main project directory, **data**, with a sub-directory **temp**
   2. Within **temp**, create 2 sub-directories, **img** and **txt**
       - Place your input images in the _img_ sub-directory and the corresponding ground truth labels (if available) in the _txt_ sub-directory
            - Set the appropriate path for **INPUT_DATA_DIR** in _config.py_
            - In case ground truth isn't available, set **GROUND_TRUTH_AVAILABLE** within config.py as `False`
       - Choose the appropriate Labeling functions within config.py file from the **lab_funcs** list and also set the respective quality quide for CAGE
3. Finally, run the main.py code to get the predictions in the _results_ folder (outside the main project directory) defined in config.py


## Methodology

1. Passing images to the CAGE model, which has several labeling functions which generate weak labels of pixel level information of Image data describing Textual or Non-textual information of the corresponding pixel
2. Usage of effective post processing steps to generate bounding boxes for the corresponding detected word level text
3. This method can be applied to documents of various settings and provides SOTA and near to SOTA results
4. Labeling functions could be used as a plug and play model to analyze results of different configurations


## Labeling Functions

- Pretrained Models based Labelling Functions
- 1. DocTR 
- Image Processing based Labelling Functions
- 1. Convex hull Labeling Function
- 2. Edges based Labeling Function
- 3. Contour based Labeling Function
- 4. Segmentation based Labeling Function
- 5. Mask Region based Labeling Function
- 6. Tesseract Model for Text Detection

## Datasets


The Datasets could be found at this [link](https://iitbacin-my.sharepoint.com/:f:/g/personal/22m2119_iitb_ac_in/EghqK7T05VdEhQhxAFz9wDAB51FTKm8VDJStPL3ZxoXpQw?e=lVQXeu)

Alternatively, Data has been made available at Huggingface [link](https://huggingface.co/datasets/badrivishalk/TEXTRON_INDIC_DATASETS)


## Results

|**Class** | **Coverage%** | **DBNet Model** |**Textron3LF** |**Textron4LF** |
| :---:   | :---: | :---: | :---:   | :---: |
Date       | 00.02\%    | 33.34     | **100.00** | **66.67**|
Author     | 00.08\%    | 76.40     | 75.79      | **77.78**|
Title      | 00.13\%    | **77.94** | 26.29      | 58.54|
Section    | 00.79\%    | 57.08     | **61.30**  | **66.13**|
List       | 00.86\%    | 52.37     | **66.95**  | **62.02**|
Abstract   | 01.34\%    | 51.54     | **76.87**  | **69.52**|
Footer     | 01.57\%    | 54.67     | **72.10**  | **67.64**|
Caption    | 02.38\%    | 42.25     | **67.65**  | **57.34**|
Table      | 04.83\%    | 22.99     | **28.48**  | 21.03|
Equation   | 07.59\%    | 2.86      | **18.31**  | **11.71**|
Reference  | 10.09\%    | 48.10     | **68.45**  | **65.27**|
Paragraph  | 70.31\%    | 49.98     | **68.36**  | **63.68**|
Overall    | 100.00\%   | 46.24   | **63.38** | **58.91**|

Textron results on classwise data of Docbank for 100 test images

| Threshold | P | R | F1        | P              | R              | F1             | P              | R              | F1             |
| :---:   | :---: | :---: | :---:   | :---: | :---: | :---:   | :---: | :---: | :---: |
| 0.5 | 40.49 | 74.03  | 52.35  | **90.49** | 80.00          | 84.92          | 87.23          | **84.46** | **85.82** |
| 0.6 | 29.63 | 54.16  | 38.30  | 76.45          | 67.59          | 71.75          | **79.97** | **77.43** | **78.68** |
| 0.7 | 13.21 | 24.15  | 17.08  | 43.56          | 49.27          | 46.24          | **64.42** | **62.38** | **63.38** |
| 0.8 | 04.62 | 08.44  | 05.97  | 18.82          | 16.64          | 17.67          | **33.63** | **32.56** | **33.09** |
| 0.9 | 00.36 | 00.65  | 00.46  | 03.11          | 02.75          | 02.92          | **33.63** | **32.56** | **33.09** |

TEXTRON yields a better overall performance and also shows significant improvement in detecting classes like equations and footers as compared to DBNet

## References

1. [SPEAR](https://github.com/decile-team/spear)
2. [Scikit-Image](https://scikit-image.org/)
3. [Docbank dataset](https://github.com/doc-analysis/DocBank)
4. [DocTR](https://github.com/mindee/doctr)


## License

The work has been licensed by GNU license

## Acknowledgements

1. We wish to Acknowledge IITB annotators for annotating the Text Detection dataset to perform our experiments.
2. We acknowledge the support of a grant from IRCC, IIT
Bombay, and MEITY, Government of India, through the
National Language Translation Mission-Bhashini project.

## Authors Contact Information

1. Badri Vishal Kasuba
2. Dhruv Kudale

## Questions or Issues

we conclude with opening doors to more innovative contributions bringing about seamless multilingual text detection. Thank you for your interest in our research paper!










