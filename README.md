## Deep interactive segmentation of satellite imagery

This repository is an extension of the original repository that came along the article [1] *Reviving Iterative Training with Mask Guidance for Interactive Segmentation* . It allows to work with the Inria Aerial Image Labeling dataset [2].

To work with it, you need to download the dataset from the website and then prepocess it using the `scripts\preprocess_im_sat_data.py`. Otherwise, already process data can be downloaded here https://drive.google.com/drive/folders/1Htq_bLGYr4bJomN0yZ3ilIQp0r3ncgaJ?usp=sharing .

Then to train a model and run the prediction, simply run the notebook `colab_im_sat.ipynb` on google colab.

[1] Konstantin Sofiiuk, Ilia A. Petrov, and Anton Konushin. “Reviving iterative training with mask guidance for interactive segmentation.” CoRR, abs/2102.06583, 2021. 5 10

[2] Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat and Pierre Alliez. “Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark”. IEEE International Geoscience and Remote Sensing Symposium (IGARSS). 2017.


