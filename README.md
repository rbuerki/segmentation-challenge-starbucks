#  Customer Segmentation - Starbucks Challenge

Create distinct customer segments based on purchasing behavior using unsupervised learning. Data provided by Starbucks and Udacity.

### Introduction to project and results

The data contains 3 sets of 'events', demographic customer data and offer data. It is simulated and mimics the behavior for 17,000 customers on the Starbucks rewards mobile app over the course of 29 days. Once every few days, Starbucks sends out an offer to users of the mobile app. The goal is to find specific customer segments that react differently to different types of offer.

What made this task extremely hard, was the structure of the data (demanding complex cleaning) and the lack of any experimental setup, so that there was barely any variable control. With help of PCA and k-means clustering, 12 distinct segments could be identified based on their purchasing behavior. They were each exposed to certain offer types and reacted to them in different ways. 

The number of 12 segments was chosen based on silhouette score, but I also checked the segments usefullness based on the distinction of the original features for the different segments. There was a good amount of experimentation and tweaking before the results were satisfying.

A final attempt to translate these behaviour-based segments into demographic customer groups showed only minor success.

An overview over the project and a more thorough discussion of the results can be found in this Medium [blogpost](https://medium.com/@raph_91654/starbucks-challenge-find-hidden-customer-segments-with-unsupervised-learning-cf81081cc324). (A PDF-copy is also included in the reports section.)

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](http://seaborn.org)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tqdm](https://pypi.org/project/tqdm/)
- [pathlib](https://docs.python.org/3/library/pathlib.html)

You will also need the custom files `cleaning_functions.py`, `EDA_functions.py`. And the software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

The main code is split-up into 4 Jupyter notebooks, numbered 1 to 5. Each of them has an introductory summary:

`1-data_prep.ipynb`: cleaning and extensive data preparation
`2-EDA.ipynb`: some basic exploratory data analysis
`3-EDA_offer_types.ipynb`: some more EDA focused on different offer types, no breaktrough (can probably be skipped)
`4-PCA_and_clustering.ipynb`: data transformation / pre-processing, dimensionality reduction and clustering
`5-Analyze_clusters.ipynb`: analysis of the different clusters based on their mean characteristics

### Data

The input data consists of three JSON files, that are described in detail in the first section of the data preparation notebook:

- `transcript.json`: 317,000 interactions / transactions on the mobile app
- `profile.json`: demographic customers data, containing 17,000 customers but 2'175 have missing demographic data
- `portfolio.json`: offer data, describing the 10 offers sent out to customers during the test period