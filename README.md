#  Customer Segmentation - Starbucks Challenge

Create distinct customer segments based on purchasing behavior using unsupervised learning. Data provided by Starbucks and Udacity.

### Introduction to project and results

The data contains 3 sets of 'events', demographic customer data and offer data. It is simulated and mimics the behavior for 17,000 customers on the Starbucks rewards mobile app over the course of 29 days. Once every few days, Starbucks sends out an offer to users of the mobile app. The goal is to find specific customer segments that react differently to different types of offer.

What made this task extremely hard, was the structure of the data (demanding complex cleaning) and the lack of any experimental setup, so that there was barely any variable control. With help of PCA and k-means clustering, 12 distinct segments could be identified based on their purchasing behavior. They were each exposed to certain offer types and reacted to them in different ways. 

The number of 12 segments was chosen based on silhouette score, but I also checked the segments usefullness based on the distinction of the original features for the different segments. There was a good amount of experimentation and tweaking before the results were satisfying.

A final attempt to translate these behaviour-based segments into demographic customer groups showed only minor success.

An overview over the project and a more thorough discussion of the results can be found in this Medium [blogpost](https://medium.com/@raph_91654/starbucks-challenge-find-hidden-customer-segments-with-unsupervised-learning-cf81081cc324). (A PDF-copy is also included in the reports section.)

The original project description can be found at the end of this README.

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


### Original project description (Udacity)

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

Example: To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.

Cleaning: This makes data cleaning especially important and tricky. You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.

Final Advice: Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (ie 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).