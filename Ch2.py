import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Head function shows first 5 entries (index 0)
housing = load_housing_data()
housing.head()

#Provides a quick description of the data
housing.info()

"""
-Based on the .info() command, we see there are 20640 entries in the data set. 
-Notice that total_bedrooms has 20433 non-null entries meaning there's some missing data
-All the data types are floats/integers expect ocean_proximity. 
Since we imported from a CSV file we know this has to be a text file (not some other object)
"""

#Let's see the distribution of the ocean_proximity variable
housing["ocean_proximity"].value_counts()

#To see other fields too, use describe()
housing.describe()

#Null values are not counted in the describe() table.

#Now we can plot a histogram to see how our data is distributed
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

"""
A few things to notice from the plots. Median income doesn't seem to be in normal USD.
Look at the x-axis (it goes from 0-15). This turns out to be scaled roughly to represent
tens of thousands of dollars.
Housing median age and value are also capped. 
    If we were to resolve this we could either collect proper labels for districts where
    the labels are capped, or remove those districts from the training set.
These attributes have very different scales - will be discussed later in the scaling section.
These hisograms are tail-heavy - they extend farther to the rigth of the median than
to the left.  
"""
