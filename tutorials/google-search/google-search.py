#!/usr/bin/env python3

# From: https://linuxhint.com/google_search_api_python/

# How to search google with python.
# Limited to 100 searches per day for free.

# Create a custom search engine here: 
# https://cse.google.com/cse/all
# Private CSE ID: #############################

# Create an api key here:
# https://developers.google.com/custom-search/v1/overview?csw=1
# Private API Key: ##################################

# Install Google api with pip:
# $ pip install google-api-python-client

# Or, with Conda:
# $ conda install -c conda-forge google-api-python-client

from googleapiclient.discovery import build

my_api_key = "AIzaSyBnkRwluy5yu0ESYUPSrIgwzAvo_gbhSw4"
my_cse_id = "004723198555886821667:lr9hohyciei"

# Define a function to search google.
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    query = service.cse().list(q=search_term, cx=cse_id,**kwargs)
    result = query.execute()
    return result

result = google_search("Coffee", my_api_key, my_cse_id)

# Parse the result.
print(result)

