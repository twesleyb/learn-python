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

# Imports.
import subprocess
from googleapiclient.discovery import build

# Define a function to get credentials from pass.
def get_pass_credentials(entry):
    ''' Get credentials from pass.
    Extracts a dictionary of key:value pairs from entry
    in password store. Expects delimiter to be ":".
    '''
    cmd = ["pass",entry]
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    output = process.communicate()
    result = list(output)[0].decode('utf-8').split("\n")
    result = [entry.split(": ") for entry in result]
    result = [res for res in result if res != ['']]
    credentials = {key: value for (key, value) in result}
    return credentials

# Define a function to search google.
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    query = service.cse().list(q=search_term, cx=cse_id,**kwargs)
    result = query.execute()
    return result

# Get credentials.
credentials = get_pass_credentials("google-search-cse-id")

# Search google.
query = "Tyler Bradshaw"
result = google_search(query, credentials.get('api_key'),credentials.get('cse_id'))

# Parse the result.
result.keys()

# Information about the search query.
result.get('queries')

# What type of search was done.
result.get('kind')

# Api url?
result.get('url')

# API name.
result.get('context')

# Statistics about search.
result.get('searchInformation')

# Actual responses:
# List of search results
response = result.get('items')

# Each entry in the list of responses is a dictionary.
r0 = response[0]
r0.keys()

# Most useful bits:
r0.get('title')
r0.get('link')
r0.get('snippet')
