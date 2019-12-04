#!/usr/bin/env python3
'''
Search for a beer on BeerAdvocate.com and get its rating.
'''

# Imports.
import sys
import argparse
import requests
from bs4 import BeautifulSoup as bs

# Parse the user's query.
ap = argparse.ArgumentParser()
ap.add_argument("query",help="Users search query for BeerAdvocate.com")
args = vars(ap.parse_args())
query = args["query"].replace(" ","+") 

# Start a session.
session = requests.session()

# Request the page you want to scrape.
url = "https://www.beeradvocate.com/search/?q="
page = session.get(url+query)

# Parse the webpage.
soup = bs(page.content,"lxml")

# Get all 'body' tags, these are our results.
results = soup.find_all('b') 

# Total number of results.
nbeers = int(results[0].text.split()[2])
print("{} beers found!".format(nbeers))

# Terminate if no beers found.
if nbeers == 0:
    sys.exit("Error, no beers were found!")

# Top 10 results.
print("Top result:")
top10 = [e.text for e in results[1:11]]
print(top10[0])

# Details for top result.
deets = [e.text for e in soup.find_all('span',{"class": "muted"})]
print(deets[0])
