#!/usr/bin/python3

import requests
from bs4 import BeautifulSoup as bs
import re

# Get the page's contents with beautifulsoup.
url = 'https://www.cmi.ac.in/~madhavan/courses/prog2-2012/docs/diveintopython3/iterators.html'
session = requests.session()
page = session.get(url)
soup = bs(page.content, 'lxml')

# Print links.
[link for link in soup.findAll('a', attrs={'href'})]

[link for link in soup.findAll('a', attrs={'href': re.compile("^http://")})]

