#!/usr/bin/env python3

import urllib
import urllib3
import webbrowser


import requests

url = "http://duckduckgo.com/html"
payload = {'q':'python'}
r = requests.post(url, payload)

with open("requests_results.html", "w") as f:
    f.write(r.content)
