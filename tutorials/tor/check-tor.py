#!/usr/bin/env python3

# Randomize IP address with Tor.
# Tor must be running.
# Use start-tor utility in root/bin.

import requests
from torrequest import TorRequest

# Defaults.
TORPASS='<tor hashed pasword>'

# Add hashed password.
tr=TorRequest(password=TORPASS)

# Make a request.
response= requests.get('http://ipecho.net/plain')
print("My Original IP Address:",response.text)

# Reset Tor.
print("Reseting tor...")
tr.reset_identity()

# Make another request and check new ip.
response= tr.get('http://ipecho.net/plain')
print("New Ip Address",response.text)
