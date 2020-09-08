#!/usr/bin/env python3

import sys
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.firefox.options import Options

# Paths to Tor-Firefox and geckodriver executables (Windows):
firefox = '/mnt/c/Users/User/Downloads/Tor Browser/Browser/firefox.exe'
gecko = '/mnt/c/Program Files/Mozilla Firefox/geckodriver.exe'

# Options.
options = Options()
options.headless = True

# Create the webdriver.
driver = webdriver.Firefox(options=options,executable_path=gecko)

# But, it isn't working through tor. 
driver.get('https://check.torproject.org') 
print(driver.title,file=sys.stderr)
