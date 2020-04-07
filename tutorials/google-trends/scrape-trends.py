#!/usr/bin/env python3

# Scape Google trends with python.
# See library documentation: 
# https://pypi.org/project/pytrends/

# Installation.
# $ pip install pytrends

# Connect to api.
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)

# Use of a proxy.
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25),
        proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1)


# Trending searches:
pytrends.trending_searches(pn='united_states') # trending searches in real time for United States

# Top charts.
pytrends.top_charts(date=2016, hl='en-US', tz=300, geo='GLOBAL')
#pytrends.trending_searches(pn='japan') # Japan

# Build a payload.
kw_list = ["oscars"]
# Results are returned weekly.
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

# Interest over time.
pytrends.interest_over_time()

# Related topics.
pytrends.related_topics()
