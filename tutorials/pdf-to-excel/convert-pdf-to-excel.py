#!/usr/bin/env python3

#git clone https://github.com/pdftables/python-pdftables-api.git
#python setup.py install

my_api_key = "x4ayc1c4lzmy"

input_pdf = "M4-00103a.pdf"

#replace c.xlsx with c.csv to convert to CSV
#replace c.xlsx with c.xml to convert to XML
#replace c.xlsx with c.html to convert to HTML

import pdftables_api

output_file = input_pdf.split(".")[0] + ".xlsx"
c = pdftables_api.Client('my-api-key')
c.xlsx(input_pdf, output_file) 
