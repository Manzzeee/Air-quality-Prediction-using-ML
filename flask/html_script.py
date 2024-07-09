# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import time
import requests
import sys


def retrieve_html():
    for year in range(2013,2021):
        for month in range(1,13):
            if month<10:
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-421820.html'.format(month, year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-421820.html'.format(month, year)
            result_text = requests.get(url)
            result_text_utf = result_text.text.encode('utf=8')
        
            if not os.path.exists("data/html_data/{}".format(year)):
                os.makedirs("data/html_data/{}".format(year))
            
            with open("data/html_data/{}/{}.html".format(year,month),"wb") as output:
                output.write(result_text_utf)
            
        sys.stdout.flush()
        
if __name__ == "__main__":
    start = time.time()
    retrieve_html()
    stop = time.time()
    print("Time taken {}".format(stop-start))
        