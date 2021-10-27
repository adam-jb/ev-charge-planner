import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

#Settings for scraping
headers= {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

"""Opening URL list to services"""
mso_url_filepath = "/Users/camerongibb/Documents/OS_Hack/service_stations.txt"
with open(mso_url_filepath) as f:
    lines = f.readlines()
url_list = [a_string.replace("\n", "") for a_string in lines]

#Initialising dataframe with 3 columns, URL, postcode and rating (out of 5)
ss_dataframe = pd.DataFrame([["", "", 0]])

"""TESTING AREA"""
# url = 'https://motorwayservicesonline.co.uk/Lisburn'
# print(url, len(url))
# # Scraping HTML
# pageTree = requests.get(url, headers=headers)
# soup = BeautifulSoup(pageTree.content, 'html.parser')
#
# rating_script = soup.find_all('div', {'class': 'infobyte1'})
# rating_contents = rating_script[1].contents[2].contents[0]
# rating = float(rating_contents[0:3])


"""Scraping loop 1
Iterating the scrape process for each URL and concatanating to dataframe"""
for urls in url_list:
    url = urls
    print(url, len(url))
    #Scraping HTML
    pageTree = requests.get(url, headers=headers)
    soup = BeautifulSoup(pageTree.content, 'html.parser')
    try:
        #Extracting postcode
        url_p = soup.find_all('p')
        postcode = url_p[4].contents[0][:-1]
        postcode = postcode.split()
        postcode = (postcode[0] + ' ') + postcode[1]
        print(postcode, ' ', postcode.isupper())
        if not postcode.isupper():
            postcode = 'NA'
    except:
        try:
            url_p = soup.find_all('p')
            postcode = url_p[5].contents[0][:-1]
            postcode = postcode.split()
            postcode = (postcode[0] + ' ') + postcode[1]
            print(postcode, ' correct format: ', postcode.isupper())
            if not postcode.isupper():
                postcode = 'NA'
        except:
            postcode = 'NA'
            print('Error in extracting postcode- returning NA')
    try:
        #Extracting rating (1 dp)
        # rating_script = soup.find_all('script')
        # rating_contents = rating_script[0].contents[0] # Returns a string- can use rfind and extract the index and deduct 2 for rating
        # rating = rating_contents[rating_contents.rfind("star services") - 2]
        #Extracts 2 dp rating
        rating_script = soup.find_all('div', {'class': 'infobyte1'})
        rating_contents = rating_script[1].contents[2].contents[0]
        rating = float(rating_contents[0:4])
    except:
        rating = 'NA'
        print('Error in extracting rating- returning NA')
    #Concatanating to existing dataframe
    data = pd.DataFrame([[url, postcode, rating]])
    ss_dataframe = pd.concat([ss_dataframe, data])

"""Formatting and writing output to csv"""
ss_dataframe = ss_dataframe.iloc[1:,:]
ss_dataframe = ss_dataframe.rename(columns={0: 'URL', 1: 'Postcode', 2: 'Rating/5'})
#Set to correct path
ss_dataframe.to_csv('/Users/camerongibb/Documents/OS_Hack/service_stations_rated.csv')

print('ScrapeRatings complete!')