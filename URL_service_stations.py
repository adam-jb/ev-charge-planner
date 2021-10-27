"""Importing URLs of service stations for scraping"""
import requests
from bs4 import BeautifulSoup

""" Extracting soup"""

mso_url = "https://motorwayservicesonline.co.uk"

headers= {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
pageTree = requests.get(mso_url, headers=headers)
soup = BeautifulSoup(pageTree.content, 'html.parser')

#Edit as necessary- link to the main_roads.txt file which contains a list of roads that will be passed through the scraping function
roads_list_path = '/Users/camerongibb/Documents/OS_Hack/main_roads.txt'

with open(roads_list_path) as f:
    lines = f.readlines()
roads_list = [a_string.replace("\n", "") for a_string in lines]

road_url = "https://motorwayservicesonline.co.uk/" + "A1(M)"
pageTree = requests.get(road_url, headers=headers)
soup = BeautifulSoup(pageTree.content, 'html.parser')
# Finding names of service stations on single road
service_stations_road = []

def FindServiceStations_road(road_name):
    road_url = "https://motorwayservicesonline.co.uk/" + road_name
    pageTree = requests.get(road_url, headers=headers)
    soup = BeautifulSoup(pageTree.content, 'html.parser')
    #Finding names of service stations on single road
    service_stations_road = []
    spans = soup.find_all('span', {'class': 'mw-headline'})
    for span in spans:
        links = span.find_all('a')
        for link in links:
            print(link['href'])
            try:
                service_stations_road.append(link['href'])
            except:
                print('Error- loop 1')
    if len(service_stations_road) == 0: # If the webpage format is table, this should trigger
        tr_scrape = soup.find_all('tr')
        for tr in tr_scrape:
            try:
                links = tr.find_all('a')
                href = links[0]['href']
                service_stations_road.append(href)
            except:
                print('Error- loop 2 (minor)')
    return(service_stations_road)

service_stations = []

for roads in roads_list:
    print(roads)
    temp_ss_list = FindServiceStations_road(roads)
    print('sslist ', len(temp_ss_list))
    service_stations = service_stations + temp_ss_list
    print(temp_ss_list)

service_stations_unique = list(set(service_stations))
#URL output
service_stations_urls = [mso_url + s for s in service_stations_unique]

# Note- this also extracts some of the service station services (such as Waitrose and WHS Smith)

with open('/Users/camerongibb/Documents/OS_Hack/service_stations.txt', 'w') as f:
    for station in service_stations_urls:
        f.write('%s\n' % station)