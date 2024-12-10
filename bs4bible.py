import requests
from bs4 import BeautifulSoup
import time

def getdata(url):
    r = requests.get(url)
    return r.text


def search(prompt):
    htmldata = getdata(f'https://www.bing.com/images/search?q={prompt.replace(' ', '+')}+bible')
    images = []
    soup = BeautifulSoup(htmldata, 'html.parser')
    time.sleep(1)
    for item in soup.find_all('img', class_='mimg'):
        try:
            images.append(item['src'])
        except Exception as e:
            continue

    return images


def searchmap(prompt):
    images = search(f'{prompt}+map')
    return images
