import requests
from bs4 import BeautifulSoup

url = "https://gorkhapatraonline.com/news/126236"

response = requests.get(url)
response.encoding = 'utf-8'  # Ensure UTF-8 encoding
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

def clean_text(soup):
    for a in soup.find_all('a', href=True):
        a.replace_with(a.text)
    text = soup.get_text(strip=True)
    return text.encode('utf-8').decode('utf-8')  # Ensure proper decoding

cleaned_text = clean_text(soup)
print(cleaned_text)

with open('demo.txt','w') as file:
    file.write(cleaned_text)