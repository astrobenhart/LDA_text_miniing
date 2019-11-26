import pandas as pd
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from tqdm import tqdm

import matplotlib
matplotlib.use('QT5Agg')



class GetTopics:
	
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()

	def __init__(self, url):
		self.url = 'https://www.nzherald.co.nz/business/news/headlines.cfm?c_id=3'
		self.url_parts = self.url.split('/')
		self.base_url = self.url_parts[0]+'//'+self.url_parts[2]
		self.now = pd.datetime.now()

	def request_url(self):
		r1 = requests.get(self.url)
		status = r1.status_code
		return status, r1

	def get_content(self, r1):
		coverpage = r1.content
		soup = BeautifulSoup(coverpage, 'html5lib')
		return soup

	def pull_articles(self, soup):
		coverpage_news = soup.find_all('article')

		news_contents = []
		news_dates = []
		list_links = []

		for n in tqdm(np.arange(0, len(coverpage_news))):
			link = coverpage_news[n].find('a')['href']
			list_links.append(link)

			try:
				article = requests.get(self.base_url + str(link))
			except:
				article = 0
			if article != 0:
				article_content = article.content
				soup_article = BeautifulSoup(article_content, 'html5lib')
				article_date = soup_article.find_all('div', class_='publish')
				body = soup_article.find_all('div', id='article-body')
				if body != []:
					x = body[0].find_all('p')
					for child in article_date[0].find_all("div"):
						child.decompose()
					article_date = article_date[0].get_text()
					list_paragraphs = []
					try:
						for p in np.arange(0, len(x)):
							paragraph = x[p].get_text()
							list_paragraphs.append(paragraph)
							final_article = " ".join(list_paragraphs)
					except:
						pass

			else:
				final_article = np.nan
				article_date = np.nan
			news_contents.append(final_article)
			news_dates.append(article_date)
		df_features = pd.DataFrame({'Article Content': news_contents, 'Article Link': list_links, 'Article Date': news_dates})
		df_features = df_features.dropna(axis=0)
		df_features = df_features.drop_duplicates(subset='Article Content', keep='first')
		df_features = df_features.reset_index(drop=True)
		return df_features

	def save_news(self, df):
		df.to_pickle('{}_business_{}-{}-{}_{}-{}-{}.pkl'.format(self.url_parts[2].split('.')[1], self.now.day, self.now.month, self.now.year, self.now.hour, self.now.minute, self.now.second))

	def read_file(self):
		file_name = '{}_business_{}-{}-{}_{}-{}-{}.pkl'.format(self.url_parts[2].split('.')[1], self.now.day, self.now.month, self.now.year, self.now.hour, self.now.minute, self.now.second)
		df = pd.read_pickle(file_name)
		return df

