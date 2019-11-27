import pandas as pd
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
import sys

import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('QT5Agg')



class GetTopics:

	def __init__(self):
		self.url = 'https://www.nzherald.co.nz/business/news/headlines.cfm?c_id=3'
		self.url_parts = self.url.split('/')
		self.cat = self.url_parts[3]
		self.base_url = self.url_parts[0]+'//'+self.url_parts[2]
		self.now = pd.datetime.now()
		self.stop = set(stopwords.words('english'))
		self.exclude = set(string.punctuation)
		self.lemma = WordNetLemmatizer()

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
		df.to_pickle('{}_{}_{}-{}-{}_{}-{}-{}.pkl'.format(self.url_parts[2].split('.')[1], self.cat, self.now.day, self.now.month, self.now.year, self.now.hour, self.now.minute, self.now.second))

	def read_file(self, file_name = None):
		if file_name == None:
			sys.exit('no file name, wtf')
		try:
			df = pd.read_pickle(file_name)
		except:
			sys.exit('No file {} found'.format(file_name))
		return df

	def clean(self, doc):
		stop_free = " ".join([i for i in doc.lower().split() if i not in self.stop])
		punc_free = ''.join(ch for ch in stop_free if ch not in self.exclude)
		normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
		return normalized

	def get_doc_clean(self, df):
		doc_clean = [self.clean(doc).split() for doc in df['Article Content']]
		return doc_clean

	def train_model(self, doc_clean, num_topics=8, passes=500, chunksize=5, update_every=0, eta='auto', iterations=10, randomstate=42):
		dictionary = corpora.Dictionary(doc_clean)
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		Lda = gensim.models.ldamodel.LdaModel
		perplexity_logger = PerplexityMetric(corpus=doc_term_matrix, logger='shell')
		ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes, chunksize=chunksize, update_every=update_every, eta=eta, iterations=iterations, random_state=randomstate, callbacks=[perplexity_logger])
		return ldamodel

	def frequency_filter(self,doc_clean, cutoff=0.1, plot=False):
		doc_clean_flat = [val for sublist in doc_clean for val in sublist]
		doc_clean_flat_df = pd.DataFrame({'words': doc_clean_flat})
		doc_clean_flat_df = doc_clean_flat_df.words.value_counts()
		doc_clean_flat_df = pd.DataFrame({'word': doc_clean_flat_df.index, 'count': doc_clean_flat_df}).reset_index(drop=True)
		doc_clean_flat_df['count_norm'] = doc_clean_flat_df['count'] / doc_clean_flat_df['count'].max()
		if plot == True:
			fig = plt.figure(figsize=(12, 8))
			plt.bar(x=doc_clean_flat_df[doc_clean_flat_df.count_norm > cutoff]['word'],
			        height=doc_clean_flat_df[doc_clean_flat_df.count_norm > cutoff]['count'])
			plt.xticks(rotation=90)
			plt.show()
		word_filter = doc_clean_flat_df[doc_clean_flat_df.count_norm > cutoff]['word']
		return word_filter

	def POS_filter(self, doc_clean, plot=False, drop_catagories=['IN','MD','CD']):
		POS_list = list()
		doc_clean_flat = [val for sublist in doc_clean for val in sublist]
		for item in doc_clean_flat:
			tokenized = nltk.word_tokenize(item)
			tagged = nltk.pos_tag(tokenized)

			chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>}"""
			chunkParser = nltk.RegexpParser(chunkGram)

			chunked = chunkParser.parse(tagged)
			POS_list.append(chunked)
		POS_list = [i[0] for i in POS_list]
		POS_cat = [i[1] for i in POS_list]
		if plot == True:
			fig = plt.figure(figsize=(12, 8))
			plt.bar(x=Counter(POS_cat).keys(), height=Counter(POS_cat).values())
			plt.xticks(rotation=90)
			plt.show()
		new_POS_list = []
		for word in POS_list:
			if word[1] not in drop_catagories:
				new_POS_list.append(word[0])
		POS_filter = list(dict.fromkeys(new_POS_list))
		return POS_filter

	def filter_words(self, doc_clean, frequency_filter=[], POS_filter=[], freq_cutoff=0.1):
		if (len(frequency_filter)>0) and (len(POS_filter)==0):
			print('Only using frequency filter')
			word_filter = self.frequency_filter(doc_clean, cutoff=freq_cutoff)
			word_filter = set(word_filter)
			doc_clean_2 = []
			for doc in doc_clean:
				doc_2 = [x for x in doc if x in word_filter]
				doc_clean_2.append(doc_2)
			return doc_clean_2
		if (len(frequency_filter)==0) and (len(POS_filter)>0):
			print('Only using POS filter')
			word_filter = self.frequency_filter(doc_clean, cutoff=freq_cutoff)
			word_filter = set(word_filter)
			doc_clean_2 = []
			for doc in doc_clean:
				doc_2 = [x for x in doc if x in word_filter]
				doc_clean_2.append(doc_2)
			return doc_clean_2
		if (len(frequency_filter)>0) and (len(POS_filter)>0):
			print('Using both frequency and POS filters')
			freq_filter = self.frequency_filter(doc_clean, cutoff=freq_cutoff)
			freq_filter = set(freq_filter)
			pos_filter = self.frequency_filter(doc_clean, cutoff=freq_cutoff)
			pos_filter = set(pos_filter)
			doc_clean_2 = []
			for doc in doc_clean:
				doc_2 = [x for x in doc if (x in freq_filter) and (x in pos_filter)]
				doc_clean_2.append(doc_2)
			return doc_clean_2
		else:
			print('Frequency and POS filters are both empty, not doing anything')
			return doc_clean

	def get_topics(self, file=False):
		if file == False:
			print('not using file, pulling from source')
			print('getting page')
			status, r1 = self.request_url()
			if status != 200:
				print('status is not 200, closing')
				exit()
			print('getting content')
			soup = self.get_content(r1=r1)
			print('extracting articles')
			df = self.pull_articles(soup=soup)
			print('saving to file')
			self.save_news(df=df)
			print('cleaning data')
			doc_clean = self.get_doc_clean(df=df)
			print('creating frequency filter')
			freq_filter = self.frequency_filter(doc_clean=doc_clean)
			print('creating POS filter')
			pos_filter = self.POS_filter(doc_clean=doc_clean)
			print('filtering data')
			doc_clean = self.filter_words(doc_clean=doc_clean, frequency_filter=freq_filter, POS_filter=pos_filter)
			print('training model')
			model = self.train_model(doc_clean=doc_clean)
		elif type(file) == str:
			if file.split('.')[1] == 'pkl':
				print('reading file {}'.format(file))
				df = self.read_file(file)
			else:
				sys.exit('file does not have pkl as file type')
			print('cleaning data')
			doc_clean = self.get_doc_clean(df=df)
			print('creating frequency filter')
			freq_filter = self.frequency_filter(doc_clean=doc_clean)
			print('creating POS filter')
			pos_filter = self.POS_filter(doc_clean=doc_clean)
			print('filtering data')
			doc_clean = self.filter_words(doc_clean=doc_clean, frequency_filter=freq_filter, POS_filter=pos_filter)
			print('training model')
			model = self.train_model(doc_clean=doc_clean)
		else:
			sys.exit('Unrecognised file type, acceptable file values are [False, str]')
		return model
