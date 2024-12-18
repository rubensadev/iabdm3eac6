"""
@ IOC - CE IABD

Prova d'avaluació continua per reforzar les bones pràctiques
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

import numpy as np

def load_dataset(path: str) -> pd.DataFrame:
	"""
	Carrega el dataset de registres dels ciclistes

	Parameters
	----------
	path: str
		Directory d'on s'ha de llegir les dades dels ciclistes

	Returns
	----------
	pd.DataFrame
		Dades dels ciclistes llegides del document csv
	"""
	df = pd.read_csv(path, delimiter=',')
	return df

def EDA(df: pd.DataFrame):
	"""
	Exploratory Data Analysis del dataframe

	Parameters
	----------
	df: pd.DataFrame
		Dades dels ciclistes
	"""
	logging.debug('\n%s', df.shape)
	logging.debug('\n%s', df[:5])
	logging.debug('\n%s', df.columns)
	logging.debug('\n%s', df.info())

def clean(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	Parameters
	----------
	df: pd.DataFrame
		Dades dels ciclistes

	Returns
	----------
	pd.DataFrame:
		Dades dels ciclistes netejades (sense les columnes id, tt)
	"""
	df.drop(['id','tt'],axis=1,inplace=True)
	logging.debug('\nDataframe:\n%s\n...', df[:3])
	return df

def extract_true_labels(df: pd.DataFrame) -> np.ndarray:
	"""
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	Parameters
	----------
	df: pandas.DataFrame
		Dades dels ciclistes

	Returns
	-------
	numpy.ndarray:
		Array amb els tipus de ciclistes (al cluster que pertany el cicliste)
	"""
	return df['tipus'].values

def visualitzar_pairplot(df: pd.DataFrame):
	"""
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	Parameters
	----------
	df: pd.DataFrame
	 	Dades dels ciclistes a visualitzar
	"""
	# Generem el pairplot
	sns.pairplot(df)

	# Mostrem la gràfica
	plt.savefig('img/pairplot.png')

def clustering_kmeans(data: pd.DataFrame, n_clusters: int =4) -> KMeans:
	"""
	Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	Parameters
	----------
	data: pd.DataFrame
		Dades a clusteritzar: tp i tb

	Returns
	----------
	KMeans: 
		Model (objecte KMeans)
	"""
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	kmeans.fit(data)
	return kmeans

def visualitzar_clusters(data: pd.DataFrame, labels: np.ndarray):
	"""
	Visualitza els clusters en diferents colors. Provem diferents 
	combinacions de parells d'atributs

	Parameters
	----------
	data:
		El dataset sobre el qual hem entrenat
	labels:
		Array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)
	"""
	# Afegir les etiquetes al DataFrame
	ax = plt.subplot(1, 1, 1)
	sns.scatterplot(
		x='tp',
		y='tb',
		hue=labels,
		data=data,
		ax=ax
	)
	plt.savefig('./img/clusters.png')

def associar_clusters_patrons(tipus, model: KMeans):
	"""
	Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
	S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

	Parameters
	----------
	tipus:
		un array de tipus de patrons que volem actualitzar associant els labels
	model: KMeans
		Model KMeans entrenat

	Returns
	----------
	List:
		Array de diccionaris amb l'assignació dels tipus als labels
	"""
	# proposta de solució

	dicc = {'tp':0, 'tb': 1}

	logging.info('Centres:')
	for j in range(len(tipus)):
		logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'
			   .format(j, model.cluster_centers_[j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

	# Procés d'assignació
	ind_label_0 = ind_label_3 = 0
	def sum_cluster(cluster):
		return cluster[dicc['tp']]+cluster[dicc['tb']]

	suma_min = suma_max = sum_cluster(model.cluster_centers_[0])

	for j, center in enumerate(model.cluster_centers_, 1):
		sum_cluster_ = sum_cluster(center)
		if sum_cluster_ < suma_min:
			ind_label_0 = j
			suma_min = sum_cluster_
		if sum_cluster_ > suma_max:
			suma_max = sum_cluster_
			ind_label_3 = j


	tipus[0].update({'label': ind_label_0})
	tipus[3].update({'label': ind_label_3})

	lst = [0, 1, 2, 3]
	lst.remove(ind_label_0)
	lst.remove(ind_label_3)

	if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
		ind_label_1 = lst[0]
		ind_label_2 = lst[1]
	else:
		ind_label_1 = lst[1]
		ind_label_2 = lst[0]

	tipus[1].update({'label': ind_label_1})
	tipus[2].update({'label': ind_label_2})

	logging.info('\nHem fet l\'associació')
	logging.info('\nTipus i labels:\n%s', tipus)
	return tipus

def generar_informes(df: pd.DataFrame, tipus: dict):
	"""
	Generació dels informes a la carpeta informes/. 
	Tenim un dataset de ciclistes i 4 clústers, i generem 4 fitxers de ciclistes 
	per cadascun dels clústers

	Parameters
	----------
	df: pd.DataFrame
		Dades d'entrada
	tipus: Dict
		Diccionary que associa els patrons de comportament amb els labels dels clústers
	"""
	for t in tipus:
		label = t['name']
		df_t = df[df['t'] == label]
		file_path = f"./informes/{label}.pkl"
		df_t.to_csv(file_path, index=False)
	logging.info('S\'han generat els informes en la carpeta informes/\n')

def nova_prediccio(dades, model: KMeans) -> tuple:
	"""
	Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

	Parameters
	----------
	dades: List
		llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
	model: KMeans
	 	clustering model
	
	Returns
	-------
	tuple(df, int)
		(dades agrupades, predicció del model)
	"""
	df_nous_ciclistes = pd.DataFrame(dades, columns=['id','tp','tb','tt'])
	df_nous_ciclistes_clean = clean(df_nous_ciclistes)
	return (df_nous_ciclistes_clean, model.predict(df_nous_ciclistes))

# ----------------------------------------------

if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,  # Nivel de logging
		format='%(asctime)s - %(levelname)s - %(message)s',  # Formato
		handlers=[
			logging.StreamHandler()  # Mostrar logs en la consola
		]
	)


	path_dataset = './data/ciclistes.csv'
	path_model = './model/clustering_model.pkl'
	path_tipus = './model/tipus_dict.pkl'
	path_scores = './model/scores.pkl'
	
	df = load_dataset(path_dataset)
	EDA(df)
	df = clean(df)
	true_labels = extract_true_labels(df)
	df.drop('tipus', axis=1, inplace=True)
	visualitzar_pairplot(df)
	model = clustering_kmeans(df, 4)
	with open(path_model, "wb") as file:
		pickle.dump(model, file)
	pred_labels = model.predict(df)

	scores = {
		"homogeneity": homogeneity_score(true_labels, pred_labels),
		"completeness": completeness_score(true_labels, pred_labels),
		"v_measure": v_measure_score(true_labels, pred_labels)
	}

	logging.info("Guardant els scores del model kmeans")
	with open(path_scores, "wb") as file:
		pickle.dump(scores, file)
	visualitzar_clusters(df, true_labels)

	# array de diccionaris que assignarà els tipus als labels
	tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
	
	df['t'] = true_labels
	tipus_dict = associar_clusters_patrons(tipus, model)
	with open(path_tipus, "wb") as file:
		pickle.dump(tipus, file)
	generar_informes(df, tipus_dict)
	# Classificació de nous valors
	nous_ciclistes = [
		[500, 3230, 1430, 4670], # BEBB
		[501, 3300, 2120, 5420], # BEMB
		[502, 4010, 1510, 5520], # MEBB
		[503, 4350, 2200, 6550] # MEMB
	]
	
	(df_nous_ciclistes, pred) = nova_prediccio(nous_ciclistes, model)
	for i, p in enumerate(pred):
		t = [t for t in tipus_dict if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)

	