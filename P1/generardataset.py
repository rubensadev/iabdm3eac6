import os
import logging
from typing import Dict, List
import numpy as np



def generar_dataset(num: int, ind: int, tipus_ciclista: Dict) -> List[Dict]:
    """
    Genera dades sintétiques per la pujada i baixada del Port del Cantó.
	
	Parameters
	----------
    num: int
		Indica el número de files/ciclistes a generar
    ind: int
		Indica l'índex identificador del dorsal
    dicc: Dict
		Dicionary amb categoría de ciclista, temps mig de pujada, temps mig de baxada i sigma
	
	Returns
	-------
    List[Dict]
		Array numpy amb dades sintetiques
    """
    dades = []
    for i in range(num):
        pujada = int(np.random.normal(tipus_ciclista["mu_p"], tipus_ciclista["sigma"]))
        baixada = int(np.random.normal(tipus_ciclista["mu_b"], tipus_ciclista["sigma"]))
        dades.append([ind + i,pujada,baixada,pujada+baixada,tipus_ciclista['name']])
    return dades

if __name__ == "__main__":

	str_ciclistes = 'data/ciclistes.csv'


	# BEBB: bons escaladors, bons baixadors
	# BEMB: bons escaladors, mal baixadors
	# MEBB: mal escaladors, bons baixadors
	# MEMB: mal escaladors, mal baixadors

	# Port del Cantó (18 Km de pujada, 18 Km de baixada)
	# pujar a 20 Km/h són 54 min = 3240 seg
	# pujar a 14 Km/h són 77 min = 4268 seg
	# baixar a 45 Km/h són 24 min = 1440 seg
	# baixar a 30 Km/h són 36 min = 2160 seg
	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]

	dades = []
	samples = np.random.randint(1000, 2000)
	for case in dicc:
		ds = generar_dataset(samples, 1, case)
		dades +=ds

	with open(str_ciclistes, "w", encoding='utf-8') as file:
		file.write("id,tp,tb,tt,tipus\n")
		for ciclista in dades:
			linia = ",".join(map(str, ciclista))
			file.write(f"{linia}\n")


	logging.info("s'ha generat data/ciclistes.csv")
