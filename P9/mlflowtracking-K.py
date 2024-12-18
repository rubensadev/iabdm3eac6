""" @ IOC - Joan Quintana - 2024 - CE IABD """

import os
import pickle
import sys
import logging
import shutil
import mlflow

import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
	client = MlflowClient()
	experiment = "K sklearn ciclistes"

	exp = client.get_experiment_by_name(experiment)
	if not exp:
		exp = client.create_experiment(experiment)
		mlflow.set_experiment_tags(
			{
				"mlflow.note.content":"ciclistes variació de paràmetre k",
				"version": "1.0",
				"scikit-learn":"K"
			})
		mlflow.set_experiment(experiment)

	id = exp if isinstance(exp, str) else exp.experiment_id

	# netejar runs
	runs = MlflowClient().search_runs(experiment_ids=id)
	# for run in runs:
	# 	mlflow.delete_run(run.info.run_id)
	# 	shutil.rmtree(run.info.artifact_uri[7:-10])
	path = os.path.abspath('./data/ciclistes.csv')
	ds = load_dataset(path)
	ds = clean(ds)
	true_labels  = extract_true_labels(ds)
	ds = ds.drop('tipus', axis=1)
	
	with open('./model/clustering_model.pkl', 'rb') as file:
		model = pickle.load(file)
	
	for k in range(2,9):
		df = mlflow.data.pandas_dataset.from_pandas(ds, source=path)

		run = mlflow.start_run(experiment_id=id, description='K={}'.format(k))
		
		mlflow.log_input(df, context='training')

		kmodel = clustering_kmeans(ds, k)
		labels = kmodel.labels_
		h_score = round(homogeneity_score(true_labels, labels), 5)
		c_score = round(completeness_score(true_labels, labels), 5)
		v_score = round(v_measure_score(true_labels, labels), 5)

		tags = {
			"engineering": "RSA-IOC",
			"release.candidate" : "RC1",
			"release.version" : "0.0.1"
		}

		mlflow.set_tags(tags)

		mlflow.log_param("K",k)
		mlflow.log_metric("h", h_score)
		mlflow.log_metric("c", c_score)
		mlflow.log_metric("v_score", v_score)

		mlflow.log_artifact(path, artifact_path="data")
		mlflow.end_run()
		
	


	# TODO
	print('s\'han generat els runs')
