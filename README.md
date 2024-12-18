# Instut Ober de Catalunya
## Inteligencia Artificial i Big Data

Aquest es un projecte pràctic per treballar la EAC6 del modul m03.

Per executar el códi has de:
instalar paquets i preparar entorn de virtualizacio
```
pip install -r requirements.txt
```
generar dades
```
python .\generardataset.py
```
generar el model amb les dades entrenades
```
python .\clustersciclistes.py
```
aixecar en una terminal mlflow
```
mlflow ui
```
Executar els experiments
```
python .\mlflowtracking-K.py
```
