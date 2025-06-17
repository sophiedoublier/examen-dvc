# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

Ajout Sophie DOUBLIER
Lorsque j'ai crée un nouvel environnement, le fichier requirements.txt ne semble pas avoir été inclus dans le depot GitHuk que j'ai forké. Je le crée partant du fichier requirements proposé dans le cours (Template_accidents_MLOps) et je retire les outils inutiles quite à les installer si besoin lors de l'éxécution de mon script.

Pour le choix du modèle, j'ai choisi le Gradient Boosting et j'aurais pu prendre le Random Forest car ces deux modèles semblent particulièrement adaptés aux jeux de données tabulaires avec des relations potentiellement non linéaires entre les variables explicatives et la variable cible. GradientBoosting va réduire à la fois le biais et la variable du modèle. Il est à privilégier pour des taches de régression sur des données industrielles comme la prédiction de la concentration de silice. 
