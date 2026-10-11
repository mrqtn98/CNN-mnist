# Réseau de neurones convolutif en C — reconnaissance de chiffres manuscrits (MNIST)

Projet étudiant (2021), réalisé en binôme par Tony MARQUES et Hugo MIRANDA QUEIROS à l'ISAE-SupAéro de Toulouse. Il s'agit d'un réseau de neurones convolutif et son apprentissage par rétropropagation du gradient, programmés « à la main » en langage C, sans aucune bibliothèque d'apprentissage automatique (seule la bibliothèque mathématique standard est utilisée). Le réseau est entraîné et évalué sur la base MNIST : images en niveaux de gris de 28 × 28 pixels, 10 classes (chiffres 0 à 9), 60 000 images d'entraînement et 10 000 images de test.

Il s'agit d'un projet pédagogique : le code n'est pas optimisé et le réseau est volontairement très simple, loin de l'état de l'art (> 99 % sur MNIST).

Fichier README généré a posteriori grâce à Claude Sonnet 5.5 en 2026.

## Architecture (configuration par défaut)

| Étape | Détail | Taille de sortie |
|---|---|---|
| Entrée | niveaux de gris divisés par 255 | 28 × 28 |
| Convolution | 1 seul filtre 3 × 3, sans remplissage (*padding*), biais, sigmoïde | 26 × 26 |
| *Pooling* | moyenne sur blocs 2 × 2 sans recouvrement | 13 × 13 |
| Couche dense 1 | 169 → 128 neurones, sigmoïde | 128 |
| Couche dense 2 (sortie) | 128 → 10 neurones, sigmoïde | 10 |

- **Prédiction** : classe dont la sortie est maximale.
- **Coût** : entropie croisée binaire, sommée sur les 10 sorties (étiquette codée en *one-hot*).
- **Optimisation** : descente de gradient stochastique par mini-lots de 8 images tirées au hasard. Le taux d'apprentissage vaut 1,0 au départ puis est multiplié par 0,9999 à chaque itération. L'entraînement dure 25 000 itérations, soit 200 000 images vues (≈ 3,3 fois la taille de la base d'entraînement).
- **Initialisation** : poids tirés uniformément dans [−1 ; 1], biais nuls, graines fixes.

## Utilisation

1. Télécharger MNIST **au format CSV** (une image par ligne : l'étiquette puis les 784 pixels, sans ligne d'en-tête), par exemple depuis <https://pjreddie.com/projects/mnist-in-csv/>, et placer `mnist_train.csv` et `mnist_test.csv` dans le dossier `data/mnist/`.
2. Compiler et lancer (nécessite `gcc` et `make`) :

```
make
./cnn
```

Le programme affiche à chaque itération le coût et la précision sur 250 images de test, puis la précision finale sur les 10 000 images de test. Il écrit aussi un fichier CSV dans `data/` (voir `exemple/`).

## Résultat

Avec les paramètres fournis : **≈ 97 % de bonnes classifications sur les 10 000 images de test** (97,0 % lors du test de référence ; entre 96,6 % et 97,0 % sur 4 essais avec des graines différentes), en environ 8 minutes sur un seul cœur de processeur (machine de test).

## Remarques

- Les hyperparamètres (taux d'apprentissage, taille de lot, nombre d'itérations) ont été choisis en observant les résultats sur le jeu de test, sans jeu de validation séparé : la précision annoncée est donc légèrement optimiste.
- La précision affichée pendant l'entraînement porte sur 250 images de test toujours identiques ; c'est un simple indicateur de suivi. Le fichier CSV de sortie (colonnes `num_iter, cost, accur, time`) n'enregistre que les itérations où ce score est battu ; `time` est le temps de calcul cumulé de l'entraînement, en secondes.
- Avec une sortie sigmoïde et ce coût en simple précision (`float`), le calcul devient numériquement instable (`NaN`) dès qu'une sortie correcte sature à 1,0 exactement. C'est pourquoi le taux d'apprentissage décroît au cours de l'entraînement (constante `BETA`, dans `backprop_runner.h`) : lors d'un essai avec un taux constant de 0,5 (même architecture, une seule graine), le calcul a divergé vers 24 000 itérations.
- Les résultats peuvent varier légèrement d'une machine ou d'un compilateur à l'autre (génération pseudo-aléatoire, arrondis).
- Les paramètres se règlent au début de `main.c`, ainsi que dans `nn.h` (nombre de couches, fonction d'activation, fonction de coût) et `kernel.h` (constantes `PAD` et `POOL_SIZE`).
