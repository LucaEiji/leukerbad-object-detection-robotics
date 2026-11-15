# SO-101 — apprentissage d'une prise en simulation

Cette expérience entraîne réellement une politique SAC par renforcement dans
MuJoCo, sans démonstrations humaines et sans accès aux ports des bras réels.

## Ce qui est appris

La politique choisit quatre paramètres : décalage x, décalage y, hauteur et
orientation de prise. Un contrôleur de cinématique inverse exécute ensuite
l'approche, la fermeture des mâchoires et le levage. Cette trajectoire en phases
est programmée ; elle n'est pas apprise de bout en bout.

La politique reçoit la position et l'orientation exactes de l'objet fournies par
le simulateur. Elle n'utilise pas les images d'une caméra. Pour l'utiliser en réel,
il faudrait une estimation de pose de l'objet et une adaptation du contrôleur,
ou entraîner une nouvelle politique visuelle. Le résultat n'est donc pas un
checkpoint ACT directement déployable sur le robot de l'équipe.

## Scène et périmètre

- Modèle officiel SO-101 de MuJoCo Menagerie, moteurs et collisions physiques.
- Objet libre : capsule de 92 mm de long, diamètre 16 mm, masse approximative 26 g.
- Position autour de x=23 cm, y=0, avec variation de ±5 mm sur chaque axe.
- Orientation de l'objet uniforme entre −1 et +1 radian (environ ±57°).
- Masse, frottements, montage et éclairage ne sont pas identifiés sur le matériel réel.
- Gravité et contacts actifs, aucune contrainte de soudure/attache de l'objet.
- Réussite : objet levé de plus de 4 cm, encore en contact avec chacune des deux
  mâchoires après la phase de maintien. Les essais ne comprennent pas le dépôt.

La récompense combine l'alignement géométrique de la prise, la hauteur physique
atteinte et un bonus de réussite. Les paramètres de prise sont optimisés par SAC ;
le contrôleur géométrique n'est pas une politique experte utilisée pour entraîner.

## Exécution

Depuis ce dossier, avec l'environnement dédié déjà installé :

```bash
.venv/bin/python validate_physics.py
.venv/bin/python train.py --steps 8000
.venv/bin/python render_results.py
```

`train.py --resume --steps 4000` poursuit le dernier checkpoint. Les dépendances
exactes sont dans `requirements-lock.txt`. Les scripts n'importent aucun pilote
matériel, n'ouvrent aucun port série et ne commandent jamais le bras réel.

## Livrables

- `outputs/best.zip` : meilleure politique sur le lot de validation.
- `outputs/latest.zip` : dernier checkpoint.
- `outputs/progress.json` : résultats pendant l'entraînement.
- `outputs/random_baseline.json` : 100 prises aléatoires de comparaison.
- `outputs/heldout_evaluation.json` : 200 nouveaux essais, seeds 50000 à 50199.
- `outputs/physics_checks.json` : tests de déterminisme, contact, relâchement et stabilité.
- `outputs/simulation_result.mp4` : six premiers essais consécutifs du lot final,
  échecs compris ; pas de sélection des seules réussites.

Les lots d'entraînement, de validation et d'évaluation utilisent des seeds distincts.
Le périmètre de variation reste étroit ; le taux simulé ne mesure pas la réussite réelle.

## Attribution

Modèle et maillages : [MuJoCo Menagerie, robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101),
commit `370c43965323c0944b433b202aac5058b5f907a9`, dérivé du SO-101 de The Robot Studio.
La licence Apache-2.0 du modèle est conservée dans
`vendor/mujoco_menagerie/robotstudio_so101/LICENSE`.

Moteur physique : [MuJoCo](https://mujoco.readthedocs.io/).
Algorithme : [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html).
