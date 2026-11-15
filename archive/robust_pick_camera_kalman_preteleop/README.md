# Robust pick — camera + Kalman (baseline pré-téléopération)

Cette archive est une **reconstruction commit-able** de l'étape visée :
centrage caméra USB, filtre de Kalman, tolérance à une occultation très brève,
et verrou exigeant une détection fraîche avant la descente. Elle sert à garder
ce chemin séparé du contrôleur courant.

## Limite de provenance

Cette version n'existait pas sous forme de commit ou de sauvegarde Git. Le dépôt
disponible ne contient que le commit de base LeRobot pour l'historique suivi ;
les scripts du hackathon étaient non suivis. Le contrôleur a donc été reconstruit
à partir des fichiers présents, des journaux et de la description de l'étape.
Ce n'est **pas** un checkout exact de la version d'hier.

La reconstruction retire le chargement de `teleop_grasp_pose.json` du contrôleur,
revient au pré-survol historique de 80 mm et à l'ouverture nominale de pince de
36 points, puis remet le verrou caméra décrit à cette étape à cinq corrections,
avec 3,5 mm d'erreur brute et 2,4 mm d'erreur filtrée maximum. Le code du
détecteur copié est celui disponible au moment de l'archivage; il contient des
améliorations ultérieures et ne peut pas être ramené exactement à son état
historique faute de snapshot source.

## Contenu

- `robust_pick.py`, `puff_vision.py`, `so101_ik.py`, `wrist_object.py` : boucle
  robuste, caméra/détection/Kalman et cinématique.
- `hardware_guard.py` : l'accès au matériel exige explicitement `--hardware`.
- `calibrations/` et `real_transfer/` : calibrations et références conservées.
  La trace et la pose téléopérées sont des éléments d'archive; le contrôleur
  pré-téléopération ne charge pas cette pose.
- `outputs/robust_pick_runs/` : journaux historiques copiés sans modification.
- `sim_pick/` : code MuJoCo, modèle SO-101 requis, checkpoints et sorties de
  simulation. L'environnement virtuel local n'est pas inclus.
- `tests_robust_pick.py` : tests présents au moment de la copie.

Les sorties MuJoCo (dont l'évaluation historique 200/200) caractérisent le
modèle simulé seulement; elles ne prouvent pas une prise réussie sur le bras.
La dernière tentative réelle connue a centré l'objet par vision, mais ne l'a
pas maintenu après fermeture et levée.

## Commandes hors matériel

Depuis ce dossier, avec les dépendances du projet installées :

```bash
python robust_pick.py --dry-run --x-mm 0 --y-mm 0
python tests_robust_pick.py
cd sim_pick
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python validate_physics.py
.venv/bin/python simulate.py --seed 50000 --policy best
```

Les commandes de simulation ne commandent pas le robot. Aucune commande réelle
n'est fournie ici comme recommandation de lancement : une prise matérielle exige
une personne présente au robot et une vérification indépendante de l'objet.

## Commit ciblé

Pour inclure uniquement cette archive dans un commit depuis la racine du dépôt :

```bash
git add archive/robust_pick_camera_kalman_preteleop
git diff --cached --stat
git status --short
```

Vérifier la liste préparée avant le commit. Ne pas ajouter globalement l'arbre :
il contient des suppressions préexistantes dans le clone LeRobot.
