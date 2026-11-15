# Préparation du passage simulation → réel

Le checkpoint SAC existe. Le contrôle en lecture seule confirme que les six
moteurs répondent et que leur calibration correspond au fichier local.
La politique n'est pas encore autorisée à commander le matériel.

## Pourquoi le checkpoint ne se lance pas directement

- Le simulateur reçoit la pose exacte de l'objet. L'ordre binaire AUKEY ne la fournit pas.
- Les angles mécaniques du modèle MuJoCo ne sont pas automatiquement les angles
  normalisés de la calibration LeRobot. Les copier tels quels serait incorrect.
- La pince et le contrôleur réels ont du jeu et un écart de suivi déjà mesurés.

## Références physiques à recueillir

Depuis le dossier `lerobot`, lancer dans Terminal :

```bash
python teleop_wrist_offset.py --hardware --capture-reference
```

Positionner les deux bras comme aux essais réussis : follower pour sa caméra,
leader pour la main droite. Valider les poses, puis taper BOUGER.

Conserver la vape au même emplacement et dans la même orientation pour cette
première référence. Capturer avec `s` puis Entrée, en immobilisant le leader :

1. `rest` : repos habituel.
2. `above_object` : pince ouverte environ 5 cm au-dessus de l'objet.
3. `grasp_open` : pince ouverte autour de l'objet, sans appuyer sur la table.
4. `grasp_closed` : objet saisi, sans encore le lever.
5. `lift` : objet soulevé d'environ 3 cm.

Le programme sauvegarde progressivement les positions mesurées du follower et
les cibles corrigées du leader dans `reference_<date>.json`. Il ne crée pas un
dataset visuel d'imitation. Après cinq captures, il s'arrête avec maintien moteur.
Ctrl+C demande également un arrêt avec maintien ; `python release_follower.py --hardware`
libère volontairement le bras après l'avoir soutenu.

Ces références servent à vérifier un contrôleur de prise réel et une trajectoire
de secours. Elles ne suffisent pas, seules, à identifier la correspondance complète
entre les coordonnées 3D du modèle et les calibrations réelles : il faudra aussi
valider les conventions/signes des articulations, des poses géométriques connues
et la position de l'objet par rapport au socle.

## Étapes restantes avant un essai autonome

1. Vérifier les références physiques et la correspondance des articulations du modèle.
2. Mesurer une pose d'objet fixe par rapport au bras pour un premier essai, ou
   intégrer centre et angle issus de la reconnaissance après calibration caméra/table.
3. Comparer la trajectoire simulée à ces références et vérifier le dégagement réel.
4. Faire un essai réel supervisé, limité à une prise et un petit levage.

Aucun taux de réussite simulé ne valide ces étapes à lui seul.
