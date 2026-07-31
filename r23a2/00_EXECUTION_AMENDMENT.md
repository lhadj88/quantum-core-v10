# R23A2 — Amendment d’exécution gelé avant accès aux trajectoires

**ID :** `R23A2_EXECUTION_AMENDMENT_v0_1`  
**Date :** 31 juillet 2026  
**Statut :** gelé avant téléchargement et avant calcul statistique  
**Usage :** recherche uniquement ; aucune autorisation de trading.

## Garde-fou scientifique principal

R23A2 valide ou invalide uniquement le mécanisme partiel **P2 — budget de réversion déjà consommé avant 20:00 UTC**. Il ne recherche pas une route dominante, ne remplace pas les autres procédés et ne peut pas, seul, autoriser un classifieur directionnel global.

Un succès de P2 signifie seulement qu’un état temporel mesurable peut être intégré ultérieurement à une **phénoménologie causale multi-processus**. Un échec de P2 ne falsifie ni l’existence d’autres mécanismes, ni la phénoménologie multi-processus ; il ferme seulement cette représentation de P2.

## Définitions temporelles gelées

- `price_20utc` : prix d’ouverture de la bougie spot 5 minutes ouverte exactement à 20:00:00 UTC, équivalent au prix de clôture de la bougie 19:55–20:00 hors micro-écarts de continuité.
- fenêtre : 144 intervalles consécutifs de cinq minutes, de 20:00 inclus à 08:00 UTC exclu le lendemain ; dernier intervalle 07:55–08:00.
- première heure T1 : les douze intervalles 20:00–21:00.
- excursion contre-directionnelle : extrême intrabougie opposé au signe du choc — `low` pour un choc positif, `high` pour un choc négatif.
- excursion alignée : `high` pour un choc positif, `low` pour un choc négatif.
- temps de première-passage : borne droite de l’intervalle de cinq minutes qui franchit la barrière, soit 5, 10, …, 720 minutes ; censure administrative à 720 minutes.
- en cas d’égalité de maximum, retenir le premier intervalle.

## Source et intégrité

Les fichiers sont les archives quotidiennes officielles `data.binance.vision` prévues au préenregistrement : jour de l’événement et jour calendaire suivant. Les SHA-256 connus du jour événement sont ceux du ledger R18 ; les autres SHA-256 sont calculés et enregistrés. Les timestamps spot à partir de 2025 sont automatiquement reconnus comme microsecondes puis normalisés en millisecondes.

L’exécution s’arrête si les 144 timestamps attendus ne sont pas présents exactement et sans doublon.

## Inférence gelée

- 10 000 permutations de `P2_score` à l’intérieur de chaque année, graine `2302`.
- 10 000 bootstraps par grappes `année-mois`, rééchantillonnés séparément dans chaque année, graine `2302`.
- T1 : corrélation de Spearman, alternative unilatérale `rho < 0`.
- T2 : Cox à un paramètre, risques de base stratifiés par année, approximation de Breslow pour les ties, alternative unilatérale `beta < 0` / `HR < 1`.
- correction de Holm sur T1 et T2.
- les barrières secondaires 0,10 ; 0,50 ; 1,00 forment une famille séparée corrigée par Holm.

## Interprétation multi-processus

- T1 et T2 passent : P2 devient un **état de transition temporel admissible**, sans promotion directionnelle et sans dominance universelle.
- un seul passe : P2 reste mécanisme partiel ; aucune intégration prédictive.
- les deux échouent : P2 est rétrogradé à une association de route et cette formulation est fermée.

Les procédés P1, P3, P4 et P5 conservent leur statut propre. Leur échec R23A signifie que les coordonnées publiques testées ne les identifient pas, non qu’ils n’existent pas.
