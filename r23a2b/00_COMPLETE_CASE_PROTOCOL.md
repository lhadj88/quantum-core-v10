# R23A2B — Protocole complete-case gelé avant statistiques

**ID :** `R23A2B_P2_TEMPORAL_BUDGET_COMPLETE_CASE_v0_1`  
**Statut :** préenregistré avant toute statistique sur les trajectoires P2  
**Usage :** recherche uniquement ; aucune autorisation de trading.

## 1. Séparation avec R23A2 v0.1

R23A2 v0.1 est fermé avec le verdict `DATA_AVAILABILITY_STOP_BEFORE_STATISTICS`. Sa règle exigeait 203 trajectoires intégralement continues ; l’audit indépendant a trouvé une fenêtre incomplète. Aucun test T1/T2 n’a été exécuté.

R23A2B est une branche sœur distincte. Elle ne réécrit pas le résultat de R23A2 et ne modifie ni P2, ni les cibles, ni les barrières, ni les gates.

## 2. Règle d’admissibilité objective

Un événement est admissible si et seulement si les archives officielles Binance permettent de reconstruire exactement 144 timestamps uniques de cinq minutes entre 20:00 UTC inclus et 08:00 UTC exclu, sans imputation ni raccord avec une autre place.

Cette règle a été appliquée avant toute statistique de trajectoire. Elle exclut uniquement :

```text
2021-03-05T16:00:00Z
```

La fenêtre possède 126/144 intervalles ; 18 bougies officielles sont absentes entre 02:00 et 03:30 UTC le 6 mars 2021.

## 3. Tribunal inchangé

Les 202 événements complets conservent exactement :

- le score P2 gelé par R23A ;
- l’échelle `abs(event_ret_4h) + retreat_from_extreme` ;
- la fenêtre 20:00 → +12 h ;
- T1, T2, les barrières et les horizons préenregistrés ;
- 10 000 permutations intra-annuelles ;
- 10 000 bootstraps par grappes année-mois ;
- la correction de Holm ;
- les règles de décision et d’arrêt.

Toute nouvelle lacune ou divergence de hash arrête R23A2B.

## 4. Garde-fou multi-processus

R23A2B ne teste que la cohérence temporelle de P2 — la quantité de réversion déjà consommée avant 20:00 et le budget restant. P2 ne peut devenir qu’un état de transition au sein d’une phénoménologie causale multi-processus.

Même en cas de succès :

- P2 n’est pas une route dominante ;
- les procédés d’épuisement, absorption, continuation, libération retardée et ambiguïté restent distincts ;
- aucun classifieur directionnel global n’est autorisé ;
- `R23B_authorized = false`.
