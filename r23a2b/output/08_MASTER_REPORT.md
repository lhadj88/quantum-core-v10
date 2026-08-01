# R23A2B — Rapport maître

**Verdict :** `P2_ROUTE_ASSOCIATION_ONLY_MECHANISM_FORMULATION_CLOSED`  
**Événements :** 202  
**Usage :** recherche uniquement ; aucune autorisation de trading.

## Résultats primaires

| test   |   estimate | metric       |    ci_low |   ci_high |   p_one_sided |   p_holm |   years_direction_pass | gate_pass   |        beta |    score_z |
|:-------|-----------:|:-------------|----------:|----------:|--------------:|---------:|-----------------------:|:------------|------------:|-----------:|
| T1     |  0.0409981 | Spearman_rho | -0.100518 |  0.172799 |      0.580042 |        1 |                      3 | False       | nan         | nan        |
| T2     |  1.03719   | hazard_ratio |  0.883218 |  1.20172  |      0.717728 |        1 |                      3 | False       |   0.0365146 |   0.571094 |

## Interprétation canonique

R23A2B ne teste qu’un mécanisme partiel : le budget de réversion déjà consommé avant 20:00 UTC. Même si les deux tests passent, P2 n’est ni une route dominante ni une loi universelle. Il devient seulement un état de transition admissible dans une future phénoménologie causale multi-processus.

Les mécanismes d’épuisement, d’absorption, de continuation, de libération retardée et d’ambiguïté restent distincts. Leur représentation devra être recherchée séparément avec des observables causaux adaptés.

`R23B_authorized = false` : aucune procédure directionnelle globale n’est promue par ce tribunal.
