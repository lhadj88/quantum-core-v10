# R23A2 — Audit de continuité des trajectoires 5 minutes

**Portée :** disponibilité et intégrité uniquement. Aucun test T1/T2, aucune excursion et aucune inférence causale n’ont été exécutés.

- événements audités : **203**
- archives quotidiennes requises : **387**
- événements complets : **202**
- événements incomplets : **1**
- intervalles 5 minutes manquants : **18**
- échecs de hash sur jours événement : **0**
- règle d’arrêt originale R23A2 satisfaite : **False**

## Garde-fou multi-processus

Cet audit ne teste que la disponibilité des trajectoires nécessaires au mécanisme P2. Il ne recherche pas une route dominante et ne modifie le statut d’aucun autre procédé.

## Fenêtres incomplètes

- `2021-03-05T16:00:00Z` : 126/144 lignes, 18 manquantes, première lacune `2021-03-06T02:00:00+00:00`.
