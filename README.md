# predictcoursSemioDeepL
Prédire O3 Sémio, Selon mes prédictions, cela devrait être Endoc, Cardiovasc, et (Rhumatologie/resp non Onco)
## 🧠 Entraînement du modèle

- **Epoch final** : 29 800  
- **Loss finale** : **0.4071**

---

## ✅ Vérification de l’apprentissage (TOP-3)

Comparaison entre les **prédictions du modèle** et les **cours réellement tombés**.

### 📅 Année 2022
- **Prédit (TOP-3)** : Cardiovasculaire · Endocrinologie · Rhumatologie  
- **Réel** : Endocrinologie · Cardiovasculaire · Rhumatologie  
✔️ *Correspondance parfaite (ordre non significatif)*

### 📅 Année 2023
- **Prédit (TOP-3)** : Cardiovasculaire · Respiratoire non oncologique · Ophtalmologie  
- **Réel** : Cardiovasculaire · Respiratoire non oncologique · Ophtalmologie  
✔️ *Correspondance parfaite*

### 📅 Année 2024
- **Prédit (TOP-3)** : Respiratoire non oncologique · Rhumatologie · Endocrinologie  
- **Réel** : Endocrinologie · Respiratoire non oncologique · Rhumatologie  
✔️ *Correspondance parfaite*

### 📅 Année 2025
- **Prédit (TOP-3)** : Respiratoire oncologique / Plèvre · Psychiatrie · Uro-néphrologie  
- **Réel** : Uro-néphrologie · Psychiatrie · Respiratoire oncologique / Plèvre  
✔️ *Correspondance parfaite*

---

## 🔮 Prédiction pour 2026

### 🎯 **TOP-3 prédits**
1. **Endocrinologie**
2. **Cardiovasculaire**
3. **Rhumatologie**

---

## 📊 Scores de probabilité (triés)

| Cours | Score |
|------|-------|
| Endocrinologie | **0.750** |
| Cardiovasculaire | **0.723** |
| Rhumatologie | **0.673** |
| Respiratoire non oncologique | 0.443 |
| Ophtalmologie | 0.081 |
| Uro-néphrologie | 0.040 |
| Psychiatrie | 0.038 |
| Respiratoire oncologique / Plèvre | 0.029 |

---

### 🧩 Interprétation
- Le modèle **reconstruit parfaitement** les années 2022 → 2025 (TOP-3 exact).
- La prédiction 2026 met en avant les **chapitres structurels majeurs** :
  - **Endocrinologie**
  - **Cardiovasculaire**
  - **Rhumatologie**

➡️ Résultat cohérent avec l’analyse heuristique et les graphiques fréquence × ancienneté:
<img width="889" height="589" alt="prédiction 2025" src="https://github.com/user-attachments/assets/b77a53b9-46f4-4b3c-88b5-3fc8e47686b7" />




