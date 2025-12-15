import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

courses = [
    "Endocrinologie",
    "Uro-néphrologie",
    "Cardiovasculaire",
    "Rhumatologie",
    "Respiratoire non oncologique",
    "Respiratoire oncologique / Plèvre",
    "Psychiatrie",
    "Ophtalmologie",
]
C = len(courses)
idx = {c:i for i,c in enumerate(courses)}

# ⚠️ Vérifie que ton history contient bien TOUTES les années que tu veux (2015-2016 + rattrapage si tu veux)
history = {
    2016: ["Cardiovasculaire","Psychiatrie","Respiratoire oncologique / Plèvre"],
    2017: ["Endocrinologie","Uro-néphrologie"],
    2018: ["Endocrinologie","Uro-néphrologie","Psychiatrie"],
    2019: ["Endocrinologie","Endocrinologie","Uro-néphrologie","Rhumatologie","Respiratoire oncologique / Plèvre"],
    2020: ["Endocrinologie","Uro-néphrologie","Rhumatologie"],
    2021: ["Endocrinologie"],
    2022: ["Endocrinologie","Cardiovasculaire","Rhumatologie"],
    2023: ["Cardiovasculaire","Respiratoire non oncologique","Ophtalmologie"],
    2024: ["Endocrinologie","Respiratoire non oncologique","Rhumatologie"],
    2025: ["Uro-néphrologie","Psychiatrie","Respiratoire oncologique / Plèvre"],
}

def build(year, unseen_age=10):
    freq = np.zeros(C, dtype=np.float32)
    last = np.full(C, -1, dtype=np.int32)

    for y in sorted(history.keys()):
        if y < year:
            for c in history[y]:
                i = idx[c]
                freq[i] += 1.0
                last[i] = y

    age = np.array([year - last[i] if last[i] != -1 else unseen_age for i in range(C)], dtype=np.float32)

    # ⚠️ Normalisation simple (très utile avec peu de data)
    freq_n = freq / (freq.max() if freq.max() > 0 else 1.0)
    age_n  = age  / (age.max()  if age.max()  > 0 else 1.0)

    X = np.stack([freq_n, age_n], axis=1).reshape(-1)  # 16 dims

    y = np.zeros(C, dtype=np.float32)
    for c in dict.fromkeys(history.get(year, [])):  # uniques
        y[idx[c]] = 1.0

    return X, y

train_years = [2022, 2023, 2024, 2025]
X_train, Y_train = [], []
for y in train_years:
    x, t = build(y)
    X_train.append(x); Y_train.append(t)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)

class TinyMultiLabel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(12, C)   # logits
        )
    def forward(self, x):
        return self.net(x)

model = TinyMultiLabel()

# BCEWithLogitsLoss = stable (sigmoid intégré)
crit = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-4)

model.train()
for epoch in range(30000):
    opt.zero_grad()
    logits = model(X_train)
    loss = crit(logits, Y_train)
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss={loss.item():.4f}")

def top3_from_scores(scores):
    order = np.argsort(-scores)[:3]
    return [courses[i] for i in order]

print("\n=== Fit check (TOP-3) sur 2022–2025 ===")
model.eval()
with torch.no_grad():
    logits = model(X_train)
    scores = torch.sigmoid(logits).cpu().numpy()  # scores [0,1] indépendants

for i, y in enumerate(train_years):
    pred = top3_from_scores(scores[i])
    true = list(dict.fromkeys(history[y]))
    print(f"{y}: Pred TOP-3={pred} | True={true}")

# ---- Predict 2026 ----
x26, _ = build(2026)
X_2026 = torch.tensor(x26, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    logits26 = model(X_2026)
    scores26 = torch.sigmoid(logits26).cpu().numpy()[0]

print("\n=== Prédiction 2026 ===")
print("TOP-3 prédits:", top3_from_scores(scores26))
print("\nScores (triés):")
for c, s in sorted(zip(courses, scores26), key=lambda x: -x[1]):
    print(f"{c:35s} {s:.3f}")
