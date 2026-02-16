# Modell-Speicherung in AlphaZero Light

## Wie werden Modelle gespeichert?

### Verzeichnisstruktur

```
checkpoints/
├── gomoku_9x9/              # 9x9 Gomoku Modelle
│   ├── model_0.pt           # Iteration 0
│   ├── model_1.pt           # Iteration 1
│   ├── optimizer_0.pt       # Optimizer State
│   ├── optimizer_1.pt
│   └── training_history.json # Metriken
├── gomoku/                  # 15x15 Gomoku Modelle
│   └── ...
└── (tictactoe models)       # TicTacToe Modelle
```

### Was wird gespeichert?

#### 1. **Model Checkpoints** (`model_X.pt`)
- **Inhalt**: Alle Gewichte des neuronalen Netzes
- **Format**: PyTorch State Dict
- **Größe**: ~1-5 MB (abhängig von Netzwerkgröße)
- **Wann**: Nach jeder Iteration

**Was ist drin?**
- Convolutional Layer Gewichte
- Residual Block Parameter
- Policy Head Gewichte (für Zugwahrscheinlichkeiten)
- Value Head Gewichte (für Gewinnvorhersage)

#### 2. **Optimizer State** (`optimizer_X.pt`)
- **Inhalt**: Adam Optimizer Zustand
- **Warum**: Zum Fortsetzen des Trainings
- **Enthält**: Momentum, adaptive Learning Rates

#### 3. **Training History** (`training_history.json`)
```json
{
  "iterations": [0, 1, 2, ...],
  "total_loss": [2.5, 2.1, 1.8, ...],
  "policy_loss": [1.2, 1.0, 0.9, ...],
  "value_loss": [1.3, 1.1, 0.9, ...],
  "eval_win_rate": [0.3, 0.5, 0.7, ...],
  "eval_wins": [3, 5, 7, ...],
  "eval_losses": [7, 5, 3, ...],
  "eval_draws": [0, 0, 0, ...]
}
```

## Wie funktioniert das Laden?

### Automatisches Laden im UI
```python
# In app.py
checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
if checkpoints:
    latest_checkpoint = checkpoints[-1]  # Nimmt das neueste
    model.load_state_dict(torch.load(latest_checkpoint))
```

### Manuelles Laden
```python
# Spezifisches Modell laden
model.load_state_dict(torch.load("checkpoints/gomoku_9x9/model_5.pt"))
```

## Training fortsetzen

Das Training setzt **automatisch** fort, wenn Checkpoints existieren:
1. Lädt das neueste Modell
2. Lädt den Optimizer State
3. Startet bei der nächsten Iteration

## Vergleich der Modelle

| Spiel | Board | Modell | Checkpoints |
|-------|-------|--------|-------------|
| TicTacToe | 3×3 | Klein (4 blocks, 64 hidden) | `checkpoints/` |
| Gomoku 9×9 | 9×9 | Mittel (4 blocks, 64 hidden) | `checkpoints/gomoku_9x9/` |
| Gomoku 15×15 | 15×15 | Groß (8 blocks, 128 hidden) | `checkpoints/gomoku/` |

## Nützliche Befehle

### Checkpoints ansehen
```bash
ls -lh checkpoints/gomoku_9x9/
```

### Training History ansehen
```bash
cat checkpoints/gomoku_9x9/training_history.json | jq
```

### Altes Modell löschen (Neustart)
```bash
rm -rf checkpoints/gomoku_9x9/
```

## Warum separate Branches?

**Branch `gomoku-9x9`**:
- Schnelles Experimentieren
- Kleine Modelle
- Kurze Trainingszeiten (~15 Minuten)

**Branch `main`**:
- Produktionsreife Modelle
- Vollständige 15×15 Implementation
- Längere Trainingszeiten

So kannst du schnell testen ohne die Hauptversion zu beeinflussen! 🚀
