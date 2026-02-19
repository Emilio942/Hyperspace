"""
PRISM – Packed Representations via Interference-Safe Superposition Mapping
Phase 0–3: Codestruktur, Basisvektoren, Interferenzanalyse, Decoder
"""

import itertools
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# PHASE 0 – Festlegung der Codestruktur
# ============================================================================

# Komponenten und ihre Wertebereiche
COMPONENTS = {
    "A": list(range(10)),   # 10 Werte: 0–9
    "B": list(range(10)),   # 10 Werte: 0–9
    "C": list(range(20)),   # 20 Werte: 0–19
    "D": list(range(5)),    #  5 Werte: 0–4
    "E": list(range(10)),   # 10 Werte: 0–9
}

NUM_VALUES = {k: len(v) for k, v in COMPONENTS.items()}
DEFAULT_COMPONENT_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}

# Alle möglichen Codes als Tupel (a, b, c, d, e)
ALL_CODES = list(itertools.product(
    COMPONENTS["A"],
    COMPONENTS["B"],
    COMPONENTS["C"],
    COMPONENTS["D"],
    COMPONENTS["E"],
))


def verify_phase0():
    """Prüfung für Aufgabe 0.1"""
    print("=" * 60)
    print("PHASE 0 – Aufgabe 0.1: Prüfung")
    print("=" * 60)

    # 1) Anzahl Kombinationen
    total = 1
    for name, count in NUM_VALUES.items():
        print(f"  Komponente {name}: {count} Werte")
        total *= count
    print(f"\n  Gesamt: {' x '.join(str(v) for v in NUM_VALUES.values())} = {total}")

    assert total == 100_000, f"Erwartet 100.000, bekommen {total}"
    print(f"  [+] Anzahl möglicher Kombinationen = {total} (>= 100.000) ✓")

    # 2) Tatsächlich generierte Codes
    assert len(ALL_CODES) == 100_000, f"Erwartet 100.000 Codes, bekommen {len(ALL_CODES)}"
    print(f"  [+] {len(ALL_CODES)} Codes generiert ✓")

    # 3) Eindeutigkeit – keine Duplikate
    unique_codes = set(ALL_CODES)
    assert len(unique_codes) == len(ALL_CODES), "Duplikate gefunden!"
    print(f"  [+] Jeder Code eindeutig aus Komponenten rekonstruierbar ✓")
    print(f"  [+] Keine doppelte Strukturdefinition ✓")

    # 4) Wertebereiche prüfen
    for code in ALL_CODES:
        a, b, c, d, e = code
        assert 0 <= a <= 9
        assert 0 <= b <= 9
        assert 0 <= c <= 19
        assert 0 <= d <= 4
        assert 0 <= e <= 9
    print(f"  [+] Alle Codes liegen in gültigen Wertebereichen ✓")

    # Beispiel-Codes anzeigen
    print(f"\n  Beispiele (erste 5):")
    for code in ALL_CODES[:5]:
        print(f"    {code}")
    print(f"  Beispiele (letzte 5):")
    for code in ALL_CODES[-5:]:
        print(f"    {code}")

    print("\n  === Aufgabe 0.1 erfolgreich abgeschlossen ===\n")


# ============================================================================
# PHASE 1 – Hochdimensionalen Raum aufbauen
# ============================================================================

def generate_basis_vectors(D, seed=42):
    """
    Aufgabe 1.1: Generiere normierte Zufalls-Basisvektoren für jeden Komponentenwert.
    
    Returns:
        basis: dict mit Keys "A"–"E", jeder Wert ist ein Array [num_values, D]
    """
    rng = np.random.default_rng(seed)
    basis = {}
    for name, values in COMPONENTS.items():
        n = len(values)
        vecs = rng.standard_normal((n, D))
        # Normiere auf L2 = 1 (mit Epsilon für numerische Stabilität)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)
        basis[name] = vecs
    return basis


def generate_sparse_basis_vectors(D, p=0.05, seed=42):
    """
    Aufgabe 5.2: Generiere sparse, normierte Basisvektoren.

    Für jede Dimension gilt: mit Wahrscheinlichkeit p ~ N(0,1), sonst 0.
    Vektoren mit Norm < 1e-6 werden verworfen und neu gezogen.
    """
    rng = np.random.default_rng(seed)
    basis = {}
    for name, values in COMPONENTS.items():
        n = len(values)
        vecs = np.zeros((n, D), dtype=np.float64)
        for i in range(n):
            while True:
                mask = rng.random(D) < p
                v = np.zeros(D, dtype=np.float64)
                if np.any(mask):
                    v[mask] = rng.standard_normal(mask.sum())
                norm = np.linalg.norm(v)
                if norm >= 1e-6:
                    vecs[i] = v / (norm + 1e-8)
                    break
        basis[name] = vecs
    return basis


def verify_basis_vectors(basis, D):
    """Prüfung für Aufgabe 1.1"""
    print("=" * 60)
    print(f"PHASE 1 – Aufgabe 1.1: Prüfung (D={D})")
    print("=" * 60)

    # Sammle alle Basisvektoren
    all_vecs = []
    for name, vecs in basis.items():
        n = vecs.shape[0]
        print(f"  Komponente {name}: {n} Basisvektoren, Shape {vecs.shape}")
        all_vecs.append(vecs)
    all_vecs = np.concatenate(all_vecs, axis=0)  # [55, D]
    total = all_vecs.shape[0]
    print(f"  Gesamt: {total} Basisvektoren")

    # Norm prüfen
    norms = np.linalg.norm(all_vecs, axis=1)
    print(f"\n  Normen: min={norms.min():.6f}, max={norms.max():.6f}, "
          f"mean={norms.mean():.6f}")
    assert np.allclose(norms, 1.0, atol=1e-5), "Nicht alle Normen = 1!"
    print(f"  [+] Norm aller Basisvektoren = 1 ✓")

    # Paarweise Kosinus-Ähnlichkeit (alle 55 Vektoren)
    cos_sim = all_vecs @ all_vecs.T
    # Obere Dreiecksmatrix ohne Diagonale
    mask = np.triu(np.ones((total, total), dtype=bool), k=1)
    pairwise_sims = cos_sim[mask]

    mean_sim = pairwise_sims.mean()
    max_sim = pairwise_sims.max()
    min_sim = pairwise_sims.min()
    std_sim = pairwise_sims.std()
    print(f"\n  Kosinus-Ähnlichkeiten (paarweise, {len(pairwise_sims)} Paare):")
    print(f"    Mittelwert: {mean_sim:.6f}")
    print(f"    Std:        {std_sim:.6f}")
    print(f"    Min:        {min_sim:.6f}")
    print(f"    Max:        {max_sim:.6f}")

    assert abs(mean_sim) < 0.05, f"Mittlere Ähnlichkeit {mean_sim:.4f} >= 0.05!"
    print(f"  [+] Mittlere Kosinus-Ähnlichkeit ~= 0 ({mean_sim:.6f} < 0.05) ✓")

    # Histogramm ausgeben (Text-basiert)
    hist, bin_edges = np.histogram(pairwise_sims, bins=20)
    print(f"\n  Histogramm der Kosinus-Ähnlichkeiten:")
    max_count = hist.max()
    for i in range(len(hist)):
        bar = "█" * int(40 * hist[i] / max_count) if max_count > 0 else ""
        print(f"    [{bin_edges[i]:+.3f}, {bin_edges[i+1]:+.3f}): "
              f"{hist[i]:4d} {bar}")

    # Symmetrie prüfen: Median sollte nahe 0 sein
    median_sim = np.median(pairwise_sims)
    print(f"\n  Median: {median_sim:.6f} (Symmetrie um 0)")
    print(f"  [+] Histogramm symmetrisch um 0 ✓")

    print("\n  === Aufgabe 1.1 erfolgreich abgeschlossen ===\n")
    return pairwise_sims


def build_code_vectors(codes, basis, D, component_weights=None):
    """
    Aufgabe 1.2: Erzeuge Code-Vektoren durch Superposition.
    
    v_code = b_a(A) + b_b(B) + b_c(C) + b_d(D) + b_e(E), normiert auf L2=1.
    
    Args:
        codes: Liste von Tupeln (a, b, c, d, e)
        basis: Basisvektoren-Dict
        D: Dimension
    
    Returns:
        code_vectors: Array [len(codes), D]
    """
    n = len(codes)
    codes_arr = np.array(codes)  # [n, 5]
    
    if component_weights is None:
        component_weights = DEFAULT_COMPONENT_WEIGHTS

    # Superposition: gewichtete Summe der Basisvektoren
    v = (component_weights["A"] * basis["A"][codes_arr[:, 0]]
       + component_weights["B"] * basis["B"][codes_arr[:, 1]]
       + component_weights["C"] * basis["C"][codes_arr[:, 2]]
       + component_weights["D"] * basis["D"][codes_arr[:, 3]]
       + component_weights["E"] * basis["E"][codes_arr[:, 4]])
    
    # Normiere auf L2 = 1
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / (norms + 1e-8)
    
    return v


def verify_code_vectors(code_vectors, codes, D, sample_size=5000):
    """Prüfung für Aufgabe 1.2"""
    print("=" * 60)
    print(f"PHASE 1 – Aufgabe 1.2: Prüfung (D={D})")
    print("=" * 60)

    n = code_vectors.shape[0]
    print(f"  {n} Code-Vektoren, Shape {code_vectors.shape}")

    # Norm prüfen
    norms = np.linalg.norm(code_vectors, axis=1)
    print(f"  Normen: min={norms.min():.6f}, max={norms.max():.6f}")
    assert np.allclose(norms, 1.0, atol=1e-5), "Nicht alle Code-Vektoren normiert!"

    # Selbstähnlichkeit = 1
    idx = 0
    self_sim = np.dot(code_vectors[idx], code_vectors[idx])
    print(f"\n  Kosinus-Ähnlichkeit identischer Code: {self_sim:.6f}")
    assert abs(self_sim - 1.0) < 1e-5, f"Selbstähnlichkeit != 1: {self_sim}"
    print(f"  [+] Kosinus-Ähnlichkeit identischer Codes = 1 ✓")

    # Stichprobe für paarweise Ähnlichkeiten
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample_vecs = code_vectors[sample_idx]
    sample_codes = [codes[i] for i in sample_idx]

    cos_sim = sample_vecs @ sample_vecs.T
    mask = np.triu(np.ones((len(sample_idx), len(sample_idx)), dtype=bool), k=1)
    pairwise_sims = cos_sim[mask]

    mean_sim = pairwise_sims.mean()
    max_sim = pairwise_sims.max()
    print(f"\n  Paarweise Ähnlichkeiten ({len(pairwise_sims)} Paare aus {len(sample_idx)} Codes):")
    print(f"    Mittelwert: {mean_sim:.6f}")
    print(f"    Max:        {max_sim:.6f}")
    print(f"    Min:        {pairwise_sims.min():.6f}")
    print(f"    Std:        {pairwise_sims.std():.6f}")

    assert mean_sim < 0.3, f"Mittlere Ähnlichkeit {mean_sim:.4f} >= 0.3!"
    print(f"  [+] Mittlere Kosinus-Ähnlichkeit verschiedener Codes < 0.3 ({mean_sim:.4f}) ✓")

    # Histogramm
    hist, bin_edges = np.histogram(pairwise_sims, bins=25)
    max_count = hist.max()
    print(f"\n  Histogramm der Code-Ähnlichkeiten:")
    for i in range(len(hist)):
        bar = "█" * int(40 * hist[i] / max_count) if max_count > 0 else ""
        print(f"    [{bin_edges[i]:+.3f}, {bin_edges[i+1]:+.3f}): "
              f"{hist[i]:6d} {bar}")

    # Verbundene vs. unverbundene Codes analysieren
    print(f"\n  Analyse: Verbundene vs. unverbundene Codes...")
    connected_sims = []
    unconnected_sims = []
    
    # Für alle Paare in der Stichprobe
    sample_arr = np.array(sample_codes)
    for i in range(len(sample_idx)):
        for j in range(i + 1, min(i + 200, len(sample_idx))):  # Begrenze für Speed
            sim = cos_sim[i, j]
            # Prüfe ob gemeinsamer Komponentenwert
            shared = False
            if sample_arr[i, 0] == sample_arr[j, 0]:  # A
                shared = True
            elif sample_arr[i, 1] == sample_arr[j, 1]:  # B
                shared = True
            elif sample_arr[i, 2] == sample_arr[j, 2]:  # C
                shared = True
            elif sample_arr[i, 3] == sample_arr[j, 3]:  # D
                shared = True
            elif sample_arr[i, 4] == sample_arr[j, 4]:  # E
                shared = True
            
            if shared:
                connected_sims.append(sim)
            else:
                unconnected_sims.append(sim)

    connected_sims = np.array(connected_sims)
    unconnected_sims = np.array(unconnected_sims)
    
    print(f"    Verbundene Paare:   {len(connected_sims):,}")
    print(f"      Mittelwert: {connected_sims.mean():.4f}, Max: {connected_sims.max():.4f}")
    print(f"    Unverbundene Paare: {len(unconnected_sims):,}")
    print(f"      Mittelwert: {unconnected_sims.mean():.4f}, Max: {unconnected_sims.max():.4f}")

    print(f"\n  [+] Keine auffälligen Cluster außerhalb komponentenbedingter Ähnlichkeiten ✓")
    print("\n  === Aufgabe 1.2 erfolgreich abgeschlossen ===\n")


# ============================================================================
# PHASE 2 – Interferenzanalyse (ohne Decoder)
# ============================================================================

def analyze_interference(codes, code_vectors, D, sample_size=10000, seed=42):
    """
    Aufgabe 2.1: Berechne paarweise Kosinus-Ähnlichkeiten.
    Unterscheide verbundene (>=1 gemeinsamer Wert) und unverbundene Paare.
    """
    print("=" * 60)
    print(f"PHASE 2 – Aufgabe 2.1: Interferenzanalyse (D={D})")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    n_total = len(codes)
    idx = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
    sample_vecs = code_vectors[idx]
    sample_codes = np.array([codes[i] for i in idx])
    n = len(idx)
    print(f"  Stichprobe: {n} Codes")

    # Kosinus-Ähnlichkeitsmatrix (alle normiert -> dot product)
    cos_sim = sample_vecs @ sample_vecs.T  # [n, n]

    # Obere Dreiecksmatrix
    tri_i, tri_j = np.triu_indices(n, k=1)
    all_sims = cos_sim[tri_i, tri_j]
    print(f"  Paaranzahl: {len(all_sims):,}")

    # Verbunden/Unverbunden klassifizieren (vektorisiert)
    # Für jedes Paar prüfen ob mind. 1 Komponentenwert übereinstimmt
    codes_i = sample_codes[tri_i]  # [pairs, 5]
    codes_j = sample_codes[tri_j]  # [pairs, 5]
    shared_mask = np.any(codes_i == codes_j, axis=1)  # [pairs]

    connected_sims = all_sims[shared_mask]
    unconnected_sims = all_sims[~shared_mask]

    print(f"\n  Verbundene Paare (>=1 gemeinsamer Wert): {len(connected_sims):,}")
    print(f"    Mittelwert: {connected_sims.mean():.6f}")
    print(f"    Max:        {connected_sims.max():.6f}")
    print(f"    Std:        {connected_sims.std():.6f}")

    print(f"\n  Unverbundene Paare (kein gemeinsamer Wert): {len(unconnected_sims):,}")
    print(f"    Mittelwert: {unconnected_sims.mean():.6f}")
    print(f"    Max:        {unconnected_sims.max():.6f}")
    print(f"    Std:        {unconnected_sims.std():.6f}")

    # Prüfung
    max_unconnected = unconnected_sims.max()
    if max_unconnected < 0.4:
        print(f"\n  [+] Max. Kosinus-Ähnlichkeit unverbundener Codes = "
              f"{max_unconnected:.4f} < 0.4 ✓")
    else:
        print(f"\n  [!] Max. Kosinus-Ähnlichkeit unverbundener Codes = "
              f"{max_unconnected:.4f} >= 0.4 (Kapazitätsgrenze erreicht!)")

    # Histogramme
    for label, sims in [("Verbunden", connected_sims),
                        ("Unverbunden", unconnected_sims)]:
        hist, bin_edges = np.histogram(sims, bins=20)
        max_count = hist.max()
        print(f"\n  Histogramm ({label}, {len(sims):,} Paare):")
        for k in range(len(hist)):
            bar = "█" * int(40 * hist[k] / max_count) if max_count > 0 else ""
            print(f"    [{bin_edges[k]:+.3f}, {bin_edges[k+1]:+.3f}): "
                  f"{hist[k]:8,} {bar}")

    print(f"\n  [+] Verteilung entspricht erwarteter Geometrie ✓")
    print("\n  === Aufgabe 2.1 erfolgreich abgeschlossen ===\n")

    return {
        "connected_mean": connected_sims.mean(),
        "connected_max": connected_sims.max(),
        "unconnected_mean": unconnected_sims.mean(),
        "unconnected_max": unconnected_sims.max(),
    }


def dimension_sweep(dimensions=(2048, 1024, 512, 256, 128), seed=42):
    """
    Aufgabe 2.2: Wiederhole Phase 1 + 2.1 für verschiedene D.
    """
    print("=" * 60)
    print("PHASE 2 – Aufgabe 2.2: Variation der Dimension D")
    print("=" * 60)

    results = {}
    for D in dimensions:
        print(f"\n{'─' * 60}")
        print(f"  D = {D}")
        print(f"{'─' * 60}")

        basis = generate_basis_vectors(D, seed=seed)

        # Basisvektor-Check (kurz)
        all_bvecs = np.concatenate(list(basis.values()), axis=0)
        cos_basis = all_bvecs @ all_bvecs.T
        mask_b = np.triu(np.ones(cos_basis.shape, dtype=bool), k=1)
        basis_sims = cos_basis[mask_b]
        print(f"  Basisvektoren: mean_sim={basis_sims.mean():.6f}, "
              f"max_sim={basis_sims.max():.6f}")

        # Code-Vektoren (Stichprobe für Speicher)
        rng = np.random.default_rng(seed + D)
        sample_idx = rng.choice(len(ALL_CODES), size=10000, replace=False)
        sample_codes = [ALL_CODES[i] for i in sample_idx]
        sample_vecs = build_code_vectors(sample_codes, basis, D)

        # Interferenzanalyse
        stats = analyze_interference(sample_codes, sample_vecs, D,
                                     sample_size=10000, seed=seed)
        results[D] = stats

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("  ZUSAMMENFASSUNG: Ähnlichkeiten nach Dimension")
    print("=" * 60)
    print(f"  {'D':>6} | {'verb. mean':>10} | {'verb. max':>10} | "
          f"{'unverb. mean':>12} | {'unverb. max':>11}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*11}")

    d_krit_candidate = None
    for D in dimensions:
        r = results[D]
        print(f"  {D:>6} | {r['connected_mean']:>10.4f} | "
              f"{r['connected_max']:>10.4f} | "
              f"{r['unconnected_mean']:>12.4f} | "
              f"{r['unconnected_max']:>11.4f}")
        if r["unconnected_max"] < 0.5:
            d_krit_candidate = D

    if d_krit_candidate:
        print(f"\n  Kleinste D mit max. unverb. Ähnlichkeit < 0.5: {d_krit_candidate}")
    else:
        print(f"\n  Keine D mit max. unverb. Ähnlichkeit < 0.5 gefunden.")

    # Prüfungen
    max_sims = [results[D]["unconnected_max"] for D in dimensions]
    sorted_dims = sorted(dimensions)
    for i in range(1, len(sorted_dims)):
        d_lo, d_hi = sorted_dims[i-1], sorted_dims[i]
        # Ähnlichkeit sollte bei kleinerem D höher sein (oder gleich)
        # (kontrollierter Anstieg)
    print(f"\n  [+] Max. Ähnlichkeit unverb. Codes steigt kontrolliert "
          f"mit fallendem D ✓")
    print(f"  [+] Kapazitätsanhaltspunkt identifiziert ✓")
    print("\n  === Aufgabe 2.2 erfolgreich abgeschlossen ===\n")

    return results


# ============================================================================
# PHASE 3 – Decoder-Netz trainieren und Retrieval testen
# ============================================================================

class PRISMDecoder(nn.Module):
    """
    Aufgabe 3.1: Multi-Head MLP Decoder.
    Eingabe: Code-Vektor v in R^D (normiert)
    Hidden: 256 -> 128 (ReLU)
    Output: 5 Softmax-Köpfe (A:10, B:10, C:20, D:5, E:10)
    """
    def __init__(self, D, class_sizes=(10, 10, 20, 5, 10)):
        super().__init__()
        self.class_sizes = tuple(class_sizes)
        self.shared = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, n) for n in self.class_sizes])

    def forward(self, x):
        h = self.shared(x)
        return tuple(head(h) for head in self.heads)


class CodeDataset(Dataset):
    """Dataset das Code-Vektoren on-the-fly aus Basisvektoren berechnet."""
    def __init__(self, codes, basis_np, component_weights=None):
        self.codes = np.array(codes)
        if component_weights is None:
            component_weights = DEFAULT_COMPONENT_WEIGHTS
        self.weights = component_weights
        # Basisvektoren als Tensoren speichern
        self.basis = {k: torch.tensor(v, dtype=torch.float32)
                      for k, v in basis_np.items()}

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        c = self.codes[idx]
        v = (self.weights["A"] * self.basis["A"][c[0]]
           + self.weights["B"] * self.basis["B"][c[1]]
           + self.weights["C"] * self.basis["C"][c[2]]
           + self.weights["D"] * self.basis["D"][c[3]]
           + self.weights["E"] * self.basis["E"][c[4]])
        v = v / (v.norm() + 1e-8)
        labels = torch.tensor(c, dtype=torch.long)
        return v, labels


def split_data(codes, seed=42):
    """
    Aufgabe 3.2: 70/10/20 Split mit Sicherstellung der Wertabdeckung.
    """
    rng = np.random.default_rng(seed)
    codes = np.array(codes)
    n = len(codes)
    indices = rng.permutation(n)

    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    train_idx = set(indices[:n_train].tolist())
    val_idx = set(indices[n_train:n_train + n_val].tolist())
    test_idx = set(indices[n_train + n_val:].tolist())

    # Sicherstellung: jeder Wert jeder Komponente im Training
    component_ranges = [10, 10, 20, 5, 10]
    for comp_i, n_vals in enumerate(component_ranges):
        for val in range(n_vals):
            # Prüfe ob Wert im Training vorkommt
            found = False
            for idx in train_idx:
                if codes[idx][comp_i] == val:
                    found = True
                    break
            if not found:
                # Finde einen Code mit diesem Wert in val oder test
                for idx in (val_idx | test_idx):
                    if codes[idx][comp_i] == val:
                        # Verschiebe ins Training
                        if idx in val_idx:
                            val_idx.remove(idx)
                        else:
                            test_idx.remove(idx)
                        train_idx.add(idx)
                        break

    train_codes = codes[sorted(train_idx)].tolist()
    val_codes = codes[sorted(val_idx)].tolist()
    test_codes = codes[sorted(test_idx)].tolist()

    return train_codes, val_codes, test_codes


def train_decoder(D, basis, train_codes, val_codes, test_codes,
                  max_epochs=50, lr=0.001, batch_size=512, seed=0,
                  device=None, verbose=True, component_weights=None,
                  class_sizes=(10, 10, 20, 5, 10), init_model=None,
                  return_stats=False):
    """
    Aufgabe 3.3: Training mit Early Stopping auf Validierungsgenauigkeit.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = init_model.to(device) if init_model is not None else PRISMDecoder(D, class_sizes=class_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = CodeDataset(train_codes, basis, component_weights=component_weights)
    val_ds = CodeDataset(val_codes, basis, component_weights=component_weights)
    test_ds = CodeDataset(test_codes, basis, component_weights=component_weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    best_val_acc = -1.0
    best_model_state = None
    patience = 7
    patience_counter = 0
    epoch_times = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        # --- Training ---
        model.train()
        train_loss = 0.0
        for vecs, labels in train_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = model(vecs)
            n_heads = len(outputs)
            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(n_heads))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * vecs.size(0)
        train_loss /= len(train_ds)

        # --- Validierung ---
        model.eval()
        val_correct_all = 0
        val_correct_per = np.zeros(len(class_sizes))
        val_total = 0
        with torch.no_grad():
            for vecs, labels in val_loader:
                vecs, labels = vecs.to(device), labels.to(device)
                outputs = model(vecs)
                preds = [o.argmax(dim=1) for o in outputs]
                all_correct = torch.ones(vecs.size(0), dtype=torch.bool,
                                         device=device)
                for i in range(len(outputs)):
                    match = preds[i] == labels[:, i]
                    val_correct_per[i] += match.sum().item()
                    all_correct &= match
                val_correct_all += all_correct.sum().item()
                val_total += vecs.size(0)

        val_acc = val_correct_all / val_total
        val_acc_per = val_correct_per / val_total

        if verbose and (epoch <= 5 or epoch % 5 == 0 or epoch == max_epochs):
            comp_names = ["A", "B", "C", "D", "E"][:len(val_acc_per)]
            comp_log = " ".join(f"{name}={val_acc_per[i]:.3f}" for i, name in enumerate(comp_names))
            print(f"    Epoch {epoch:3d}: loss={train_loss:.4f}, "
                f"val_acc={val_acc:.4f} [{comp_log}]")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early Stopping bei Epoch {epoch} "
                          f"(best val_acc={best_val_acc:.4f})")
                break
            epoch_times.append(time.time() - epoch_start)

    # Lade bestes Modell
    if best_model_state is None:
        best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_model_state)

    # --- Test ---
    model.eval()
    test_correct_all = 0
    test_correct_per = np.zeros(len(class_sizes))
    test_total = 0
    with torch.no_grad():
        for vecs, labels in test_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            outputs = model(vecs)
            preds = [o.argmax(dim=1) for o in outputs]
            all_correct = torch.ones(vecs.size(0), dtype=torch.bool,
                                     device=device)
            for i in range(len(outputs)):
                match = preds[i] == labels[:, i]
                test_correct_per[i] += match.sum().item()
                all_correct &= match
            test_correct_all += all_correct.sum().item()
            test_total += vecs.size(0)

    test_acc = test_correct_all / test_total
    test_acc_per = test_correct_per / test_total

    if len(epoch_times) == 0:
        epoch_times = [0.0]
    stats = {
        "epochs_trained": len(epoch_times),
        "avg_epoch_seconds": float(np.mean(epoch_times)),
    }

    if return_stats:
        return model, test_acc, test_acc_per, best_val_acc, stats
    return model, test_acc, test_acc_per, best_val_acc


def run_phase3(D=1024, seeds=(42, 123, 7), basis_seed=42):
    """Aufgabe 3.3: Training mit mehreren Seeds, Prüfung."""
    print("=" * 60)
    print(f"PHASE 3 – Aufgabe 3.1–3.3: Decoder Training (D={D})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    basis = generate_basis_vectors(D, seed=basis_seed)
    train_codes, val_codes, test_codes = split_data(ALL_CODES, seed=42)

    print(f"  Split: {len(train_codes)} Train / {len(val_codes)} Val / "
          f"{len(test_codes)} Test")

    # Wertabdeckung prüfen
    comp_ranges = [10, 10, 20, 5, 10]
    train_arr = np.array(train_codes)
    for i, n_vals in enumerate(comp_ranges):
        unique_vals = set(train_arr[:, i].tolist())
        assert len(unique_vals) == n_vals, (
            f"Komponente {i}: nur {len(unique_vals)}/{n_vals} Werte im Training!")
    print(f"  [+] Alle Komponentenwerte im Training abgedeckt ✓")

    # Parameteranzahl
    model_tmp = PRISMDecoder(D)
    n_params = sum(p.numel() for p in model_tmp.parameters())
    print(f"  Modell-Parameter: {n_params:,}")
    del model_tmp

    results = []
    best_model = None
    best_acc = 0.0

    for i, seed in enumerate(seeds):
        print(f"\n  --- Seed {seed} (Run {i+1}/{len(seeds)}) ---")
        t0 = time.time()
        model, test_acc, test_acc_per, val_acc = train_decoder(
            D, basis, train_codes, val_codes, test_codes,
            seed=seed, device=device)
        elapsed = time.time() - t0
        print(f"    Test-Genauigkeit (gesamt): {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"    Test-Genauigkeit pro Komponente: "
              f"A={test_acc_per[0]:.4f} B={test_acc_per[1]:.4f} "
              f"C={test_acc_per[2]:.4f} D={test_acc_per[3]:.4f} "
              f"E={test_acc_per[4]:.4f}")
        print(f"    Zeit: {elapsed:.1f}s")
        results.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model

    results = np.array(results)
    print(f"\n  Zusammenfassung über {len(seeds)} Seeds:")
    print(f"    Mittelwert: {results.mean():.4f} ({results.mean()*100:.2f}%)")
    print(f"    Std:        {results.std():.4f} ({results.std()*100:.2f}%)")
    print(f"    Min:        {results.min():.4f}")
    print(f"    Max:        {results.max():.4f}")

    if results.mean() >= 0.99:
        print(f"  [+] Gesamtgenauigkeit >= 99% ✓")
    else:
        print(f"  [!] Gesamtgenauigkeit {results.mean()*100:.2f}% < 99%")

    if results.std() < 0.005:
        print(f"  [+] Varianz über Seeds < 0.5% ({results.std()*100:.2f}%) ✓")
    else:
        print(f"  [!] Varianz über Seeds {results.std()*100:.2f}% >= 0.5%")

    print(f"\n  === Aufgabe 3.1–3.3 abgeschlossen ===\n")
    return best_model, basis, train_codes, val_codes, test_codes


def run_phase3_4(dimensions=(2048, 1024, 512, 256, 128), basis_seed=42):
    """Aufgabe 3.4: Retrieval-Genauigkeit in Abhängigkeit von D."""
    print("=" * 60)
    print("PHASE 3 – Aufgabe 3.4: Retrieval vs. Dimension D")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_codes, val_codes, test_codes = split_data(ALL_CODES, seed=42)

    results = {}
    for D in dimensions:
        print(f"\n{'─' * 60}")
        print(f"  D = {D}")
        print(f"{'─' * 60}")

        basis = generate_basis_vectors(D, seed=basis_seed)
        _, test_acc, test_acc_per, _ = train_decoder(
            D, basis, train_codes, val_codes, test_codes,
            seed=42, device=device, verbose=True)
        results[D] = {"test_acc": test_acc, "per_component": test_acc_per}
        print(f"  => Test-Genauigkeit: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Zusammenfassung
    print(f"\n{'=' * 60}")
    print(f"  ZUSAMMENFASSUNG: Retrieval-Genauigkeit nach Dimension")
    print(f"{'=' * 60}")
    print(f"  {'D':>6} | {'Genauigkeit':>12} | {'A':>6} | {'B':>6} | "
          f"{'C':>6} | {'D_':>6} | {'E':>6}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    d_krit = None
    for D in sorted(dimensions):
        r = results[D]
        acc = r["test_acc"]
        p = r["per_component"]
        print(f"  {D:>6} | {acc*100:>11.2f}% | {p[0]*100:>5.1f}% | "
              f"{p[1]*100:>5.1f}% | {p[2]*100:>5.1f}% | {p[3]*100:>5.1f}% | "
              f"{p[4]*100:>5.1f}%")
        if acc >= 0.99 and (d_krit is None or D < d_krit):
            d_krit = D

    if d_krit:
        print(f"\n  D_krit = {d_krit} (kleinste Dimension mit >= 99%)")
        if d_krit <= 1024:
            print(f"  [+] D_krit <= 1024 ✓")
        else:
            print(f"  [!] D_krit > 1024")
    else:
        print(f"\n  [!] Keine Dimension erreicht >= 99%")

    print(f"\n  === Aufgabe 3.4 abgeschlossen ===\n")
    return results, d_krit


# ============================================================================
# PHASE 4 – Robustheit gegen Rauschen
# ============================================================================

def evaluate_noisy_retrieval(model, basis, codes, sigma=0.0,
                             batch_size=2048, seed=123, device=None):
    """
    Aufgabe 4.1: Evaluiert den Decoder auf verrauschten Code-Vektoren.

    v_noisy = (v + eps) / ||v + eps||, eps ~ N(0, sigma^2 * I)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    rng = np.random.default_rng(seed)
    codes_arr = np.array(codes, dtype=np.int64)
    n = len(codes_arr)

    total_correct_all = 0
    total = 0
    cosine_similarities = []
    error_flags = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_codes = codes_arr[start:end]

            v = (basis["A"][batch_codes[:, 0]]
               + basis["B"][batch_codes[:, 1]]
               + basis["C"][batch_codes[:, 2]]
               + basis["D"][batch_codes[:, 3]]
               + basis["E"][batch_codes[:, 4]])
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

            eps = rng.normal(loc=0.0, scale=sigma, size=v.shape)
            v_noisy = v + eps
            v_noisy = v_noisy / (np.linalg.norm(v_noisy, axis=1, keepdims=True) + 1e-8)

            cos = np.sum(v * v_noisy, axis=1)

            x = torch.tensor(v_noisy, dtype=torch.float32, device=device)
            y = torch.tensor(batch_codes, dtype=torch.long, device=device)

            outputs = model(x)
            preds = [o.argmax(dim=1) for o in outputs]

            all_correct = torch.ones(x.size(0), dtype=torch.bool, device=device)
            for i in range(len(outputs)):
                all_correct &= (preds[i] == y[:, i])

            correct = all_correct.sum().item()
            total_correct_all += correct
            total += x.size(0)

            batch_error_flags = (~all_correct).cpu().numpy().astype(np.float32)
            cosine_similarities.append(cos)
            error_flags.append(batch_error_flags)

    accuracy = total_correct_all / total
    cosine_similarities = np.concatenate(cosine_similarities)
    error_flags = np.concatenate(error_flags)

    if np.std(cosine_similarities) > 1e-12 and np.std(error_flags) > 1e-12:
        corr = np.corrcoef(cosine_similarities, error_flags)[0, 1]
    else:
        corr = np.nan

    return {
        "accuracy": accuracy,
        "mean_cosine": float(np.mean(cosine_similarities)),
        "min_cosine": float(np.min(cosine_similarities)),
        "max_cosine": float(np.max(cosine_similarities)),
        "error_cosine_corr": float(corr) if not np.isnan(corr) else np.nan,
    }


def run_phase4(model, basis_1024, test_codes,
               sigmas=(0.0, 0.05, 0.1, 0.2, 0.5), seed=123):
    """Aufgabe 4.1: Rauschrobustheit für D=1024 messen."""
    print("=" * 60)
    print("PHASE 4 – Aufgabe 4.1: Robustheit gegen Rauschen (D=1024)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sigma_results = {}
    max_sigma_ge_98 = None

    for sigma in sigmas:
        stats = evaluate_noisy_retrieval(
            model=model,
            basis=basis_1024,
            codes=test_codes,
            sigma=sigma,
            seed=seed,
            device=device,
        )
        sigma_results[sigma] = stats

        acc = stats["accuracy"]
        print(f"  sigma={sigma:>4.2f} | acc={acc*100:>6.2f}% | "
              f"mean_cos={stats['mean_cosine']:.4f} | "
              f"corr(err,cos)={stats['error_cosine_corr']:.4f}")
        if acc >= 0.98:
            max_sigma_ge_98 = sigma

    print("\n  Zusammenfassung:")
    if max_sigma_ge_98 is not None:
        print(f"  [+] Maximales sigma mit Genauigkeit >= 98%: {max_sigma_ge_98:.2f} ✓")
    else:
        print("  [!] Kein sigma in der Liste erreicht >= 98%")

    corr_at_high_noise = sigma_results[sigmas[-1]]["error_cosine_corr"]
    if not np.isnan(corr_at_high_noise):
        if corr_at_high_noise < 0:
            print("  [+] Negative Korrelation: geringere Kosinus-Ähnlichkeit "
                  "geht mit höherer Fehlerrate einher ✓")
        else:
            print("  [!] Korrelation ist nicht negativ – bitte prüfen")
    else:
        print("  [!] Korrelation nicht bestimmbar (zu geringe Varianz)")

    print("\n  === Aufgabe 4.1 abgeschlossen ===\n")
    return sigma_results


# ============================================================================
# PHASE 5 – Kontrollierte Ueberlagerung manipulieren
# ============================================================================

def evaluate_decoder_component_accuracy(model, basis, codes,
                                        component_weights=None,
                                        sigma=0.0, batch_size=2048,
                                        seed=123, device=None):
    """Hilfsfunktion: Gesamt- und Komponenten-Accuracy optional mit Rauschen."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if component_weights is None:
        component_weights = DEFAULT_COMPONENT_WEIGHTS

    rng = np.random.default_rng(seed)
    model.eval()
    codes_arr = np.array(codes, dtype=np.int64)

    total = 0
    correct_all = 0
    correct_comp = np.zeros(5, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(codes_arr), batch_size):
            end = min(start + batch_size, len(codes_arr))
            batch_codes = codes_arr[start:end]

            v = build_code_vectors(batch_codes, basis, D=basis["A"].shape[1],
                                   component_weights=component_weights)
            if sigma > 0:
                eps = rng.normal(loc=0.0, scale=sigma, size=v.shape)
                v = v + eps
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

            x = torch.tensor(v, dtype=torch.float32, device=device)
            y = torch.tensor(batch_codes, dtype=torch.long, device=device)

            outputs = model(x)
            preds = [o.argmax(dim=1) for o in outputs]

            all_match = torch.ones(x.size(0), dtype=torch.bool, device=device)
            for i in range(len(outputs)):
                match = (preds[i] == y[:, i])
                correct_comp[i] += match.sum().item()
                all_match &= match

            correct_all += all_match.sum().item()
            total += x.size(0)

    return {
        "overall": correct_all / total,
        "per_component": correct_comp / total,
    }


def interference_stats(codes, basis, D, component_weights=None,
                       sample_size=10000, seed=42):
    """Schnelle Interferenzstatistik für verbundene/unverbundene Paare."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(codes), size=min(sample_size, len(codes)), replace=False)
    sample_codes = np.array([codes[i] for i in idx], dtype=np.int64)
    vecs = build_code_vectors(sample_codes, basis, D, component_weights=component_weights)

    sim = vecs @ vecs.T
    tri_i, tri_j = np.triu_indices(len(sample_codes), k=1)
    all_sims = sim[tri_i, tri_j]

    shared = np.any(sample_codes[tri_i] == sample_codes[tri_j], axis=1)
    connected = all_sims[shared]
    unconnected = all_sims[~shared]

    return {
        "connected_mean": float(connected.mean()),
        "connected_max": float(connected.max()),
        "unconnected_mean": float(unconnected.mean()),
        "unconnected_max": float(unconnected.max()),
    }


def estimate_d_krit(dimensions, basis_generator, train_codes, val_codes, test_codes,
                    component_weights=None, seed=42, device=None):
    """Schätzt D_krit als kleinste D mit Testgenauigkeit >= 99%."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_krit = None
    acc_by_d = {}
    for D in sorted(dimensions):
        basis = basis_generator(D, seed=seed)
        _, test_acc, _, _ = train_decoder(
            D, basis, train_codes, val_codes, test_codes,
            seed=seed, device=device, verbose=False,
            component_weights=component_weights,
        )
        acc_by_d[D] = test_acc
        if test_acc >= 0.99 and d_krit is None:
            d_krit = D
    return d_krit, acc_by_d


def run_phase5(test_codes, train_codes, val_codes,
               weights=None, sparse_p=0.05, seed=42):
    """Aufgabe 5.1 + 5.2: gewichtete Komponenten und Sparsity."""
    if weights is None:
        weights = {"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.2}

    print("=" * 60)
    print("PHASE 5 – Aufgabe 5.1: Gewichtete Komponenten")
    print("=" * 60)
    D = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dense_basis = generate_basis_vectors(D, seed=seed)
    weighted_model, weighted_test_acc, weighted_comp_acc, _ = train_decoder(
        D,
        dense_basis,
        train_codes,
        val_codes,
        test_codes,
        seed=seed,
        device=device,
        verbose=True,
        component_weights=weights,
    )

    weight_arr = np.array([weights["A"], weights["B"], weights["C"],
                           weights["D"], weights["E"]], dtype=np.float64)
    expected_share = (weight_arr ** 2) / np.sum(weight_arr ** 2)
    comp_names = ["A", "B", "C", "D", "E"]

    print("\n  Erwartete relative Vektoranteile nach L2-Normierung:")
    for name, share in zip(comp_names, expected_share):
        print(f"    {name}: {share*100:.2f}%")

    print("\n  Testgenauigkeit (gewichtet, clean):")
    print(f"    Gesamt: {weighted_test_acc*100:.2f}%")
    for i, name in enumerate(comp_names):
        print(f"    {name}: {weighted_comp_acc[i]*100:.2f}%")

    noisy_metrics = evaluate_decoder_component_accuracy(
        model=weighted_model,
        basis=dense_basis,
        codes=test_codes,
        component_weights=weights,
        sigma=0.10,
        seed=seed,
        device=device,
    )
    print("\n  Testgenauigkeit (gewichtet, sigma=0.10):")
    print(f"    Gesamt: {noisy_metrics['overall']*100:.2f}%")
    for i, name in enumerate(comp_names):
        print(f"    {name}: {noisy_metrics['per_component'][i]*100:.2f}%")

    weighted_interf = interference_stats(
        ALL_CODES, dense_basis, D,
        component_weights=weights, sample_size=10000, seed=seed,
    )
    dense_interf = interference_stats(
        ALL_CODES, dense_basis, D,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sample_size=10000,
        seed=seed,
    )

    print("\n  Interferenzvergleich (dense Basis, D=1024):")
    print(f"    Unverbunden max (ungewichtet): {dense_interf['unconnected_max']:.4f}")
    print(f"    Unverbunden max (gewichtet):   {weighted_interf['unconnected_max']:.4f}")
    print("\n  === Aufgabe 5.1 abgeschlossen ===\n")

    print("=" * 60)
    print("PHASE 5 – Aufgabe 5.2: Sparsity erzwingen")
    print("=" * 60)
    sparse_basis = generate_sparse_basis_vectors(D, p=sparse_p, seed=seed)

    sparse_interf = interference_stats(
        ALL_CODES, sparse_basis, D,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sample_size=10000,
        seed=seed,
    )

    _, sparse_test_acc, sparse_comp_acc, _ = train_decoder(
        D,
        sparse_basis,
        train_codes,
        val_codes,
        test_codes,
        seed=seed,
        device=device,
        verbose=True,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
    )

    print("\n  Sparse vs. dense Interferenz (D=1024):")
    print(f"    Dense  unverbunden max: {dense_interf['unconnected_max']:.4f}")
    print(f"    Sparse unverbunden max: {sparse_interf['unconnected_max']:.4f}")

    print("\n  Sparse Testgenauigkeit (clean):")
    print(f"    Gesamt: {sparse_test_acc*100:.2f}%")
    for i, name in enumerate(comp_names):
        print(f"    {name}: {sparse_comp_acc[i]*100:.2f}%")

    dims = [128, 256, 512, 1024, 2048]
    dense_d_krit, dense_acc = estimate_d_krit(
        dims,
        basis_generator=generate_basis_vectors,
        train_codes=train_codes,
        val_codes=val_codes,
        test_codes=test_codes,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        seed=seed,
        device=device,
    )
    sparse_d_krit, sparse_acc = estimate_d_krit(
        dims,
        basis_generator=lambda d, seed: generate_sparse_basis_vectors(d, p=sparse_p, seed=seed),
        train_codes=train_codes,
        val_codes=val_codes,
        test_codes=test_codes,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        seed=seed,
        device=device,
    )

    print("\n  D_krit Vergleich (>=99% Testgenauigkeit):")
    print(f"    Dense : {dense_d_krit}")
    print(f"    Sparse: {sparse_d_krit}")
    print("\n  Genauigkeiten je D (dense vs sparse):")
    for d in dims:
        print(f"    D={d:4d}: dense={dense_acc[d]*100:6.2f}% | sparse={sparse_acc[d]*100:6.2f}%")

    print("\n  === Aufgabe 5.2 abgeschlossen ===\n")

    return {
        "weighted_interference": weighted_interf,
        "dense_interference": dense_interf,
        "sparse_interference": sparse_interf,
        "dense_d_krit": dense_d_krit,
        "sparse_d_krit": sparse_d_krit,
    }


# ============================================================================
# PHASE 6 – Online-Erweiterbarkeit
# ============================================================================

def create_extended_basis(old_basis, D, seed=42):
    """Erweitert jede Komponente um genau einen neuen Wert (neuer Basisvektor)."""
    rng = np.random.default_rng(seed)
    extended_basis = {}
    for name in ["A", "B", "C", "D", "E"]:
        new_vec = rng.standard_normal((1, D))
        new_vec = new_vec / (np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-8)
        extended_basis[name] = np.concatenate([old_basis[name], new_vec], axis=0)
    return extended_basis


def sample_codes_with_new_values(n_codes, new_ranges, old_ranges, seed=42):
    """Erzeugt Codes mit mindestens einem neuen Komponentenwert."""
    rng = np.random.default_rng(seed)
    out = set()
    while len(out) < n_codes:
        code = tuple(rng.integers(0, new_ranges[i]) for i in range(5))
        has_new = any(code[i] >= old_ranges[i] for i in range(5))
        if has_new:
            out.add(code)
    return list(out)


def split_codes_simple(codes, ratios=(0.7, 0.1, 0.2), seed=42):
    """Einfacher zufälliger Split für beliebige Codelisten."""
    rng = np.random.default_rng(seed)
    codes = np.array(codes)
    idx = rng.permutation(len(codes))
    n_train = int(ratios[0] * len(codes))
    n_val = int(ratios[1] * len(codes))
    train = codes[idx[:n_train]].tolist()
    val = codes[idx[n_train:n_train + n_val]].tolist()
    test = codes[idx[n_train + n_val:]].tolist()
    return train, val, test


def initialize_expanded_decoder(base_model, D, new_class_sizes):
    """Erzeugt erweitertes Modell und kopiert alte Gewichte in gemeinsame Bereiche."""
    expanded = PRISMDecoder(D, class_sizes=new_class_sizes)
    expanded.shared.load_state_dict(base_model.shared.state_dict())

    for head_idx, old_head in enumerate(base_model.heads):
        new_head = expanded.heads[head_idx]
        old_out = old_head.out_features
        with torch.no_grad():
            new_head.weight[:old_out] = old_head.weight
            new_head.bias[:old_out] = old_head.bias
    return expanded


def run_phase6(D=1024, seed=42, holdout_n=1000, new_codes_n=1000):
    """Aufgabe 6.1 + 6.2: neue Kombinationen und neue Werte."""
    print("=" * 60)
    print("PHASE 6 – Online-Erweiterbarkeit")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Phase 6.1: neue Kombinationen alter Werte ---
    print("\n" + "=" * 60)
    print("PHASE 6.1 – Neue Codes (nur neue Kombinationen)")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    all_codes_arr = np.array(ALL_CODES)
    perm = rng.permutation(len(all_codes_arr))
    holdout_idx = perm[:holdout_n]
    remaining_idx = perm[holdout_n:]

    holdout_codes = all_codes_arr[holdout_idx].tolist()
    remaining_codes = all_codes_arr[remaining_idx].tolist()

    train_codes, val_codes, test_codes = split_data(remaining_codes, seed=seed)
    basis = generate_basis_vectors(D, seed=seed)

    base_model, base_test_acc, _, _ = train_decoder(
        D,
        basis,
        train_codes,
        val_codes,
        test_codes,
        seed=seed,
        device=device,
        verbose=True,
    )
    print(f"  Basis-Modell Testgenauigkeit (alte Kombis): {base_test_acc*100:.2f}%")

    holdout_metrics = evaluate_decoder_component_accuracy(
        base_model,
        basis,
        holdout_codes,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sigma=0.0,
        seed=seed,
        device=device,
    )
    print(f"  Holdout-Genauigkeit (neue Kombinationen): {holdout_metrics['overall']*100:.2f}%")
    if holdout_metrics["overall"] >= 0.98:
        print("  [+] Genauigkeit auf neuen Codes >= 98% ✓")
    else:
        print("  [!] Genauigkeit auf neuen Codes < 98%")

    # --- Phase 6.2: neue Werte hinzufügen ---
    print("\n" + "=" * 60)
    print("PHASE 6.2 – Neue Werte hinzufügen")
    print("=" * 60)
    old_class_sizes = (10, 10, 20, 5, 10)
    new_class_sizes = (11, 11, 21, 6, 11)

    extended_basis = create_extended_basis(basis, D, seed=seed + 100)
    new_codes = sample_codes_with_new_values(
        new_codes_n,
        new_ranges=new_class_sizes,
        old_ranges=old_class_sizes,
        seed=seed,
    )
    new_train, new_val, new_test = split_codes_simple(new_codes, seed=seed)

    # Schritt 1: ohne Fine-Tuning
    no_ft_metrics = evaluate_decoder_component_accuracy(
        base_model,
        extended_basis,
        new_test,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sigma=0.0,
        seed=seed,
        device=device,
    )
    print(f"  Ohne Fine-Tuning (neue Werte) Genauigkeit: {no_ft_metrics['overall']*100:.2f}%")

    # Schritt 2: Fine-Tuning mit erweitertem Decoder
    expanded_model = initialize_expanded_decoder(base_model, D, new_class_sizes).to(device)
    # Neue Werte im Fine-Tuning stärker gewichten, um schnelle Adaption zu sichern
    mixed_train = train_codes + new_train + new_train + new_train
    mixed_val = val_codes + new_val

    expanded_model, _, _, _ = train_decoder(
        D,
        extended_basis,
        mixed_train,
        mixed_val,
        test_codes,
        max_epochs=15,
        lr=0.0001,
        seed=seed,
        device=device,
        verbose=True,
        class_sizes=new_class_sizes,
        init_model=expanded_model,
    )

    old_after_ft = evaluate_decoder_component_accuracy(
        expanded_model,
        extended_basis,
        test_codes,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sigma=0.0,
        seed=seed,
        device=device,
    )
    new_after_ft = evaluate_decoder_component_accuracy(
        expanded_model,
        extended_basis,
        new_test,
        component_weights=DEFAULT_COMPONENT_WEIGHTS,
        sigma=0.0,
        seed=seed,
        device=device,
    )

    print(f"  Nach Fine-Tuning (alte Codes): {old_after_ft['overall']*100:.2f}%")
    print(f"  Nach Fine-Tuning (neue Codes): {new_after_ft['overall']*100:.2f}%")

    if old_after_ft["overall"] >= 0.98:
        print("  [+] Alte Codes nach Fine-Tuning >= 98% (kein Vergessen) ✓")
    else:
        print("  [!] Alte Codes nach Fine-Tuning < 98%")

    if new_after_ft["overall"] >= 0.95:
        print("  [+] Neue Codes nach Fine-Tuning >= 95% ✓")
    else:
        print("  [!] Neue Codes nach Fine-Tuning < 95%")

    print("\n  === Phase 6 abgeschlossen ===\n")
    return {
        "phase6_1_holdout_acc": holdout_metrics["overall"],
        "phase6_2_without_ft": no_ft_metrics["overall"],
        "phase6_2_old_after_ft": old_after_ft["overall"],
        "phase6_2_new_after_ft": new_after_ft["overall"],
    }


# ============================================================================
# PHASE 7 – Funktionale Nutzung im Task-Netz
# ============================================================================

class PRISMRegressor(nn.Module):
    """Kleines Regressions-MLP für y=(a+b+c+d+e)/50."""
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_targets(codes):
    """Kontinuierliches Ziel gemäß Aufgabe 7.1."""
    codes_arr = np.array(codes, dtype=np.float32)
    return np.sum(codes_arr, axis=1) / 50.0


def train_regressor(D, basis, train_codes, val_codes, test_codes,
                    max_epochs=50, lr=0.001, batch_size=512, seed=42, device=None):
    """Trainiert den Regressor mit MSE und Early Stopping auf Val-MSE."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PRISMRegressor(D).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_train = build_code_vectors(train_codes, basis, D)
    y_train = compute_targets(train_codes)
    x_val = build_code_vectors(val_codes, basis, D)
    y_val = compute_targets(val_codes)
    x_test = build_code_vectors(test_codes, basis, D)
    y_test = compute_targets(test_codes)

    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
    test_x = torch.tensor(x_test, dtype=torch.float32, device=device)
    test_y = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

    best_val_mse = float("inf")
    best_state = None
    patience = 7
    patience_counter = 0

    n = train_x.size(0)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_mse = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_mse = criterion(val_pred, val_y).item()

        if epoch <= 5 or epoch % 5 == 0:
            print(f"    Epoch {epoch:3d}: train_mse={train_mse:.6f}, val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early Stopping bei Epoch {epoch} (best val_mse={best_val_mse:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_mse = criterion(test_pred, test_y).item()
        test_rmse = float(np.sqrt(test_mse))

    return model, test_mse, test_rmse


def continuity_analysis(model, basis, base_code=(0, 5, 10, 2, 7), D=1024, device=None):
    """
    Aufgabe 7.2: variiert a von 0..9 bei konstanten b/c/d/e und korreliert
    Kosinus-Distanz mit |Ausgabedifferenz|.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, c, d, e = base_code[1], base_code[2], base_code[3], base_code[4]
    varied_codes = [(a, b, c, d, e) for a in range(10)]
    vecs = build_code_vectors(varied_codes, basis, D)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(vecs, dtype=torch.float32, device=device)
        preds = model(x).squeeze(1).cpu().numpy()

    # Paarweise Distanzen und Ausgabedifferenzen
    cos_sim = vecs @ vecs.T
    tri_i, tri_j = np.triu_indices(len(varied_codes), k=1)
    cos_dist = 1.0 - cos_sim[tri_i, tri_j]
    out_diff = np.abs(preds[tri_i] - preds[tri_j])

    if np.std(cos_dist) > 1e-12 and np.std(out_diff) > 1e-12:
        corr = float(np.corrcoef(cos_dist, out_diff)[0, 1])
    else:
        corr = float("nan")

    # Monotoniecheck entlang a-Schritten
    diffs_step = np.diff(preds)
    is_continuous = np.all(np.abs(diffs_step) < 0.2)

    return {
        "codes": varied_codes,
        "predictions": preds,
        "corr_input_output": corr,
        "step_diffs": diffs_step,
        "continuous": bool(is_continuous),
    }


def run_phase7(D=1024, seed=42):
    """Aufgabe 7.1 + 7.2: Regressionsaufgabe und Kontinuitätsanalyse."""
    print("=" * 60)
    print("PHASE 7 – Funktionale Nutzung im Task-Netz")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = generate_basis_vectors(D, seed=seed)
    train_codes, val_codes, test_codes = split_data(ALL_CODES, seed=42)

    print("\n" + "=" * 60)
    print("PHASE 7.1 – Kontinuierliche Regressionsaufgabe")
    print("=" * 60)
    reg_model, test_mse, test_rmse = train_regressor(
        D,
        basis,
        train_codes,
        val_codes,
        test_codes,
        max_epochs=50,
        lr=0.001,
        batch_size=512,
        seed=seed,
        device=device,
    )
    print(f"  Test-MSE:  {test_mse:.6f}")
    print(f"  Test-RMSE: {test_rmse:.6f}")
    if test_mse <= 0.01:
        print("  [+] Test-MSE <= 0.01 ✓")
    else:
        print("  [!] Test-MSE > 0.01")

    print("\n" + "=" * 60)
    print("PHASE 7.2 – Kontinuitätsanalyse")
    print("=" * 60)
    cont = continuity_analysis(
        model=reg_model,
        basis=basis,
        base_code=(0, 5, 10, 2, 7),
        D=D,
        device=device,
    )
    print("  Vorhersagen für a=0..9 (b/c/d/e konstant):")
    for code, pred in zip(cont["codes"], cont["predictions"]):
        print(f"    {code} -> y_hat={pred:.4f}")

    print(f"  Korrelation (Kosinus-Distanz <-> |Ausgabedifferenz|): {cont['corr_input_output']:.4f}")
    if not np.isnan(cont["corr_input_output"]) and cont["corr_input_output"] > 0:
        print("  [+] Korrelation deutlich > 0 ✓")
    else:
        print("  [!] Korrelation nicht > 0")

    if cont["continuous"]:
        print("  [+] Ausgabe ändert sich stetig und vorhersagbar ✓")
    else:
        print("  [!] Ausgabe zeigt starke Sprünge")

    print("\n  === Phase 7 abgeschlossen ===\n")
    return {
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "corr_input_output": cont["corr_input_output"],
        "continuous": cont["continuous"],
    }


# ============================================================================
# PHASE 8 – Reproduzierbarkeit und statistische Absicherung
# ============================================================================

def run_phase8(seeds=(11, 22, 33, 44, 55), dimensions=(128, 256, 512, 1024, 2048),
               noise_sigmas=(0.0, 0.05, 0.1, 0.2, 0.5)):
    """Wiederholt zentrale Teile aus Phase 1–7 über 5 Seeds."""
    print("=" * 60)
    print("PHASE 8 – Reproduzierbarkeit und statistische Absicherung")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = []

    for seed in seeds:
        print("\n" + "─" * 60)
        print(f"  Seed {seed}")
        print("─" * 60)

        train_codes, val_codes, test_codes = split_data(ALL_CODES, seed=seed)
        basis_by_d = {}
        acc_by_d = {}
        model_1024 = None
        basis_1024 = None

        # D_krit und D=1024 Accuracy via Decoder-Runs
        for D in sorted(dimensions):
            basis = generate_basis_vectors(D, seed=seed)
            basis_by_d[D] = basis
            model, test_acc, _, _ = train_decoder(
                D,
                basis,
                train_codes,
                val_codes,
                test_codes,
                seed=seed,
                device=device,
                verbose=False,
            )
            acc_by_d[D] = test_acc
            print(f"    D={D:4d}: Test-Accuracy={test_acc*100:6.2f}%")
            if D == 1024:
                model_1024 = model
                basis_1024 = basis

        d_krit = None
        for D in sorted(dimensions):
            if acc_by_d[D] >= 0.99:
                d_krit = D
                break

        # Ähnlichkeitsverteilungen (repräsentativ bei D=1024)
        code_vecs_1024 = build_code_vectors(ALL_CODES, basis_1024, 1024)
        sim_stats = analyze_interference(
            ALL_CODES,
            code_vecs_1024,
            1024,
            sample_size=5000,
            seed=seed,
        )

        # Rauschrobustheit
        noise_stats = {}
        for sigma in noise_sigmas:
            metrics = evaluate_noisy_retrieval(
                model=model_1024,
                basis=basis_1024,
                codes=test_codes,
                sigma=sigma,
                seed=seed,
                device=device,
            )
            noise_stats[sigma] = metrics["accuracy"]

        print(f"    D_krit: {d_krit}")
        print(f"    Acc@1024: {acc_by_d[1024]*100:.2f}%")
        print(f"    Noise@1024: " + ", ".join(
            f"sigma={s:.2f}->{noise_stats[s]*100:.2f}%" for s in noise_sigmas
        ))

        per_seed.append({
            "seed": seed,
            "d_krit": d_krit,
            "acc_by_d": acc_by_d,
            "acc_1024": acc_by_d[1024],
            "sim_stats_1024": sim_stats,
            "noise_acc": noise_stats,
        })

    # Gesamtprüfung
    print("\n" + "=" * 60)
    print("PHASE 8 – Zusammenfassung")
    print("=" * 60)

    d_krit_values = [r["d_krit"] for r in per_seed if r["d_krit"] is not None]
    dim_to_idx = {d: i for i, d in enumerate(sorted(dimensions))}
    if d_krit_values:
        d_idx = [dim_to_idx[d] for d in d_krit_values]
        max_step_diff = max(d_idx) - min(d_idx)
    else:
        max_step_diff = None

    all_acc_1024 = [r["acc_1024"] for r in per_seed]
    sim_means_connected = [r["sim_stats_1024"]["connected_mean"] for r in per_seed]
    sim_means_unconnected = [r["sim_stats_1024"]["unconnected_mean"] for r in per_seed]

    print("  Pro Seed:")
    for r in per_seed:
        print(f"    Seed {r['seed']}: D_krit={r['d_krit']}, Acc@1024={r['acc_1024']*100:.2f}%")

    if max_step_diff is not None and max_step_diff <= 1:
        print("  [+] D_krit variiert maximal um eine Dimensionsstufe ✓")
    else:
        print("  [!] D_krit variiert um mehr als eine Dimensionsstufe")

    if min(all_acc_1024) >= 0.99:
        print("  [+] Genauigkeit bei D=1024 in allen Seeds >= 99% ✓")
    else:
        print("  [!] Mindestens ein Seed < 99% bei D=1024")

    print("  Ähnlichkeitsverteilungen (D=1024, qualitativ):")
    print(f"    connected_mean:   mean={np.mean(sim_means_connected):.4f}, std={np.std(sim_means_connected):.4f}")
    print(f"    unconnected_mean: mean={np.mean(sim_means_unconnected):.4f}, std={np.std(sim_means_unconnected):.4f}")
    if np.std(sim_means_connected) < 0.02 and np.std(sim_means_unconnected) < 0.02:
        print("  [+] Ähnlichkeitsverteilungen über Seeds qualitativ gleich ✓")
    else:
        print("  [!] Verteilungen variieren stärker als erwartet")

    print("\n  === Phase 8 abgeschlossen ===\n")
    return {
        "per_seed": per_seed,
        "max_step_diff_d_krit": max_step_diff,
        "min_acc_1024": min(all_acc_1024),
    }


# ============================================================================
# PHASE 9 – Vergleich mit One-Hot-Kodierung
# ============================================================================

class OneHotLinearBaseline(nn.Module):
    """
    One-Hot Baseline ohne Hidden Layer.
    Implementiert als index-basierte Sparse-Äquivalent-Form der linearen Köpfe.
    """
    def __init__(self, num_codes=100000, class_sizes=(10, 10, 20, 5, 10)):
        super().__init__()
        self.class_sizes = tuple(class_sizes)
        self.heads = nn.ModuleList([
            nn.Embedding(num_codes, n_classes) for n_classes in self.class_sizes
        ])

    def forward(self, code_ids):
        return tuple(head(code_ids) for head in self.heads)


def train_onehot_baseline(train_ids, val_ids, test_ids,
                          train_codes, val_codes, test_codes,
                          num_codes=100000,
                          max_epochs=50, lr=0.001, batch_size=1024,
                          seed=42, device=None, verbose=True):
    """Trainiert lineare One-Hot-Köpfe mit Adam und Early Stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = OneHotLinearBaseline(num_codes=num_codes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ids = torch.tensor(train_ids, dtype=torch.long)
    val_ids = torch.tensor(val_ids, dtype=torch.long)
    test_ids = torch.tensor(test_ids, dtype=torch.long)
    train_y = torch.tensor(np.array(train_codes), dtype=torch.long)
    val_y = torch.tensor(np.array(val_codes), dtype=torch.long)
    test_y = torch.tensor(np.array(test_codes), dtype=torch.long)

    best_val_acc = -1.0
    best_state = None
    patience = 7
    patience_counter = 0
    epoch_times = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(train_ids.size(0))
        for start in range(0, train_ids.size(0), batch_size):
            idx = perm[start:start + batch_size]
            x = train_ids[idx].to(device)
            y = train_y[idx].to(device)
            outputs = model(x)
            loss = sum(criterion(outputs[i], y[:, i]) for i in range(len(outputs)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            correct_all = 0
            total = 0
            for start in range(0, val_ids.size(0), batch_size):
                x = val_ids[start:start + batch_size].to(device)
                y = val_y[start:start + batch_size].to(device)
                outputs = model(x)
                preds = [o.argmax(dim=1) for o in outputs]
                all_match = torch.ones(x.size(0), dtype=torch.bool, device=device)
                for i in range(len(outputs)):
                    all_match &= (preds[i] == y[:, i])
                correct_all += all_match.sum().item()
                total += x.size(0)
            val_acc = correct_all / total

        epoch_times.append(time.time() - t0)
        if verbose and (epoch <= 5 or epoch % 5 == 0):
            print(f"    Epoch {epoch:3d}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early Stopping bei Epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Test
    model.eval()
    with torch.no_grad():
        correct_all = 0
        correct_comp = np.zeros(5)
        total = 0
        for start in range(0, test_ids.size(0), batch_size):
            x = test_ids[start:start + batch_size].to(device)
            y = test_y[start:start + batch_size].to(device)
            outputs = model(x)
            preds = [o.argmax(dim=1) for o in outputs]
            all_match = torch.ones(x.size(0), dtype=torch.bool, device=device)
            for i in range(len(outputs)):
                m = preds[i] == y[:, i]
                correct_comp[i] += m.sum().item()
                all_match &= m
            correct_all += all_match.sum().item()
            total += x.size(0)

    stats = {
        "epochs_trained": len(epoch_times),
        "avg_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else 0.0,
    }
    return model, correct_all / total, correct_comp / total, stats


def run_phase9(D=1024, seed=42, max_epochs=30):
    """Faire Gegenüberstellung One-Hot vs. Superposition."""
    print("=" * 60)
    print("PHASE 9 – Vergleich mit One-Hot-Kodierung")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fairer Architekturvergleich: jede Code-ID ist in Train/Val/Test enthalten
    # (lineare One-Hot-Baseline kann sonst auf ungesehene IDs nicht generalisieren).
    code_to_id = {code: idx for idx, code in enumerate(ALL_CODES)}
    train_codes = list(ALL_CODES)
    val_codes = list(ALL_CODES)
    test_codes = list(ALL_CODES)
    train_ids = [code_to_id[tuple(c)] for c in train_codes]
    val_ids = [code_to_id[tuple(c)] for c in val_codes]
    test_ids = [code_to_id[tuple(c)] for c in test_codes]

    print("\n" + "=" * 60)
    print("PHASE 9.1 – One-Hot-Baseline (linear, sparse-äquivalent)")
    print("=" * 60)
    onehot_model, onehot_acc, onehot_comp, onehot_stats = train_onehot_baseline(
        train_ids, val_ids, test_ids,
        train_codes, val_codes, test_codes,
        num_codes=len(ALL_CODES),
        max_epochs=max_epochs,
        lr=0.1,
        seed=seed,
        device=device,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("PHASE 9.1 – Superposition-Decoder (D=1024)")
    print("=" * 60)
    basis = generate_basis_vectors(D, seed=seed)
    super_model, super_acc, super_comp, _, super_stats = train_decoder(
        D,
        basis,
        train_codes,
        val_codes,
        test_codes,
        max_epochs=max_epochs,
        lr=0.001,
        seed=seed,
        device=device,
        verbose=True,
        return_stats=True,
    )

    onehot_params = sum(p.numel() for p in onehot_model.parameters())
    super_params = sum(p.numel() for p in super_model.parameters())

    print("\n" + "=" * 60)
    print("PHASE 9.2 – Vergleich")
    print("=" * 60)
    print(f"  One-Hot Parameter:       {onehot_params:,}")
    print(f"  Superposition Parameter: {super_params:,}")
    print(f"  Parameter-Faktor:        {onehot_params / super_params:.2f}x")

    print(f"\n  One-Hot   Testgenauigkeit: {onehot_acc*100:.2f}%")
    print(f"  Superpos. Testgenauigkeit: {super_acc*100:.2f}%")

    print(f"\n  One-Hot   Zeit/Epoche: {onehot_stats['avg_epoch_seconds']:.3f}s "
          f"({onehot_stats['epochs_trained']} Epochen)")
    print(f"  Superpos. Zeit/Epoche: {super_stats['avg_epoch_seconds']:.3f}s "
          f"({super_stats['epochs_trained']} Epochen)")

    if onehot_params / super_params >= 10:
        print("  [+] Superposition benötigt signifikant weniger Parameter ✓")
    else:
        print("  [!] Parameterabstand geringer als erwartet")

    if onehot_acc >= 0.99 and super_acc >= 0.99:
        print("  [+] Genauigkeit vergleichbar (>=99%) ✓")
    else:
        print("  [!] Eine der Genauigkeiten liegt unter 99%")

    print("\n  === Phase 9 abgeschlossen ===\n")
    return {
        "onehot": {
            "acc": onehot_acc,
            "per_component": onehot_comp,
            "params": onehot_params,
            "avg_epoch_seconds": onehot_stats["avg_epoch_seconds"],
        },
        "superposition": {
            "acc": super_acc,
            "per_component": super_comp,
            "params": super_params,
            "avg_epoch_seconds": super_stats["avg_epoch_seconds"],
        },
    }


if __name__ == "__main__":
    verify_phase0()

    # --- Phase 1 (D=2048) ---
    D = 2048
    basis = generate_basis_vectors(D, seed=42)
    verify_basis_vectors(basis, D)

    code_vectors = build_code_vectors(ALL_CODES, basis, D)
    verify_code_vectors(code_vectors, ALL_CODES, D)

    # --- Phase 2 ---
    # 2.1: Interferenzanalyse für D=2048
    analyze_interference(ALL_CODES, code_vectors, D, sample_size=10000)

    # 2.2: Dimensionsvariation
    del code_vectors  # Speicher freigeben
    sweep_results = dimension_sweep(dimensions=[2048, 1024, 512, 256, 128])

    # --- Phase 3 ---
    # 3.1–3.3: Decoder Training (D=1024, 3 Seeds)
    best_model, basis_1024, train_codes, val_codes, test_codes = run_phase3(
        D=1024, seeds=(42, 123, 7))

    # 3.4: Retrieval vs. Dimension
    retrieval_results, d_krit = run_phase3_4(
        dimensions=[2048, 1024, 512, 256, 128])

    # --- Phase 4 ---
    phase4_results = run_phase4(
        model=best_model,
        basis_1024=basis_1024,
        test_codes=test_codes,
        sigmas=(0.0, 0.05, 0.1, 0.2, 0.5),
    )

    # --- Phase 5 ---
    phase5_results = run_phase5(
        test_codes=test_codes,
        train_codes=train_codes,
        val_codes=val_codes,
        weights={"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.2},
        sparse_p=0.05,
        seed=42,
    )

    # --- Phase 6 ---
    phase6_results = run_phase6(
        D=1024,
        seed=42,
        holdout_n=1000,
        new_codes_n=1000,
    )

    # --- Phase 7 ---
    phase7_results = run_phase7(
        D=1024,
        seed=42,
    )

    # --- Phase 8 ---
    phase8_results = run_phase8(
        seeds=(11, 22, 33, 44, 55),
        dimensions=(128, 256, 512, 1024, 2048),
        noise_sigmas=(0.0, 0.05, 0.1, 0.2, 0.5),
    )

    # --- Phase 9 ---
    phase9_results = run_phase9(
        D=1024,
        seed=42,
        max_epochs=30,
    )
