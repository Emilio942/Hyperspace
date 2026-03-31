# Hyperspace - PRISM Project

## Overview
**PRISM (Packed Representations via Interference-Safe Superposition Mapping)** is a research framework for encoding discrete structured data into high-dimensional vector spaces. By leveraging the principles of superposition and concentration of measure, PRISM allows for the creation of robust, scalable, and hardware-efficient representations of complex information.

## Key Features
- **Structural Mapping:** Discretely structured codes are mapped to the unit sphere $S^{D-1}$ such that their cosine similarity directly reflects their component overlap.
- **Mathematical Rigor:** Every aspect of the embedding—from interference patterns to capacity limits—is grounded in high-dimensional geometry and probability theory.
- **Hardware Efficiency:** Support for 1-bit quantization, allowing the entire system to run using simple bitwise operations (XNOR/Popcount).
- **Extreme Robustness:** Designed for high-reliability environments (e.g., satellite deployments) where memory corruption and radiation-induced bit-flips are common.

## Mathematical Results
Our analysis has yielded several critical theoretical insights:

1.  **The PRISM Kernel ($k/K$):** The expected cosine similarity between two codes sharing $k$ out of $K$ components is exactly:
    $$\mathbb{E}[\langle v_1, v_2 \rangle | k] = \frac{k}{K}$$
    This proves that the mapping is an isometry between the Hamming-like discrete space and the continuous vector space.

2.  **Theoretical Capacity ($N_{\max}$):** The maximum number of safe code vectors $N$ for a given dimension $D$ and interference threshold $t$ follows the scaling law:
    $$N_{\max} \approx \exp\left(\frac{D t^2}{4}\right)$$
    For $D=1024$ and $t=0.5$, the capacity exceeds 26 million codes.

3.  **1-Bit Quantization (Bussgang/Grothendieck Gain):** Quantizing vectors to $\{-1, 1\}^D$ preserves angular fidelity with a gain of:
    $$\gamma = \sqrt{\frac{2}{\pi}} \approx 0.798$$
    This allows for a 32x reduction in memory with minimal loss in retrieval accuracy ($>99.9\%$).

4.  **Satellite Robustness (Bit-Flip Resilience):**
    - **1% Bit-Flips:** ~99.96% Accuracy
    - **10% Bit-Flips:** ~99.63% Accuracy
    PRISM exhibits "Graceful Degradation," maintaining high performance even under significant data corruption.

5.  **Orthogonality Trade-off:** We compared **Quasiorthogonal** (random) and **Perfectly Orthogonal** (QR-based) bases. While perfect orthogonality eliminates interference noise, quasiorthogonal bases provide better statistical buffers against out-of-distribution noise and hardware drifts.

## Project Structure
- `prism.py`: The core implementation containing the 12 phases of research, from code generation to satellite robustness simulation.
- `5d_interference/`: Experimental modules for early-stage interference sweeps.
- `PRISM_Aufgabenliste.txt`: The roadmap and task list for the project development.

## How to Run
Ensure you have the required dependencies installed (PyTorch, NumPy, PyYAML):
```bash
python prism.py
```
The script will run through all 12 phases and provide a detailed report of both empirical measurements and theoretical predictions.
