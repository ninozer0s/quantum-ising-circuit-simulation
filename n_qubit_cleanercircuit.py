from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
from qiskit.visualization import circuit_drawer
import numpy as np, itertools, os
import matplotlib.pyplot as plt
from collections import OrderedDict

# -----------------
# User parameters
# -----------------
n = 4   
lam_list = [0.6, 1.2, 1.8]
shots = 2 * 4096
flip_all_for_lam_gt_1 = True
out_dir = "ising_histos"
circ_dir = "circuits"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(circ_dir, exist_ok=True)

# -----------------
# Display style
# -----------------
def extend_style_for_internal(style, n, color="#add8e6"):
    for k in range(n // 2):
        style["displaycolor"][f"F†[{k}|{n}]"] = (color, "black")
    style["displaycolor"]["B†"] = (color, "black")
    return style

def make_block_only_style(block_names, block_color="#add8e6"):
    try:
        from qiskit.visualization.matplotlib import matplotlib_default_style
        style = matplotlib_default_style()
    except:
        style = {"displaycolor": {"default": ("#add8e6", "black")}}

    if "displaycolor" not in style:
       style = {"displaycolor": {"default": ("#add8e6", "black")}}

    for name in block_names:
        style["displaycolor"][name] = (block_color, "black")
    return style

BLOCK_NAMES = ["FFT†", "Bogoliubov†"]
BLOCK_STYLE = make_block_only_style(BLOCK_NAMES, "#add8e6")
BLOCK_STYLE = extend_style_for_internal(BLOCK_STYLE, n, "#add8e6")

# -----------------
# Transformation blocks
# -----------------
def F_block_dagger(k, n):
    tw = np.exp(2j * np.pi * k / n)
    s2 = 1 / np.sqrt(2)
    F = np.array([
        [1, 0,     0,       0],
        [0, s2,  s2*tw,     0],
        [0, s2, -s2*tw,     0],
        [0, 0,     0,     -tw],
    ], dtype=complex)
    inst = Operator(F.conj().T).to_instruction()
    inst.name = f"F†[{k}|{n}]"
    phi = 2*np.pi*k/n
    inst.label = f"φ={phi:.3f}"
    return inst

def B_block_dagger(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    B = np.array([
        [ c, 0, 0, 1j*s],
        [ 0, 1, 0,   0 ],
        [ 0, 0, 1,   0 ],
        [1j*s, 0, 0,  c ],
    ], dtype=complex)
    inst = Operator(B.conj().T).to_instruction()
    inst.name = "B†"
    inst.label = f"θ={theta:.3f}"
    return inst

def theta_k(k, n, lam):
    ck = np.cos(2*np.pi*k/n)
    sk = np.sin(2*np.pi*k/n)
    val = (lam - ck) / np.sqrt((lam-ck)**2 + sk**2)
    return np.arccos(np.clip(val, -1, 1))

# -----------------
# Internal subcircuits
# -----------------
def fermionic_fft_dagger_subcircuit(n):
    qc = QuantumCircuit(n, name="FFT†")
    size = 2
    while size <= n:
        half = size // 2
        for j in range(0, n, size):
            for r in range(half):
                a = j + r
                b = j + r + half
                k = r * (n // size)
                qc.append(F_block_dagger(k, n), [a, b])
        if size < n:
            qc.barrier(range(n))
        size *= 2
    return qc

def bogoliubov_layer_subcircuit(n, lam):
    qc = QuantumCircuit(n, name="Bogoliubov†")
    for k in range(1, n//2):
        a = k
        b = (n - k) % n
        if a < b:
            qc.append(B_block_dagger(theta_k(k, n, lam)), [a, b])
    return qc

# -----------------
# Helper functions
# -----------------
def all_bitstrings(n):
    return [''.join(bits) for bits in itertools.product('01', repeat=n)]

def counts_full_norm(counts, n):
    total = sum(counts.values())
    return OrderedDict((b, counts.get(b, 0)/total) for b in all_bitstrings(n))

# -----------------
# n=4 case
# -----------------
def build_circuit_counts_n4(lam):
    n_local = 4
    qc = QuantumCircuit(n_local, n_local)

    # initial state
    if lam < 1:
        qc.x(1)
        qc.x(2)
        qc.x(3)
    else:
        # |0000>
        pass

    qc.barrier()

    # Undo Bogoliubov
    qc.append(bogoliubov_layer_subcircuit(n_local, lam).to_instruction(), range(n_local))
    qc.barrier()

    # Undo FFT
    qc.append(fermionic_fft_dagger_subcircuit(n_local).to_instruction(), range(n_local))
    qc.barrier()

    # optional flip for λ>1, for main peak to be at 1111
    if lam > 1 and flip_all_for_lam_gt_1:
        for q in range(n_local):
            qc.x(q)

    # qubit order correction
    qc.swap(3, 2)

    qc.measure(range(n_local), range(n_local))
    return qc

# -----------------
# general circuit for n>4
# -----------------
def build_circuit_counts_generic(n, lam):
    qc = QuantumCircuit(n, n)

    # General initial state
    # |000...01> for λ<1, |000...00> for λ>1
    if lam < 1:
        qc.x(n-1)
    else:
        pass

    qc.barrier()

    # Undo Bogoliubov
    qc.append(bogoliubov_layer_subcircuit(n, lam).to_instruction(), range(n))
    qc.barrier()

    # Undo FFT
    qc.append(fermionic_fft_dagger_subcircuit(n).to_instruction(), range(n))
    qc.barrier()

    # Optional flip for λ>1, so main peak appears at 111...1
    if lam > 1 and flip_all_for_lam_gt_1:
        for q in range(n):
            qc.x(q)

    qc.measure(range(n), range(n))
    return qc

# defines the correct circuit build
def build_circuit_counts(n, lam):
    if n == 4:
        return build_circuit_counts_n4(lam)
    else:
        return build_circuit_counts_generic(n, lam)

# -----------------
# Draw standalone internal circuits
# -----------------
fft_circ = fermionic_fft_dagger_subcircuit(n)
circuit_drawer(
    fft_circ,
    output="mpl",
    style=BLOCK_STYLE,
    reverse_bits=False,
    filename=os.path.join(circ_dir, f"fft_dagger_n{n}.png")
)

# -----------------
# Main loop
# -----------------
backend = Aer.get_backend("aer_simulator")
results = []

for lam in lam_list:
    qc = build_circuit_counts(n, lam)

    # Bogoliubov†-layer
    bog_circ = bogoliubov_layer_subcircuit(n, lam)
    circuit_drawer(
        bog_circ,
        output="mpl",
        style=BLOCK_STYLE,
        reverse_bits=False,
        filename=os.path.join(circ_dir, f"bogoliubov_dagger_n{n}_lam_{lam:.2f}.png")
    )

    circuit_drawer(
        qc,
        output="mpl",
        style=BLOCK_STYLE,
        reverse_bits=False,
        filename=os.path.join(circ_dir, f"full_circuit_n{n}_lam_{lam:.2f}.png")
    )

    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    full = counts_full_norm(counts, n)
    results.append((lam, full))

    print(f"\nTop 8 bitstrings for λ={lam}:")
    for bit, p in sorted(full.items(), key=lambda x:x[1], reverse=True)[:8]:
        print(f"  {bit}  p={p:.6f}")

    labels = all_bitstrings(n)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.bar(x, [full[b] for b in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    plt.title(f"Ising n={n} — counts (λ={lam}, shots={shots})")
    plt.ylabel("Probability")
    plt.ylim(0, max(full.values()) * 1.25)
    plt.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"hist_lambda_{lam:.2f}_n{n}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# combined grid
labels = all_bitstrings(n)
cols = len(results)
fig, axes = plt.subplots(1, cols, figsize=(5*cols, 4), sharey=True)
if cols == 1:
    axes = [axes]
ymax = max(max(f.values()) for _, f in results) * 1.25
for ax, (lam, full) in zip(axes, results):
    x = np.arange(len(labels))
    ax.bar(x, [full[b] for b in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(f"λ={lam}")
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', alpha=0.3)

axes[0].set_ylabel("Probability")
fig.suptitle(f"Ising n={n} — counts (shots={shots})", y=0.98)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"histos_grid_n{n}.png"),
            dpi=150, bbox_inches="tight")
plt.show()
