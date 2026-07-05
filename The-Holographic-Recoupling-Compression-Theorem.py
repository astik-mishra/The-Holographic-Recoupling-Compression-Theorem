```python
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import time

class HolographicRecouplingCompressor:
    """
    Simulates the Holographic Recoupling Compression Theorem (HRCT).
    
    This class constructs a gapped local Hamiltonian (Transverse Field Ising Model),
    finds its ground state, computes the Fractal Entanglement Dimension (D_f),
    applies the Holographic Recoupling Map (compression via Schmidt truncation 
    bounded by the theorem), and verifies the observable error bounds.
    """
    
    def __init__(self, N: int, J: float, h: float):
        """
        Initializes the simulator for an N-qubit system.
        
        Args:
            N (int): Number of qubits (spins).
            J (float): Interaction strength (e.g., 1.0).
            h (float): Transverse field strength. Must not equal J to ensure a spectral gap.
        """
        assert N > 2 and N % 2 == 0, "N must be an even integer > 2."
        assert J != h, "J must not equal h to maintain a non-zero spectral gap (avoiding criticality)."
        
        self.N = N
        self.J = J
        self.h = h
        self.dim = 2**N
        
        # Theorem parameters
        self.delta = 2 * abs(J - h)  # Analytical gap for 1D TFIM
        self.c = 1.0 / (8.0 * self.delta)  # c < 1/(4*Delta) as per theorem
        
        print(f"--- HRCT Simulation Initialized ---")
        print(f"System Size (N): {N} qubits")
        print(f"Hamiltonian Gap (Δ): {self.delta:.4f}")
        print(f"Theorem constant (c): {self.c:.4f}")
        print(f"Hilbert Space Dimension: {self.dim}")
        print("------------------------------------\n")

    def _pauli_matrices(self):
        """Returns sparse Pauli X and Z matrices."""
        Z = sparse.diags([1, -1], 0, format='csr', dtype=np.float64)
        X = sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        I = sparse.eye(2, format='csr', dtype=np.float64)
        return I, X, Z

    def build_hamiltonian(self) -> sparse.csr_matrix:
        """Constructs the 1D Transverse-Field Ising Model Hamiltonian."""
        I, X, Z = self._pauli_matrices()
        H = sparse.csr_matrix((self.dim, self.dim), dtype=np.float64)
        
        # Interaction term: -J * sum(Z_i Z_{i+1})
        for i in range(self.N - 1):
            ops = [I] * self.N
            ops[i] = Z
            ops[i+1] = Z
            term = sparse.kron(ops[0], ops[1], format='csr')
            for k in range(2, self.N):
                term = sparse.kron(term, ops[k], format='csr')
            H -= self.J * term
            
        # Transverse field term: -h * sum(X_i)
        for i in range(self.N):
            ops = [I] * self.N
            ops[i] = X
            term = sparse.kron(ops[0], ops[1], format='csr')
            for k in range(2, self.N):
                term = sparse.kron(term, ops[k], format='csr')
            H -= self.h * term
            
        self.H = H
        return H

    def get_ground_state(self) -> np.ndarray:
        """Computes the ground state of the Hamiltonian using Lanczos algorithm."""
        print("Computing ground state via Lanczos...")
        start_time = time.time()
        
        # v0 ensures deterministic results
        v0 = np.ones(self.dim) / np.sqrt(self.dim)
        # 'SA' finds Smallest Algebraic eigenvalue
        eigenvalues, eigenvectors = spla.eigsh(self.H, k=1, which='SA', v0=v0)
        
        self.psi0 = eigenvectors[:, 0]
        self.energy = eigenvalues[0]
        
        print(f"Ground state computed in {time.time() - start_time:.4f}s. Energy: {self.energy:.6f}\n")
        return self.psi0

    def compute_fractal_dimension(self, epsilon: float) -> float:
        """
        Computes the Fractal Entanglement Dimension (D_f) from the entanglement spectrum.
        For a gapped system, D_f is derived from the scaling of the Schmidt rank 
        with respect to the tolerance epsilon.
        """
        L = self.N // 2
        # Reshape state vector into matrix of shape (2^L, 2^(N-L))
        psi_matrix = self.psi0.reshape((2**L, 2**(self.N - L)))
        
        # Singular Value Decomposition -> Schmidt coefficients
        # np.linalg.svd is numerically stable for this
        U, S, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
        
        # Entanglement spectrum (eigenvalues of reduced density matrix)
        entanglement_spectrum = S**2
        
        # Compute D_f based on spectral rank scaling: r(epsilon) ~ epsilon^(-D_f)
        # Therefore, D_f = -log(r) / log(epsilon)
        filtered_spectrum = entanglement_spectrum[entanglement_spectrum > epsilon]
        rank = len(filtered_spectrum)
        
        if rank <= 1:
            D_f = 0.0  # Product state limit
        else:
            D_f = -np.log(rank) / np.log(epsilon)
            
        # Theorem constraint: D_f <= 2
        self.D_f = min(D_f, 2.0)
        self.Schmidt_U = U
        self.Schmidt_S = S
        self.Schmidt_Vh = Vh
        
        print(f"Entanglement Spectrum computed. Schmidt Rank > ε: {rank}")
        print(f"Computed Fractal Entanglement Dimension (D_f): {self.D_f:.4f}\n")
        return self.D_f

    def compress_and_reconstruct(self, epsilon: float) -> np.ndarray:
        """
        Applies the Holographic Recoupling Map (Phi) to compress and reconstruct the state.
        The compressed dimension is strictly bounded by the HRCT theorem.
        """
        # Calculate theoretical bound for compressed dimension
        # dim_comp = O(N * exp(c * D_f * log^2(1/epsilon)))
        log_inv_eps = np.log(1.0 / epsilon)
        exponent = self.c * self.D_f * (log_inv_eps ** 2)
        
        # Exact integer bound based on theorem formula
        dim_comp_bound = int(np.ceil(self.N * np.exp(exponent)))
        dim_comp_bound = min(dim_comp_bound, len(self.Schmidt_S)) # Cannot exceed actual Schmidt rank
        
        self.dim_comp = dim_comp_bound
        
        print(f"Applying Holographic Recoupling Map...")
        print(f"Target Compression Dimension (H_comp): {self.dim_comp}")
        print(f"Original Dimension: {self.dim} | Compression Ratio: {self.dim / self.dim_comp:.2f}x")
        
        # Truncate Schmidt decomposition to the bound
        U_c = self.Schmidt_U[:, :self.dim_comp]
        S_c = self.Schmidt_S[:self.dim_comp]
        Vh_c = self.Schmidt_Vh[:self.dim_comp, :]
        
        # Reconstruct state vector
        # psi_comp = (U_c * S_c) @ Vh_c 
        # Using broadcasting for efficient multiplication
        psi_comp = (U_c * S_c) @ Vh_c
        psi_comp = psi_comp.flatten()
        
        # Renormalize to ensure valid quantum state (due to truncation)
        norm = np.linalg.norm(psi_comp)
        if norm > 0:
            psi_comp = psi_comp / norm
            
        self.psi_comp = psi_comp
        print("State successfully compressed and reconstructed.\n")
        return psi_comp

    def verify_observables(self, epsilon: float) -> bool:
        """
        Verifies the theorem's bound: |<O>_orig - <O>_comp| < epsilon * ||O||
        Uses the local observable Z_0 * Z_1 (Nearest-neighbor correlation).
        """
        I, X, Z = self._pauli_matrices()
        
        # Construct Z_0 * Z_1 observable
        ops = [I] * self.N
        ops[0] = Z
        ops[1] = Z
        O = sparse.kron(ops[0], ops[1], format='csr')
        for k in range(2, self.N):
            O = sparse.kron(O, ops[k], format='csr')
            
        # Calculate expectations
        # <O> = <psi|O|psi>
        exp_orig = np.vdot(self.psi0, O.dot(self.psi0)).real
        exp_comp = np.vdot(self.psi_comp, O.dot(self.psi_comp)).real
        
        # Operator norm ||O|| for Pauli string is 1.0
        op_norm = 1.0 
        error = abs(exp_orig - exp_comp)
        bound = epsilon * op_norm
        
        print("--- Observable Verification ---")
        print(f"Observable: Z_0 Z_1")
        print(f"Expectation (Original): {exp_orig:.6f}")
        print(f"Expectation (Compressed): {exp_comp:.6f}")
        print(f"Absolute Error: {error:.6e}")
        print(f"Theorem Bound (ε * ||O||): {bound:.6e}")
        
        if error < bound:
            print("RESULT: PASSED. Error is strictly within theorem bounds.\n")
            return True
        else:
            print("RESULT: FAILED. Error exceeded theorem bounds.\n")
            return False

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Parameters chosen to ensure gapped Hamiltonian and testable dimensions
    # N=12 (dim=4096), J=1.0, h=0.5 (Gap = 2*|1.0-0.5| = 1.0)
    N_QUBITS = 12
    J_INTERACTION = 1.0
    H_FIELD = 0.5
    EPSILON = 0.01  # 1% tolerance
    
    print("Initializing HRCT Simulator...\n")
    simulator = HolographicRecouplingCompressor(N=N_QUBITS, J=J_INTERACTION, h=H_FIELD)
    
    # Step 1: Build Hamiltonian and find Ground State
    simulator.build_hamiltonian()
    simulator.get_ground_state()
    
    # Step 2: Compute Fractal Entanglement Dimension (D_f)
    simulator.compute_fractal_dimension(epsilon=EPSILON)
    
    # Step 3: Apply Holographic Recoupling Compression
    simulator.compress_and_reconstruct(epsilon=EPSILON)
    
    # Step 4: Verify Observable Error Bounds
    success = simulator.verify_observables(epsilon=EPSILON)
    
    if success:
        print("Simulation completed successfully. The HRCT bounds hold.")
    else:
        print("Simulation failed bounds check.")
```
