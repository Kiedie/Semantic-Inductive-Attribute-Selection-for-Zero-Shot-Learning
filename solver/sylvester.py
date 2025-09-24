from scipy.linalg import schur, get_lapack_funcs
import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import solve_sylvester

class SylvesterSolver:
    def __init__(self, B):
        # 1) Pre-factorize B once
        #    we compute schur(Bᵀ) so that in the solve call we only
        #    do the schur of A and the small forward/back-substitution.
        self.S, self.V = schur(B.conj().T, output='real')

        # 2) Grab the LAPACK trsyl routine once
        #    we can use a dummy array of the correct dtype
        #    so that get_lapack_funcs returns the right precision
        dummy = np.empty((1,1), dtype=self.S.dtype)
        self._trsyl, = get_lapack_funcs(('trsyl',), (dummy, self.S, dummy))

    def solve_fast(self, A, Q):
        """
        Solve A X + X B = Q using the pre-factorization of B.
        """
        # schur(A)
        R, U = schur(A, output='real')

        # form F = Uᴴ Q V
        F = U.conj().T @ Q @ self.V

        # solve R Y + Y Sᵀ = F
        Y, scale, info = self._trsyl(R, self.S, F, tranb='C')
        if info < 0:
            raise LinAlgError(f"illegal argument in {-info}-th parameter of TRSYL")
        Y *= scale

        # back-transform: X = U Y Vᴴ
        return U @ Y @ self.V.conj().T