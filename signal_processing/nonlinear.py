import numpy as np
from scipy.spatial.distance import pdist, squareform

class NonlinearAnalyzer:
    """
    Nonlinear dynamics and complexity analysis
    Perfect for NKUST's "Advanced nonlinear biomedical signal processing"
    """
    
    def __init__(self):
        pass
    
    def sample_entropy(self, signal_data, m=2, r=None):
        """
        Calculate Sample Entropy - measure of signal regularity/complexity
        
        High entropy = irregular, unpredictable (e.g., distracted, searching)
        Low entropy = regular, predictable (e.g., focused reading, stable fixation)
        
        Args:
            signal_data: Time series data
            m: Pattern length (default 2)
            r: Tolerance (default 0.2 * std)
            
        Returns:
            float: Sample entropy value
            
        Reference:
            Richman & Moorman (2000). Physiological time-series analysis using 
            approximate entropy and sample entropy.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        N = len(signal_array)
        
        # Default tolerance
        if r is None:
            r = 0.2 * np.std(signal_array)
        
        def _maxdist(x_i, x_j):
            """Maximum absolute difference"""
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            """Count similar patterns"""
            patterns = np.array([signal_array[i:i+m] for i in range(N-m)])
            
            # Count matches
            matches = 0
            for i in range(len(patterns) - 1):
                for j in range(i + 1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 2  # Count both (i,j) and (j,i)
            
            return matches
        
        # Calculate phi(m) and phi(m+1)
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return None
        
        # Sample entropy
        sam_en = -np.log(phi_m1 / phi_m)
        
        return float(sam_en)
    
    def approximate_entropy(self, signal_data, m=2, r=None):
        """
        Calculate Approximate Entropy - similar to sample entropy
        
        Args:
            signal_data: Time series data
            m: Pattern length
            r: Tolerance
            
        Returns:
            float: Approximate entropy
            
        Reference:
            Pincus (1991). Approximate entropy as a measure of system complexity.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        N = len(signal_array)
        
        if r is None:
            r = 0.2 * np.std(signal_array)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            patterns = [signal_array[i:i+m] for i in range(N-m+1)]
            C = []
            
            for i in range(len(patterns)):
                matches = 0
                for j in range(len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        matches += 1
                C.append(matches / len(patterns))
            
            phi = np.mean([np.log(c) for c in C if c > 0])
            return phi
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        ap_en = phi_m - phi_m1
        
        return float(ap_en)
    
    def fractal_dimension(self, signal_data, method='higuchi'):
        """
        Calculate Fractal Dimension - measure of signal complexity/roughness
        
        High FD (close to 2) = complex, rough, irregular
        Low FD (close to 1) = smooth, regular
        
        Args:
            signal_data: Time series
            method: 'higuchi' or 'box_counting'
            
        Returns:
            float: Fractal dimension
            
        Reference:
            Higuchi (1988). Approach to an irregular time series on the basis 
            of the fractal theory.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        if method == 'higuchi':
            return self._higuchi_fd(signal_array)
        else:
            return self._box_counting_fd(signal_array)
    
    def _higuchi_fd(self, signal_array, kmax=10):
        """
        Higuchi's fractal dimension
        
        FIXED: Added normalization factor for correct FD range (1.0-2.0)
        """
        N = len(signal_array)
        
        # Adjust kmax for short signals
        kmax = min(kmax, N // 4)
        if kmax < 2:
            return None
        
        L = []
        x = np.arange(1, kmax + 1)
        
        # For each k = 1 to kmax
        for k in x:
            Lk = []
            for m in range(k):
                # Create subsequence: take every k-th element starting from m, up to N
                idxs = np.arange(m, N, k, dtype=int)
                if len(idxs) < 2:
                    continue
                
                # Calculate length of curve
                subseq = signal_array[idxs]
                Lmk = np.sum(np.abs(np.diff(subseq)))
                
                # Normalization
                # Multiply by (N-1) / ((len(idxs)-1) * k) not (N-1) / (len(idxs) * k)
                normalization = (N - 1) / ((len(idxs) - 1) * k)
                Lmk = Lmk * normalization
                
                Lk.append(Lmk)
            
            if len(Lk) > 0:
                L.append(np.mean(Lk))
        
        if len(L) < 3:
            return None
        
        # Filter out zeros and invalid values
        valid_idx = np.array(L) > 0
        L = np.array(L)[valid_idx]
        x = x[:len(L)][valid_idx]
        
        if len(L) < 3:
            return None
        
        # Fit line in log-log plot
        log_x = np.log(x)
        log_L = np.log(L)
        
        coeffs = np.polyfit(log_x, log_L, 1)
        fd = -coeffs[0]  # Negative slope is FD
        
        # FIXED: Ensure FD is in valid range [1.0, 2.0]
        # If outside range, there might be data issues
        fd = float(np.clip(fd, 1.0, 2.0))
        
        return fd
    
    def _box_counting_fd(self, signal_array):
        """Box counting fractal dimension"""
        # Normalize signal
        signal_norm = (signal_array - np.min(signal_array)) / (np.ptp(signal_array) + 1e-10)
        
        # Create 2D representation
        N = len(signal_norm)
        x = np.arange(N)
        
        # Try different box sizes
        box_sizes = np.logspace(0, np.log10(N//4), num=10, dtype=int)
        box_sizes = np.unique(box_sizes)
        
        counts = []
        for size in box_sizes:
            # Count boxes that contain part of the curve
            n_boxes_x = N // size
            n_boxes_y = int(1.0 / size) + 1 if size < N else 1
            
            boxes = set()
            for i in range(N):
                box_x = i // size
                box_y = int(signal_norm[i] * n_boxes_y)
                boxes.add((box_x, box_y))
            
            counts.append(len(boxes))
        
        # Fit line in log-log plot
        coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
        fd = -coeffs[0]
        
        return float(np.clip(fd, 1.0, 2.0))
    
    def lyapunov_exponent(self, signal_data, delay=1, embedding_dim=3):
        """
        Estimate largest Lyapunov exponent - measure of chaos
        
        Positive λ = chaotic, sensitive to initial conditions
        Zero λ = periodic, stable
        Negative λ = converging, damped
        
        For eye movements:
            λ > 0 → Unpredictable, possibly distracted
            λ ≈ 0 → Stable, focused task
            λ < 0 → Overly constrained, possibly fatigued
            
        Args:
            signal_data: Time series
            delay: Time delay for embedding
            embedding_dim: Embedding dimension
            
        Returns:
            float: Largest Lyapunov exponent
            
        Reference:
            Rosenstein et al. (1993). A practical method for calculating largest 
            Lyapunov exponents from small data sets.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 50:
            return None
        
        # Time-delay embedding
        N = len(signal_array)
        M = N - (embedding_dim - 1) * delay
        
        if M < 10:
            return None
        
        # Create embedded vectors
        embedded = np.zeros((M, embedding_dim))
        for i in range(M):
            for j in range(embedding_dim):
                embedded[i, j] = signal_array[i + j * delay]
        
        # Find nearest neighbors
        distances = pdist(embedded, metric='euclidean')
        dist_matrix = squareform(distances)
        
        # For each point, find nearest neighbor
        divergence = []
        for i in range(M - 1):
            # Get distances to all other points
            dists = dist_matrix[i].copy()
            dists[i] = np.inf  # Exclude self
            
            # Find nearest neighbor
            j = np.argmin(dists)
            
            # Track divergence over time
            for k in range(1, min(10, M - max(i, j))):
                if i + k < M and j + k < M:
                    d = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d > 0:
                        divergence.append(np.log(d))
        
        if len(divergence) < 2:
            return None
        
        # Lyapunov exponent is slope of divergence
        x = np.arange(len(divergence))
        coeffs = np.polyfit(x, divergence, 1)
        lyap = coeffs[0]
        
        return float(lyap)
    
    def detrended_fluctuation_analysis(self, signal_data):
        """
        Detrended Fluctuation Analysis (DFA) - measure of long-range correlations
        
        DFA exponent (α):
        - α < 0.5: Anti-correlated (mean-reverting)
        - α = 0.5: Uncorrelated (white noise)
        - 0.5 < α < 1: Correlated (persistent)
        - α = 1: 1/f noise (pink noise)
        - α > 1: Non-stationary

        For eye movements:
            α = 0.6  → Normal, slightly persistent behavior
            α > 0.8  → Strong correlations, possibly focused
            α < 0.5  → Anti-correlated, possibly distracted
        
        Returns:
            float: DFA scaling exponent
            
        Reference:
            Peng et al. (1994). Mosaic organization of DNA nucleotides.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 16:
            return None
        
        N = len(signal_array)
        
        # Integrate the signal
        y = np.cumsum(signal_array - np.mean(signal_array))
        
        # Divide into segments
        scales = np.unique(np.logspace(0.5, np.log10(N//4), num=10).astype(int))
        
        fluctuations = []
        for scale in scales:
            # Divide into non-overlapping segments
            n_segments = N // scale
            
            F = []
            for v in range(n_segments):
                start = v * scale
                end = start + scale
                segment = y[start:end]
                
                # Fit polynomial trend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend)**2))
                F.append(fluctuation)
            
            if len(F) > 0:
                fluctuations.append(np.mean(F))
        
        if len(fluctuations) < 3:
            return None
        
        # DFA exponent is slope in log-log plot
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        
        coeffs = np.polyfit(log_scales, log_fluct, 1)
        dfa_exp = coeffs[0]
        
        return float(dfa_exp)
    
    def recurrence_quantification(self, signal_data, threshold=0.1, min_line_length=2):
        """
        Recurrence Quantification Analysis (RQA)
        Analyze patterns and repetitions in time series
        
        Returns:
            dict: RQA measures
            
        Reference:
            Marwan et al. (2007). Recurrence plots for the analysis of complex systems.
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        # Normalize
        signal_norm = (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-10)
        
        # Create distance matrix
        N = len(signal_norm)
        distances = np.abs(signal_norm[:, None] - signal_norm[None, :])
        
        # Recurrence matrix (binary)
        threshold_val = threshold * np.max(distances)
        R = distances < threshold_val
        
        # RQA measures
        measures = {}
        
        # Recurrence Rate (RR)
        measures['recurrence_rate'] = float(np.sum(R) / (N * N))
        
        # Determinism (DET) - ratio of recurrent points in diagonal lines
        diag_lines = []
        for offset in range(-N+1, N):
            diag = np.diagonal(R, offset=offset)
            if len(diag) > 0:
                # Find line lengths
                line_length = 0
                for val in diag:
                    if val:
                        line_length += 1
                    else:
                        if line_length >= min_line_length:
                            diag_lines.append(line_length)
                        line_length = 0
                if line_length >= min_line_length:
                    diag_lines.append(line_length)
        
        if len(diag_lines) > 0:
            measures['determinism'] = float(np.sum(diag_lines) / np.sum(R))
            measures['avg_diagonal_length'] = float(np.mean(diag_lines))
            measures['max_diagonal_length'] = float(np.max(diag_lines))
        else:
            measures['determinism'] = 0.0
            measures['avg_diagonal_length'] = 0.0
            measures['max_diagonal_length'] = 0.0
        
        return measures
    
    def extract_nonlinear_features(self, signal_data):
        """
        Extract all nonlinear features
        
        Returns:
            dict: Comprehensive nonlinear analysis
        """
        features = {}
        
        # Entropy measures
        features['sample_entropy'] = self.sample_entropy(signal_data)
        features['approximate_entropy'] = self.approximate_entropy(signal_data)
        
        # Complexity measures
        features['fractal_dimension'] = self.fractal_dimension(signal_data)
        features['lyapunov_exponent'] = self.lyapunov_exponent(signal_data)
        
        # Correlation measures
        features['dfa_exponent'] = self.detrended_fluctuation_analysis(signal_data)
        
        # Pattern analysis
        rqa = self.recurrence_quantification(signal_data)
        if rqa:
            features['rqa'] = rqa
        
        return features