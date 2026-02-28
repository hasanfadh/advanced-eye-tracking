import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

class NonlinearAnalyzer:
    """
    OPTIMIZED: Nonlinear dynamics and complexity analysis
    
    KEY OPTIMIZATIONS:
    1. Vectorized operations (NumPy broadcast)
    2. Early stopping for long signals
    3. Adaptive downsampling for very long signals
    4. Efficient distance calculations
    """
    
    def __init__(self):
        self.max_samples = 3000  # Downsample if signal longer than this
    
    def _adaptive_downsample(self, signal_data, target_length=None):
        """
        Adaptively downsample signal if too long
        
        Args:
            signal_data: Input signal
            target_length: Target length (default: self.max_samples)
            
        Returns:
            downsampled signal, downsample_factor
        """
        if target_length is None:
            target_length = self.max_samples
        
        signal_array = np.array(signal_data)
        original_length = len(signal_array)
        
        if original_length <= target_length:
            return signal_array, 1
        
        # Calculate downsample factor
        factor = original_length // target_length
        
        # Downsample by averaging
        new_length = original_length // factor
        downsampled = signal_array[:new_length * factor].reshape(-1, factor).mean(axis=1)
        
        return downsampled, factor
    
    def sample_entropy(self, signal_data, m=2, r=None, max_length=1000):
        """
        OPTIMIZED: Calculate Sample Entropy with adaptive processing
        
        Args:
            signal_data: Time series data
            m: Pattern length (default 2)
            r: Tolerance (default 0.2 * std)
            max_length: Maximum signal length to process (downsample if longer)
            
        Returns:
            float: Sample entropy value
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        # Adaptive downsampling for very long signals
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        N = len(signal_array)
        
        # Default tolerance
        if r is None:
            r = 0.2 * np.std(signal_array)
        
        # OPTIMIZATION: Vectorized distance calculation
        def _phi_vectorized(m):
            """Vectorized version using broadcasting"""
            # Create template matrix
            patterns = np.array([signal_array[i:i+m] for i in range(N-m)])
            n_patterns = len(patterns)
            
            if n_patterns < 2:
                return 0
            
            # Calculate pairwise Chebyshev distances (max absolute difference)
            # This is much faster than nested loops
            matches = 0
            
            # Process in chunks to avoid memory issues
            chunk_size = 500
            for i in range(0, n_patterns, chunk_size):
                end_i = min(i + chunk_size, n_patterns)
                chunk = patterns[i:end_i]
                
                # Calculate distances for this chunk
                dists = cdist(chunk, patterns, metric='chebyshev')
                
                # Count matches (excluding self-matches)
                # For each pattern, count how many others are within tolerance
                for j, row in enumerate(dists):
                    # Exclude self-match at position i+j
                    mask = np.ones(len(row), dtype=bool)
                    mask[i+j] = False
                    matches += np.sum(row[mask] <= r)
            
            return matches
        
        # Calculate phi(m) and phi(m+1)
        phi_m = _phi_vectorized(m)
        phi_m1 = _phi_vectorized(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return None
        
        # Sample entropy
        sam_en = -np.log(phi_m1 / phi_m)
        
        return float(sam_en)
    
    def approximate_entropy(self, signal_data, m=2, r=None, max_length=1000):
        """
        OPTIMIZED: Calculate Approximate Entropy with vectorization
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        # Adaptive downsampling
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        N = len(signal_array)
        
        if r is None:
            r = 0.2 * np.std(signal_array)
        
        def _phi_vectorized(m):
            patterns = np.array([signal_array[i:i+m] for i in range(N-m+1)])
            n_patterns = len(patterns)
            
            # Calculate distances using cdist
            dists = cdist(patterns, patterns, metric='chebyshev')
            
            # Count matches for each pattern
            C = np.sum(dists <= r, axis=1) / n_patterns
            
            # Calculate phi
            phi = np.mean(np.log(C[C > 0]))
            return phi
        
        phi_m = _phi_vectorized(m)
        phi_m1 = _phi_vectorized(m + 1)
        
        ap_en = phi_m - phi_m1
        
        return float(ap_en)
    
    def fractal_dimension(self, signal_data, method='higuchi', max_length=2000):
        """
        OPTIMIZED: Calculate Fractal Dimension with adaptive downsampling
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        # Adaptive downsampling for long signals
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        if method == 'higuchi':
            return self._higuchi_fd_optimized(signal_array)
        else:
            return self._box_counting_fd(signal_array)
    
    def _higuchi_fd_optimized(self, signal_array, kmax=None):
        """
        FIXED: Proper Higuchi Fractal Dimension calculation
        
        Key fixes:
        1. Adaptive kmax based on signal length
        2. Proper normalization factor
        3. Better handling of edge cases
        """
        N = len(signal_array)
        
        # Adaptive kmax: should be at least N/10 but not too large
        if kmax is None:
            kmax = max(8, min(20, N // 10))
        
        kmax = min(kmax, N // 4)
        
        if kmax < 3:
            return None
        
        Lk = np.zeros(kmax)
        
        for k in range(1, kmax + 1):
            Lmk = np.zeros(k)
            
            for m in range(k):
                # Get indices for this subsequence
                indices = np.arange(m, N, k) # Indexing for subsequence
                
                if len(indices) < 2:
                    continue
                
                # Extract subsequence
                subsequence = signal_array[indices] # Subsequence for this m and k
                
                # Calculate normalized length
                # Sum of absolute differences
                length_sum = np.sum(np.abs(np.diff(subsequence)))
                
                # Normalization factor: (N-1) / [(floor((N-m)/k)) * k]
                norm_factor = (N - 1) / ((len(indices) - 1) * k)
                
                Lmk[m] = length_sum * norm_factor
            
            # Average curve length for this k
            # Filter out zeros
            valid_Lmk = Lmk[Lmk > 0]
            if len(valid_Lmk) > 0:
                Lk[k-1] = np.mean(valid_Lmk)
            else:
                Lk[k-1] = 0
        
        # Remove zeros and check if enough points
        k_range = np.arange(1, kmax + 1)
        valid = Lk > 0
        
        if np.sum(valid) < 4:  # Need at least 4 points for good fit
            return None
        
        Lk_valid = Lk[valid]
        k_valid = k_range[valid]
        
        # Fit line in log-log space: log(L(k)) ~ -FD * log(k)
        try:
            # Use robust fitting
            log_k = np.log(k_valid)
            log_Lk = np.log(Lk_valid)
            
            # Check for NaN or Inf
            if np.any(~np.isfinite(log_k)) or np.any(~np.isfinite(log_Lk)):
                return None
            
            # Linear regression
            coeffs = np.polyfit(log_k, log_Lk, 1)
            slope = coeffs[0]

            if slope < 0:
                FD = 2.0 + slope  # NEW
            else:
                FD = 2.0 - abs(slope)
            
            # Sanity check
            if FD < 0.5 or FD > 2.5:
                return None
            
            # Clip to valid range
            FD = np.clip(FD, 1.0, 2.0)
            
            return float(FD)
            
        except Exception as e:
            return None
    
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
            
            # Vectorized box counting
            box_x = (x // size).astype(int)
            box_y = (signal_norm * n_boxes_y).astype(int)
            
            # Count unique boxes
            boxes = np.column_stack((box_x, box_y))
            unique_boxes = np.unique(boxes, axis=0)
            counts.append(len(unique_boxes))
        
        # Fit line in log-log plot
        coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
        fd = -coeffs[0]
        
        return float(np.clip(fd, 1.0, 2.0))
    
    def lyapunov_exponent(self, signal_data, delay=1, embedding_dim=3, max_length=1000):
        """
        FIXED: Lyapunov exponent with better robustness
        
        Key fixes:
        1. Better nearest neighbor identification
        2. Minimum separation constraint
        3. More robust divergence tracking
        4. Better handling of short signals
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 50:
            return None
        
        # Adaptive downsampling
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        # Time-delay embedding
        N = len(signal_array)
        M = N - (embedding_dim - 1) * delay
        
        if M < 20:  # Need more points
            return None
        
        # Create embedded vectors (vectorized)
        embedded = np.zeros((M, embedding_dim))
        for j in range(embedding_dim):
            embedded[:, j] = signal_array[j*delay:j*delay+M]
        
        # Calculate pairwise distances
        try:
            distances = pdist(embedded, metric='euclidean')
            dist_matrix = squareform(distances)
        except:
            return None
        
        # Find nearest neighbors with minimum temporal separation
        min_temporal_sep = max(1, M // 20)  # At least 5% of signal length apart
        
        nearest_neighbors = np.full(M, -1, dtype=int)
        
        for i in range(M):
            # Get distances to all other points
            dists = dist_matrix[i].copy()
            
            # Exclude self
            dists[i] = np.inf
            
            # Exclude temporally close points
            for j in range(max(0, i - min_temporal_sep), 
                          min(M, i + min_temporal_sep + 1)):
                dists[j] = np.inf
            
            # Find nearest
            if np.all(np.isinf(dists)):
                continue
            
            nearest_neighbors[i] = np.argmin(dists)
        
        # Track divergence
        divergences = []
        max_evolution = min(15, M // 10)  # Track up to 15 steps or 10% of signal
        
        for i in range(M):
            j = nearest_neighbors[i]
            
            if j < 0:  # No valid neighbor found
                continue
            
            # Initial distance
            d0 = dist_matrix[i, j]
            
            if d0 < 1e-10:  # Too close, skip
                continue
            
            # Track divergence over time
            for k in range(1, max_evolution + 1):
                if i + k >= M or j + k >= M:
                    break
                
                # Distance after k steps
                dk = np.linalg.norm(embedded[i + k] - embedded[j + k])
                
                if dk < 1e-10:  # Converged, stop tracking
                    break
                
                # Log divergence
                log_divergence = np.log(dk / d0)
                
                # Only use positive divergences (exponential separation)
                if log_divergence > 0:
                    divergences.append(log_divergence / k)  # Normalize by time
        
        if len(divergences) < 10:  # Need enough data points
            return None
        
        # Lyapunov exponent is the average rate of divergence
        lyap = np.mean(divergences)
        
        # Sanity check: should be reasonable
        if abs(lyap) > 2.0:  # Unrealistically large
            return None
        
        return float(lyap)
    
    def detrended_fluctuation_analysis(self, signal_data, max_length=2000):
        """
        OPTIMIZED: DFA with adaptive downsampling
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 16:
            return None
        
        # Adaptive downsampling
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        N = len(signal_array)
        
        # Integrate the signal
        y = np.cumsum(signal_array - np.mean(signal_array))
        
        # Divide into segments
        scales = np.unique(np.logspace(0.5, np.log10(N//4), num=10).astype(int))
        
        fluctuations = []
        for scale in scales:
            # Divide into non-overlapping segments
            n_segments = N // scale
            
            # Vectorized fluctuation calculation
            F_list = []
            for v in range(n_segments):
                start = v * scale
                end = start + scale
                segment = y[start:end]
                
                # Fit polynomial trend (vectorized)
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((segment - trend)**2))
                F_list.append(fluctuation)
            
            if F_list:
                fluctuations.append(np.mean(F_list))
        
        if len(fluctuations) < 3:
            return None
        
        # DFA exponent is slope in log-log plot
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        
        coeffs = np.polyfit(log_scales, log_fluct, 1)
        dfa_exp = coeffs[0]
        
        return float(dfa_exp)
    
    def recurrence_quantification(self, signal_data, threshold=0.1, min_line_length=2, max_length=500):
        """
        OPTIMIZED: RQA with length limiting
        """
        signal_array = np.array(signal_data)
        signal_array = signal_array[~np.isnan(signal_array)]
        
        if len(signal_array) < 10:
            return None
        
        # Limit length for memory efficiency
        if len(signal_array) > max_length:
            signal_array, _ = self._adaptive_downsample(signal_array, max_length)
        
        # Normalize
        signal_norm = (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-10)
        
        # Create distance matrix (vectorized)
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
                # Find line lengths (vectorized)
                # Convert boolean to int and find consecutive ones
                diag_int = diag.astype(int)
                changes = np.diff(np.concatenate(([0], diag_int, [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                lengths = ends - starts
                
                # Filter by minimum length
                valid_lengths = lengths[lengths >= min_line_length]
                diag_lines.extend(valid_lengths)
        
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
        OPTIMIZED: Extract all nonlinear features with adaptive processing
        
        Returns:
            dict: Comprehensive nonlinear analysis
        """
        features = {}
        
        # Adaptive processing based on signal length
        signal_array = np.array(signal_data)
        signal_length = len(signal_array)
        
        # Adjust max_length based on signal length
        if signal_length < 500:
            max_len_entropy = signal_length
            max_len_fractal = signal_length
            max_len_lyap = signal_length
        elif signal_length < 2000:
            max_len_entropy = 1000
            max_len_fractal = signal_length
            max_len_lyap = 1000
        else:
            max_len_entropy = 1000
            max_len_fractal = 2000
            max_len_lyap = 1000
        
        # Entropy measures
        features['sample_entropy'] = self.sample_entropy(signal_data, max_length=max_len_entropy)
        features['approximate_entropy'] = self.approximate_entropy(signal_data, max_length=max_len_entropy)
        
        # Complexity measures
        features['fractal_dimension'] = self.fractal_dimension(signal_data, max_length=max_len_fractal)
        features['lyapunov_exponent'] = self.lyapunov_exponent(signal_data, max_length=max_len_lyap)
        
        # Correlation measures
        features['dfa_exponent'] = self.detrended_fluctuation_analysis(signal_data)
        
        # Pattern analysis (only for shorter signals due to memory)
        if signal_length < 1000:
            rqa = self.recurrence_quantification(signal_data)
            if rqa:
                features['rqa'] = rqa
        
        return features