"""MPBind algorithm for aptamer ranking from HT-SELEX data."""

import numpy as np
import pandas as pd
from skbase.base import BaseObject

class MPBind(BaseObject):
    """
    MPBind: A Meta-motif-Based Statistical Framework to Predict Binding Potential.

    This algorithm ranks aptamers by their binding potential based on 
    enrichment of their constituent k-mers across multiple SELEX rounds.

    Parameters
    ----------
    k_mer_len : int, default=6
        Length of the k-mers (n-mers) to extract from the sequences.
    pseudo_count : float, default=1.0
        Pseudo-count added to k-mer counts to avoid division by zero.

    References
    ----------
    Jiang et al., "MPBind: A Meta-motif-Based Statistical Framework and Pipeline 
    to Predict Binding Potential of SELEX-Derived Aptamers", Bioinformatics (2014)
    """

    def __init__(self, k_mer_len=6, pseudo_count=1.0):
        self.k_mer_len = k_mer_len
        self.pseudo_count = pseudo_count
        super().__init__()
        self.meta_z_scores_ = {}

    def _get_kmers(self, sequence):
        """Extract all overlapping k-mers from a sequence."""
        k = self.k_mer_len
        if len(sequence) < k:
            return []
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    def fit(self, X):
        """
        Fit the MPBind model by computing k-mer Meta-Z-Scores across rounds.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing SELEX data. Must have a 'sequence' column,
            and integer columns indicating counts in successive rounds 
            (e.g., 'round_0', 'round_1', ...). The columns representing rounds 
            should be ordered chronologically and prefixed with 'round_'.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if "sequence" not in X.columns:
            raise ValueError("X must contain a 'sequence' column.")

        round_cols = [c for c in X.columns if str(c).startswith("round_")]
        if len(round_cols) < 2:
            raise ValueError("X must contain at least two round columns prefixed with 'round_'.")

        # Step 1: Count k-mers in each round
        kmer_counts = {}
        for round_col in round_cols:
            kmer_counts[round_col] = {}

        for _, row in X.iterrows():
            seq = row["sequence"]
            kmers = self._get_kmers(seq)
            for r_col in round_cols:
                count = row[r_col]
                if pd.isna(count) or count == 0:
                    continue
                for kmer in kmers:
                    kmer_counts[r_col][kmer] = kmer_counts[r_col].get(kmer, 0) + count

        # Collect all unique k-mers
        all_kmers = set()
        for r_col in round_cols:
            all_kmers.update(kmer_counts[r_col].keys())
        all_kmers = list(all_kmers)
        num_kmers = len(all_kmers)

        # Step 2: Compute frequencies per round with pseudo-counts
        kmer_freqs = {}
        for r_col in round_cols:
            total_count = sum(kmer_counts[r_col].values())
            denominator = total_count + self.pseudo_count * num_kmers
            kmer_freqs[r_col] = {
                kmer: (kmer_counts[r_col].get(kmer, 0) + self.pseudo_count) / denominator
                for kmer in all_kmers
            }

        # Step 3: Compute enrichment metrics (fold change) and convert to Z-scores
        z_scores_per_transition = []
        for i in range(1, len(round_cols)):
            prev_round = round_cols[i-1]
            curr_round = round_cols[i]
            
            enrichments = []
            for kmer in all_kmers:
                e = kmer_freqs[curr_round][kmer] / kmer_freqs[prev_round][kmer]
                enrichments.append(e)
            
            e_array = np.array(enrichments)
            mean_e = np.mean(e_array)
            std_e = np.std(e_array)
            if std_e == 0:
                z_scores = np.zeros_like(e_array)
            else:
                z_scores = (e_array - mean_e) / std_e
                
            z_scores_per_transition.append(z_scores)

        # Step 4: Stouffer's method for Meta-Z-Score
        z_matrix = np.vstack(z_scores_per_transition) # Shape: (num_transitions, num_kmers)
        num_transitions = z_matrix.shape[0]
        meta_z_array = np.sum(z_matrix, axis=0) / np.sqrt(num_transitions)

        self.meta_z_scores_ = {kmer: meta_z for kmer, meta_z in zip(all_kmers, meta_z_array)}
        
        return self

    def predict(self, X):
        """
        Predict binding potential scores for sequences.

        Parameters
        ----------
        X : pandas.DataFrame or list of str
            Sequences to score. If DataFrame, must contain a 'sequence' column.

        Returns
        -------
        scores : numpy.ndarray
            Array of binding potential scores (Meta-Z-Score average).
        """
        if isinstance(X, pd.DataFrame):
            if "sequence" not in X.columns:
                raise ValueError("X must contain a 'sequence' column.")
            sequences = X["sequence"].tolist()
        else:
            sequences = X

        scores = []
        for seq in sequences:
            kmers = self._get_kmers(seq)
            if not kmers:
                scores.append(0.0)
                continue
            
            # Sum/average Meta-Z-Scores of constituent k-mers
            seq_score = sum(self.meta_z_scores_.get(k, 0.0) for k in kmers) / len(kmers)
            scores.append(seq_score)

        return np.array(scores)
