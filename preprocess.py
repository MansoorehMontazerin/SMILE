"""Feynman dataset loader used by the SMILE pipeline."""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class custom_data(Dataset):
    def __init__(
        self,
        eqn="I.14.4",
        logT=True,
        frac=None,                 # e.g., 0.10 => keep exactly 10% of ORIGINAL rows
        shuffle=True,
        seed=42,
        positive_only=True,        # keep only rows where all chosen columns are > 0
        positive_on="inputs",      # "inputs" or "all" (on RAW df before drop)
        data_dir="data/Feynman_with_units/",
    ):
        """
        Guarantee: if frac is not None, we keep exactly round(frac * N_raw) rows,
        where N_raw is the original row count BEFORE positivity filtering.
        All kept rows satisfy the positivity constraint if positive_only=True.
        If the positive pool is smaller than n_keep, we sample with replacement.
        """
        # 1) Load raw (last column is the one dropped below)
        df_raw = pd.read_csv(f"{data_dir}{eqn}", header=None, sep=" ")
        N_raw = len(df_raw)
        original_num_cols = df_raw.shape[1]

        # 2) Positivity mask on RAW df
        if positive_only:
            if positive_on == "all":
                pos_mask = (df_raw > 0).all(axis=1)
            else:  # "inputs": check all columns except the last original one
                pos_mask = (df_raw.iloc[:, :-1] > 0).all(axis=1)
        else:
            pos_mask = np.ones(N_raw, dtype=bool)

        df_pos = df_raw.loc[pos_mask].reset_index(drop=True)

        # 3) Decide exactly how many rows to keep based on ORIGINAL size
        if frac is not None:
            n_keep = int(max(1, round(frac * N_raw)))
        else:
            n_keep = len(df_pos)  # keep all positive rows if frac not specified

        # If there are no positive rows but positivity required → we can't satisfy the request
        if positive_only and len(df_pos) == 0:
            raise ValueError("No positive rows available under the chosen positivity setting.")

        # 4) Sample from the positive pool to hit n_keep (with replacement if needed)
        need_replace = len(df_pos) < n_keep
        kept = df_pos.sample(n=n_keep, replace=need_replace, random_state=seed)

        # 5) Optional shuffle for final order
        if shuffle:
            kept = kept.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # 6) Drop the last original column
        kept.drop(kept.columns[original_num_cols - 1], axis=1, inplace=True)
        self.num_columns = kept.shape[1]

        # 7) To numpy + optional log
        arr = kept.astype(float).to_numpy(copy=False)
        if logT:
            arr = np.log(arr + 1e-10)

        # 8) Add constant column of ones at the end
        ones = np.ones((arr.shape[0], 1), dtype=float)
        arr = np.concatenate([arr, ones], axis=1)

        # 9) Buffers
        self._X = arr
        self.count = arr.shape[1]        # includes the constant
        self._length = arr.shape[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # inputs = all columns except the penultimate (treated as "output")
        # outputs = the penultimate column
        row = self._X[idx, :]
        y = row[self.count - 2]
        x = row[np.arange(self.count) != (self.count - 2)]
        return x.astype("float32"), np.array([y], dtype="float32")
