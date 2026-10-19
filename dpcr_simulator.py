"""
A simple tool to simulate one or more dPCR experiments.

In the former case, usage is

python3.12 poisson_dPCR_simulationv2.py -p (number of partitions -n (molecules in initial sample) -m (mode: poisson or multinomial)
-fp (false positive rate) -fn (false negative rate, per molecule) -s (seed for the random number generator)


In the latter case, the usage is

python3.12 poisson_dPCR_simulationv2.py -c (csv parameter file) -r (number of experiments for each parameter set) -o (csv file with results)

The param file should contain no header, and should have the columns:

p,n,mode,fp,fn,seed

(corresponding to the options listed above)

fp and fn should be given as written-out integers in either case and not in exponential notation.

As stated, -r is the number of experiments that each row (i.e., each set of parameters) will be run across. The csv output file will report
averages and standard deviations for the error (i.e., the difference between estimated and actual lambda). 

"""
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--partitions",
    type=int,
    help="number of dPCR partitions (single-run mode only)"
)

parser.add_argument(
    "-n", "--NMol",
    type=int,
    help="number of target molecules in the partitioned sample (single-run mode only)"
)

parser.add_argument(
    "-m",
    "--mode",
    choices=["poisson", "multinomial"],
    help="option 1: Poisson, simulating a Poisson process approximating n molecules. option 2: Multinomial, taking *exactly* n molecules and multinomially assigning them to p partitions (single-run mode only)"
)

parser.add_argument(
    "-fp",
    "--fp-rate",
    type=float,
    default=0.0,
    help="false positive rate: probability a truly-empty partition is read as positive (default 0.0)"
)

parser.add_argument(
    "-fn",
    "--fn-rate",
    type=float,
    default=0.0,
    help="per-molecule false negative (dropout) rate: probability a single molecule in a partition fails to be detected. A well with k molecules is falsely called negative with probability fn_rate**k (default 0.0)"
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=None,
    help="optional RNG seed for reproducibility (single-run mode only)"
)

parser.add_argument(
    "-c", "--csv",
    type=str,
    default=None,
    help="path to a CSV of parameter sets (columns: p,n,mode,fp,fn,seed ; no header). If given, runs a batch sweep instead of a single run."
)

parser.add_argument(
    "-r", "--replicates",
    type=int,
    default=1000,
    help="number of replicate experiments per parameter set in sweep mode (default 1000)"
)

parser.add_argument(
    "-o", "--out",
    type=str,
    default="sweep_results.csv",
    help="output CSV path for sweep mode summary statistics (default sweep_results.csv)"
)

args = parser.parse_args()


def simulate_dPCR(mode, n_partitions, true_n_mol, size=1, rng=None):
    """
    Simulate the dPCR process, batched over `size` independent replicates.

    Returns an array of shape (size, n_partitions): raw (pre-error) molecule
    counts per partition, for each of `size` independent experiments.

    if mode = "poisson": each replicate draws n_partitions samples from a
               Poisson distribution centered at lam(bda) = true_n_mol / n_partitions
               (the average concentration per partition volume).
    if mode = "multinomial": each replicate assigns exactly true_n_mol molecules
               across n_partitions wells uniformly at random.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if mode == "poisson":
        lam = true_n_mol / n_partitions
        partitions = rng.poisson(lam=lam, size=(size, n_partitions))

    elif mode == "multinomial":
        # Draw a (size, true_n_mol) matrix of partition assignments, one row per replicate.
        #That is, "which partition gets this particular molecule
        #with #partitions>#molecules, most partition ids will only appear once
        #otherwise, there will often be more than two "molecules" per partition
        chosen = rng.integers(0, n_partitions, size=(size, true_n_mol)) 
        # Offset each replicate's indices into non-overlapping ranges of one flat
        # array, so a single bincount() call can tally all replicates at once
        offsets = (np.arange(size) * n_partitions)[:, None] #column vector of multiples of size
        flat_indices = (chosen + offsets).ravel()
        flat_counts = np.bincount(flat_indices, minlength=size * n_partitions)
        partitions = flat_counts.reshape(size, n_partitions)

    else:
        raise ValueError(f"unknown mode: {mode}")

    return partitions


def digitalize(partitions, fp_rate=0.0, fn_rate=0.0, rng=None):
    """
    Digitalize imitates the dPCR luminescence based presence/absence read.
    Raw per-partition molecule counts are converted into binary calls.
    Works on arrays of any shape (single 1D run or batched 2D replicates) —

    fp_rate: probability that an empty partition is "misread" as positive.
    fn_rate: probability that a non-empty partition is "misread" as negative. It differs
             from fp_rate in an important way: it is PER-MOLECULE! fn_rate^k, where
             k is the actual number of molecules present in the well, is the chance the
             well will be misread as negative. This reflects the luminescence-based
             approach; a well with fewer molecules is more likely to be missed because
             its fluorescence will be lower.
    """
    rng = rng if rng is not None else np.random.default_rng()
    partitions = np.asarray(partitions)  # safety feature if digitalize is called from elsewhere

    true_positive_mask = partitions > 0
    true_negative_mask = ~true_positive_mask

    digital = np.zeros(partitions.shape, dtype=bool)

    n_true_neg = np.sum(true_negative_mask)
    fp_flips = rng.random(n_true_neg) < fp_rate
    digital[true_negative_mask] = fp_flips

    k = partitions[true_positive_mask]
    p_false_neg = fn_rate ** k
    fn_flips = rng.random(len(k)) < p_false_neg
    digital[true_positive_mask] = ~fn_flips

    return digital


def recover_lam(digital_partitions):
    """
    MLE recovery of lambda from binary (+/-) partition calls.

    Works on a 1D array (single run -> returns a scalar) or a 2D array of
    shape (n_replicates, n_partitions) -> returns one lambda_hat per replicate,
    summing along the partitions axis (axis=-1) rather than flattening
    everything together.
    """
    digital_partitions = np.asarray(digital_partitions)
    n_partitions = digital_partitions.shape[-1]
    n_neg = np.sum(~digital_partitions, axis=-1)

    with np.errstate(divide="ignore"):
        lam_hat = -np.log(n_neg / n_partitions)

    # saturated replicates (n_neg == 0) produce -log(0) = inf; leave as inf,
    # consistent with the original single-run behavior.
    return lam_hat


def run_single(mode, n_partitions, true_n_mol, fp_rate, fn_rate, seed):
    """Single-run mode: does not use multiple runs, but also does not interface with pandas, merely printing out one experiment."""
    rng = np.random.default_rng(seed)

    raw_partitions = simulate_dPCR(mode, n_partitions, true_n_mol, size=1, rng=rng)[0]
    digital_partitions = digitalize(raw_partitions, fp_rate=fp_rate, fn_rate=fn_rate, rng=rng)

    true_lam = true_n_mol / n_partitions
    lam_hat = recover_lam(digital_partitions)

    n_pos_true = np.sum(raw_partitions > 0)
    n_pos_observed = np.sum(digital_partitions)
    observed_error = (lam_hat - true_lam)/true_lam

    print(f"true lambda:          {true_lam:.4f}")
    print(f"recovered lambda:     {lam_hat:.4f}")
    print(f"observed error:   {observed_error:.4f}")
    print(f"recovered N:          {lam_hat * n_partitions:.2f}" if np.isfinite(lam_hat) else "recovered N: inf (saturated)")
    print(f"true positive wells:  {n_pos_true} / {n_partitions}")
    print(f"observed pos. wells:  {n_pos_observed} / {n_partitions}")


def run_sweep(csv_path, n_replicates, out_path):
    """
    Batch sweep mode: reads a CSV of parameter sets (columns p,n,mode,fp,fn,seed,
    no header), runs n_replicates vectorized experiments per row, and writes a
    summary CSV with mean/std of lambda_hat and error across replicates for
    each parameter set.
    """
    params = pd.read_csv(
        csv_path,
        header=None,
        names=["p", "n", "mode", "fp", "fn", "seed"],
    )

    per_replicate_rows = []

    for _, row in params.iterrows():
        p = int(row["p"])
        n = int(row["n"])
        mode = row["mode"]
        fp = float(row["fp"])
        fn = float(row["fn"])
        seed = int(row["seed"]) if not pd.isna(row["seed"]) else None

        rng = np.random.default_rng(seed)

        raw = simulate_dPCR(mode, p, n, size=n_replicates, rng=rng)
        digital = digitalize(raw, fp_rate=fp, fn_rate=fn, rng=rng)
        lam_hat = recover_lam(digital)  # shape (n_replicates,)

        lam_true = n / p
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = (lam_hat - lam_true) / lam_true if lam_true != 0 else np.full(n_replicates, np.nan)

        # saturated replicates (lam_hat == inf) shouldn't silently poison mean/std
        # with inf; treat them as NaN for the summary stats and report how many
        # were saturated separately. 
        lam_hat_clean = np.where(np.isfinite(lam_hat), lam_hat, np.nan)
        error_clean = np.where(np.isfinite(relative_error), relative_error, np.nan)
        n_saturated = int(np.sum(~np.isfinite(lam_hat)))

        for rep_idx in range(n_replicates):
            per_replicate_rows.append({
                "p": p, "n": n, "lambda_true": lam_true, "mode": mode,
                "fp": fp, "fn": fn, "seed": seed,
                "replicate": rep_idx,
                "lambda_hat": lam_hat_clean[rep_idx],
                "error": error_clean[rep_idx],
                "saturated": not np.isfinite(lam_hat[rep_idx]),
            })

    long_df = pd.DataFrame(per_replicate_rows)

    group_cols = ["p", "n", "lambda_true", "mode", "fp", "fn", "seed"]
    summary = (
        long_df
        .groupby(group_cols, dropna=False)
        .agg(
            mean_lambda_hat=("lambda_hat", "mean"),
            std_lambda_hat=("lambda_hat", "std"),
            mean_error=("error", "mean"),
            std_error=("error", "std"),
            n_saturated=("saturated", "sum"),
            n_replicates=("replicate", "count"),
        )
        .reset_index()
    )

    summary.to_csv(out_path, index=False)
    print(f"Wrote summary for {len(summary)} parameter set(s), "
          f"{n_replicates} replicates each, to {out_path}")

if __name__ == "__main__":
    if args.csv is not None:
        run_sweep(args.csv, args.replicates, args.out)
    else:
        missing = [name for name, val in
                   [("-p", args.partitions), ("-n", args.NMol), ("-m", args.mode)]
                   if val is None]
        if missing:
            parser.error(f"single-run mode requires {', '.join(missing)} "
                         f"(or use -c/--csv for sweep mode)")
        run_single(args.mode, args.partitions, args.NMol, args.fp_rate, args.fn_rate, args.seed)
