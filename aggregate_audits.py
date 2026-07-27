"""Pool the cross-domain matching results across audit runs.

A single audit does not have the statistical power to separate the shared
channel from the base one: with ~21 pairs per channel the smallest detectable
difference is around 40 points, and the one we are chasing is 15. Pooling
several runs -- different clusterings, different seeds, different checkpoints --
buys that power.

The runs are not fully independent (the same model, the same catalogue), so the
pooled p-value is optimistic. It is reported alongside the per-run results so
the reader can see whether the effect is consistent or driven by one run.

    python aggregate_audits.py concept_audit/audit_*.json
"""

import argparse
import glob
import json
from math import comb


def binomial_p(k, n, p=0.25):
    """P(X >= k) under the null that the LLM is guessing."""
    if not n:
        return None
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def fisher_two_sided(a, b, c, d):
    """Exact test on the 2x2 table [[a, b], [c, d]]."""
    def hyper(a, b, c, d):
        return comb(a + b, a) * comb(c + d, c) / comb(a + b + c + d, a + c)

    observed = hyper(a, b, c, d)
    total = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k = a + b - i, a + c - i
        l = c + d - k
        if j < 0 or k < 0 or l < 0:
            continue
        p = hyper(i, j, k, l)
        if p <= observed + 1e-12:
            total += p
    return total


def main():
    parser = argparse.ArgumentParser(description="Pool cross-domain matching results")
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    pooled = {}
    print(f"{'run':<44} {'canale':<8} {'accordi':>9} {'accuratezza':>12}")
    for path in paths:
        with open(path, encoding='utf-8') as f:
            report = json.load(f)
        checkpoint = report['model_checkpoint'].split('/')[-1].replace('.pth', '')
        for channel_report in report['channels']:
            channel = channel_report['channel']
            matching = channel_report['cross_domain_matching']
            agree = sum(1 for r in matching['per_pair'] if r['agree'] is True)
            scored = sum(1 for r in matching['per_pair'] if r['agree'] is not None)
            if not scored:
                continue
            pooled.setdefault(channel, [0, 0])
            pooled[channel][0] += agree
            pooled[channel][1] += scored
            print(f"{checkpoint[:36] + ' k' + str(len(channel_report['geometric_pairing'])):<44} "
                  f"{channel:<8} {f'{agree}/{scored}':>9} {agree / scored * 100:>11.1f}%")

    print(f"\n{'=' * 62}\nAGGREGATO\n{'=' * 62}")
    for channel, (agree, scored) in pooled.items():
        print(f"  {channel:<8} {agree}/{scored} = {agree / scored * 100:.1f}%  "
              f"p contro il caso (25%) = {binomial_p(agree, scored):.5f}")

    if 'shared' in pooled and 'base' in pooled:
        a, n1 = pooled['shared']
        c, n2 = pooled['base']
        p = fisher_two_sided(a, n1 - a, c, n2 - c)
        print(f"\n  differenza shared - base: "
              f"{(a / n1 - c / n2) * 100:+.1f} punti, Fisher p = {p:.4f}")
        print(f"  {'CONCLUSIVO' if p < 0.05 else 'NON CONCLUSIVO'} al 5%")


if __name__ == '__main__':
    main()
