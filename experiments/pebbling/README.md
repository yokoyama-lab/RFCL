# Bennett / pebbling measurements of real reversible programs

Small experiment: do real Janus programs (reversible-algorithms, janus-examples) and
RFCL's own Bennett transform follow the k-level prediction "T·2^k time, S + k·log₂T
space" written in `pyrev_fl/tradeoff.py:9-10`? Result: `REPORT.md` (1 figure, 1 table).

Files: `measure_corpus.py` (runs a corpus under PyJanus, writes
`results/corpus_measurements.csv`, `results/scaling_measurements.csv`,
`results/timelines/*.jsonl`), `rfcl_k1_sweep.py` (single-level Bennett on RFCL's SRL
examples, writes `results/rfcl_k1_measurements.csv`), `analyze.py` (classifies, fits,
writes `results/classification.csv`, `results/deviation_table.md`, `results/summary.json`
and `results/fig_bennett_vs_measured.{png,pdf,svg}`).

Run, from this directory, with /usr/bin/python3 (numpy + matplotlib needed by analyze.py only):
    PYJANUS=/path/to/PyJanus python3 measure_corpus.py /path/to/reversible-algorithms /path/to/janus-examples --out results
    python3 rfcl_k1_sweep.py
    python3 analyze.py
Nothing outside this directory is modified; PyJanus and the corpora are read only.
