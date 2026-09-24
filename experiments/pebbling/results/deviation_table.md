# Bennett k-level predictions vs. measured reversible programs

Corpus: reversible-algorithms, 120 programs that run under PyJanus (jana2014 dialect). Metrics: time_ratio = total_steps/fwd_steps, transient_peak = max_live_vars - max(live_init, live_final), k_space = transient_peak/log2(total_steps), k_time = log2(time_ratio). Medians over the class.

## Table 1 - deviation of each program class from the k-level prediction

| class | rule | count | representative programs | measured signature | Bennett k-level prediction for this shape | cause of the deviation |
|---|---|---|---|---|---|---|
| A1 in-place | n_uncalls = 0, other families | 54 (4 with transient_peak = 0, 50 with > 0) | gcd, bsort1, hsort2, avl_insert | median time_ratio 1.00, transient_peak 2.5, k_space 0.42 | k = 0: reversible as written, no uncomputation; T_rev = T, S_extra = 0 (Bennett not applicable) | hand-written in-place algorithms need no CCU embedding; the ancilla is a constant number of loop/index variables |
| A2 DP/table in-place | n_uncalls = 0, family in {dp, floyd_warshall, bellman_ford, apsp} | 5 | knapsack, lcs, edit_distance, floyd_warshall | median time_ratio 1.00, transient_peak 2, k_space 0.31 | k = 0 likewise; a pebbling view would charge the DP table as garbage (S_extra ~ table size) or log T for re-computation | the DP table is the *output* (live_final 5-11), not garbage; nothing is ever uncomputed, so time_ratio = 1 and the transient is 2 index variables |
| B partial uncompute | n_uncalls > 0, time_ratio < 1.6 | 16 | dijkstra, qsort2, minheap_Nohr2015, avl_search | median time_ratio 1.34, transient_peak 4, k_space 0.65 | between k = 0 and k = 1: T_rev = T + T_uncalled part, S_extra = O(|outputs of uncalled part|) | only a sub-procedure is call/uncalled (e.g. a search or heapify); the bulk of main stays forward |
| C single-level CCU | 1.6 <= time_ratio <= 2.05 | 37 | bsort2, msort1, hsort1, lis | median time_ratio 1.81, transient_peak 4, k_space 0.56 | k = 1: T_rev = 2T; docstring S_extra = log2 T, Bennett 1989 S_extra = O(|outputs|) (constant) | the CCU is real but main has extra forward work and the uncall repeats only the called body, so T_rev/T sits at 1.6-1.95, and the ancilla is a constant (median 4), not log2 T |
| D nested Bennett | time_ratio > 2.05 | 5 | fact, perm2decfac, rank, isort1 | median time_ratio 2.96, transient_peak 9, k_space 0.81 | k levels: docstring T_rev = T*2^k, S_extra = k*log2 T; nested CCU gives S_extra = k * (ancillas per level) | recursion with call+uncall at every level doubles the work per level (k = depth) but adds ~1 ancilla per level, so space is k*const, not k*log2 T |
| E inverse-as-main | n_uncalls > 0, fwd_steps < 0.10 * total_steps | 3 | rank_lexicographic, decfac2rank, ssort1 | median time_ratio 18.00, transient_peak 7, k_space 0.86 | no prediction: T_rev/T is undefined when the algorithm *is* an uncall | the forward work is an `uncall` of a generating procedure; fwd/uncall counting inverts the roles and time_ratio is an accounting artifact |

## Table 2 - every D and E program, and the dp family, individually

| path | class | time_ratio | transient_peak | k_space | k_time | call_depth_max | n_uncalls |
|---|---|---|---|---|---|---|---|
| dp/edit_distance.j | A2 | 1.0 | 2 | 0.245 | 0.0 | 1 | 0 |
| dp/knapsack.j | A2 | 1.0 | 2 | 0.314 | 0.0 | 1 | 0 |
| dp/lcs.j | A2 | 1.0 | 2 | 0.271 | 0.0 | 1 | 0 |
| dp/lis.j | C | 1.7299 | 5 | 0.563 | 0.791 | 2 | 9 |
| others/examples/fact.j | D | 7.2564 | 4 | 0.491 | 2.859 | 6 | 31 |
| isort/code/isort1.j | D | 3.8611 | 6 | 0.739 | 1.949 | 3 | 12 |
| perm/code/perm2decfac.j | D | 2.9631 | 11 | 1.158 | 1.567 | 3 | 23 |
| rank/rank.j | D | 2.7586 | 9 | 0.812 | 1.464 | 3 | 124 |
| others/examples/breadth_first_search.j | D | 2.2439 | 10 | 0.969 | 1.166 | 5 | 81 |
| rank/rank_lexicographic.j | E | 72.2 | 11 | 1.158 | 6.174 | 3 | 24 |
| others/code/decfac2rank.j | E | 18.0 | 4 | 0.774 | 4.17 | 2 | 1 |
| ssort/code/ssort1.j | E | 16.4118 | 7 | 0.862 | 4.037 | 4 | 7 |

## Table 3 - scaling series (6 input sizes per program)

| program | size | time_ratio first -> last | transient_peak first -> last | slope transient_peak vs log2 T | slope log2 T vs size |
|---|---|---|---|---|---|
| numeric/gcd.j | 6 -> 21 | 1 -> 1 | 0 -> 0 | +0.000 | 0.118 (T x1.08 per +1) |
| others/code/factorial.j | 5 -> 160 | 1 -> 1 | 2 -> 2 | +0.000 | 0.027 (T x1.02 per +1) |
| dp/lcs.j | 3 -> 24 | 1 -> 1 | 2 -> 2 | +0.000 | 0.247 (T x1.19 per +1) |
| dp/knapsack.j | 2 -> 16 | 1 -> 1 | 2 -> 2 | -0.000 | 0.357 (T x1.28 per +1) |
| dp/lis.j | 4 -> 32 | 1.57 -> 1.9 | 5 -> 5 | -0.000 | 0.170 (T x1.12 per +1) |
| bsort/bsort2.j | 4 -> 24 | 1.83 -> 1.97 | 5 -> 5 | +0.000 | 0.227 (T x1.17 per +1) |
| msort/msort1.j | 2 -> 64 | 1.85 -> 1.98 | 5 -> 12 | +0.946 | 0.096 (T x1.07 per +1) |
| others/examples/fact.j | 2 -> 12 | 1.72 -> 419 | 1 -> 11 | +0.982 | 1.018 (T x2.02 per +1) |

Reading: a k-level Bennett scheme with k growing as log2 T would show slope 1 in column 5; a fixed single-level CCU shows 0. fact.j's slope is per unit of *n* (recursion depth), which is the nested-embedding regime k = n, S_extra = k * 1 ancilla.

## RFCL single-level check (rfcl_k1_measurements.csv)

28 ok rows over 6 SRL programs: S_pred = S + |out| + 1 is an upper bound that is tight on 24 rows; on the other 4 rows (bennett_rif.srl [0 20 40]; bennett_rif.srl [0 80 160]; branch_copy.srl [20 0]; branch_copy.srl [80 0]) an output value is 0, so peak_space (a non-zero count) sits below it (S_bennett - S_pred in [-2, -1, 0]); T_bennett - T_pred in [5] (constant +5), T_bennett - 2T in [8, 9, 10].
