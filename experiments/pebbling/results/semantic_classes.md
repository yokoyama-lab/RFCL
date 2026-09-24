# Syntactic vs. semantic forward work

120 runnable programs of reversible-algorithms (commit pinned in REPORT.md). *Syntactic*: inside `uncall` = reverse (measure_corpus.py). *Strict*: reverse = invocations that exactly undo an earlier one on the same storage. *Loose*: also invocations that restore some storage an earlier one changed (garbage clearing). Same class rules as analyze.py.

## Class sizes

| class | syntactic | strict | loose |
|---|---|---|---|
| A1 | 54 | 73 | 57 |
| A2 | 5 | 5 | 5 |
| B | 16 | 22 | 22 |
| C | 37 | 19 | 32 |
| D | 5 | 1 | 4 |
| E | 3 | 0 | 0 |

## Transitions syntactic -> strict

| syntactic \ strict | A1 | A2 | B | C | D | E | total |
|---|---|---|---|---|---|---|---|
| A1 | 54 |  |  |  |  |  | 54 |
| A2 |  | 5 |  |  |  |  | 5 |
| B | 5 |  | 11 |  |  |  | 16 |
| C | 13 |  | 6 | 18 |  |  | 37 |
| D |  |  | 3 | 1 | 1 |  | 5 |
| E | 1 |  | 2 |  |  |  | 3 |

## Transitions syntactic -> loose

| syntactic \ loose | A1 | A2 | B | C | D | E | total |
|---|---|---|---|---|---|---|---|
| A1 | 53 |  | 1 |  |  |  | 54 |
| A2 |  | 5 |  |  |  |  | 5 |
| B | 2 |  | 13 | 1 |  |  | 16 |
| C | 1 |  | 4 | 31 | 1 |  | 37 |
| D |  |  | 2 |  | 3 |  | 5 |
| E | 1 |  | 2 |  |  |  | 3 |

## Programs whose ratio changes by more than 1 % under either count

| path | class syn / strict / loose | time_ratio syn | strict | loose | uncalls | full pairs | partial | unpaired uncalls | pairing calls |
|---|---|---|---|---|---|---|---|---|---|
| apsp/apsp.j | C / B / C | 1.7243 | 1.1316 | 1.7751 | 32 | 8 | 24 | 0 | 0 |
| avl/avl_search.j | B / B / B | 1.3864 | 1.4524 | 1.4524 | 2 | 2 | 0 | 0 | 0 |
| bsort/bsort2.j | C / A1 / C | 1.8855 | 1 | 1.9 | 1 | 0 | 1 | 0 | 0 |
| bsort/code/bsort2.j | C / A1 / C | 1.8257 | 1 | 1.8426 | 1 | 0 | 1 | 0 | 0 |
| bsort/code/bsort3.j | C / A1 / C | 1.8257 | 1 | 1.8426 | 1 | 0 | 1 | 0 | 0 |
| dijkstra/dijkstra.j | B / B / B | 1.3792 | 1.4006 | 1.4006 | 5 | 5 | 0 | 0 | 0 |
| dp/lis.j | C / B / C | 1.7299 | 1.1049 | 1.7687 | 9 | 1 | 7 | 1 | 0 |
| geometry/closest_pair.j | C / C / C | 1.7816 | 1.8023 | 1.8023 | 1 | 1 | 0 | 0 | 0 |
| geometry/convex_hull.j | C / C / C | 1.6854 | 1.726 | 1.726 | 5 | 5 | 0 | 0 | 0 |
| hsort/code/hsort1.j | C / A1 / C | 1.8873 | 1 | 1.8927 | 37 | 0 | 37 | 0 | 0 |
| hsort/code/hsort2.j | C / A1 / C | 1.8502 | 1 | 1.8571 | 35 | 0 | 14 | 21 | 0 |
| hsort/hsort3.j | C / A1 / C | 1.9014 | 1 | 1.906 | 48 | 0 | 21 | 27 | 0 |
| isort/code/isort1.j | D / B / B | 3.8611 | 1.5109 | 1.5109 | 12 | 6 | 0 | 6 | 0 |
| isort/code/isort2_old.j | C / A1 / C | 1.7077 | 1 | 1.7344 | 1 | 0 | 1 | 0 | 0 |
| isort/code/isort3_old.j | C / A1 / C | 1.7077 | 1 | 1.7344 | 1 | 0 | 1 | 0 | 0 |
| isort/disort.j | C / C / C | 1.7122 | 1.6081 | 1.6081 | 6 | 5 | 0 | 1 | 0 |
| isort/isort10.j | C / B / B | 1.9279 | 1.0751 | 1.0751 | 4 | 1 | 0 | 3 | 0 |
| isort/isort6.j | C / A1 / C | 1.8171 | 1 | 1.8395 | 1 | 0 | 1 | 0 | 0 |
| isort/isort9.j | C / A1 / A1 | 1.9245 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| linear_search/Lsearch1.j | C / C / C | 1.8904 | 1.9167 | 1.9167 | 1 | 1 | 0 | 0 | 0 |
| linear_search/Lsearch2.j | C / C / C | 1.9028 | 1.9296 | 1.9296 | 1 | 1 | 0 | 0 | 0 |
| linear_search/list_sentinel1.j | C / C / C | 1.8333 | 1.8723 | 1.8723 | 1 | 1 | 0 | 0 | 0 |
| linear_search/list_sentinel2.j | C / C / C | 1.7812 | 1.8387 | 1.8387 | 1 | 1 | 0 | 0 | 0 |
| linear_search/list_sentinel3.j | C / C / C | 1.631 | 1.6506 | 1.6506 | 1 | 1 | 0 | 0 | 0 |
| linear_search/list_sentinel4.j | B / B / B | 1.3333 | 1.3636 | 1.3636 | 1 | 1 | 0 | 0 | 0 |
| linear_search/list_sentinel5.j | C / C / C | 1.6463 | 1.6667 | 1.6667 | 1 | 1 | 0 | 0 | 0 |
| linear_search/sentinel1.j | C / C / C | 1.75 | 1.8065 | 1.8065 | 1 | 1 | 0 | 0 | 0 |
| linear_search/sentinel2.j | C / C / C | 1.8571 | 1.8958 | 1.8958 | 1 | 1 | 0 | 0 | 0 |
| linear_search/sentinel3.j | C / C / C | 1.6628 | 1.6824 | 1.6824 | 1 | 1 | 0 | 0 | 0 |
| minheap/minheap_GPC2018.j | B / A1 / B | 1.0811 | 1 | 1.1111 | 3 | 0 | 2 | 1 | 0 |
| minheap/minheap_Nohr2015.j | B / B / B | 1.0801 | 1.1132 | 1.1569 | 35 | 24 | 8 | 3 | 0 |
| minheap/minheap_subset.j | B / B / B | 1.0806 | 1.114 | 1.158 | 35 | 24 | 8 | 3 | 0 |
| msort/msort1.j | C / A1 / C | 1.952 | 1 | 1.957 | 22 | 0 | 14 | 8 | 0 |
| mst/prim.j | B / B / B | 1.3588 | 1.3791 | 1.3791 | 5 | 5 | 0 | 0 | 0 |
| others/code/decfac2rank.j | E / A1 / A1 | 18 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| others/code/lisort.j | B / A1 / A1 | 1.443 | 1 | 1 | 5 | 0 | 0 | 5 | 0 |
| others/examples/binarytree2e.j | B / B / B | 1.3392 | 1.0922 | 1.2032 | 24 | 8 | 8 | 8 | 0 |
| others/examples/breadth_first_search.j | D / B / D | 2.2439 | 1.5751 | 2.1641 | 81 | 13 | 59 | 42 | 2 |
| others/examples/fact.j | D / D / D | 7.2564 | 8.3235 | 8.3235 | 31 | 31 | 0 | 0 | 0 |
| others/examples/invtab2rank.j | B / A1 / A1 | 1.5 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| others/examples/linkedlist5e.j | C / B / B | 1.6505 | 1.0268 | 1.535 | 27 | 2 | 13 | 14 | 0 |
| others/examples/shell_sort1.j | C / A1 / C | 1.8655 | 1 | 1.8765 | 1 | 0 | 1 | 0 | 0 |
| perm/code/perm2decfac.j | D / B / B | 2.9631 | 1.4317 | 1.4317 | 23 | 22 | 0 | 1 | 0 |
| qsort/code/qsort.j | C / B / B | 1.8971 | 1.2772 | 1.2772 | 10 | 5 | 0 | 5 | 0 |
| qsort/code/qsort1.j | C / B / B | 1.8927 | 1.2763 | 1.2763 | 10 | 5 | 0 | 5 | 0 |
| qsort/code/qsort2.j | B / B / B | 1.2792 | 1.3022 | 1.3022 | 5 | 5 | 0 | 0 | 0 |
| qsort/qsort2.j | B / B / B | 1.2685 | 1.2843 | 1.2843 | 5 | 5 | 0 | 0 | 0 |
| qsort/queue5e.j | C / A1 / D | 1.8406 | 1 | 2.2952 | 24 | 0 | 18 | 10 | 0 |
| rank/rank.j | D / C / D | 2.7586 | 1.7636 | 2.2085 | 124 | 100 | 8 | 20 | 0 |
| rank/rank_lexicographic.j | E / B / B | 72.2 | 1.4325 | 1.4325 | 24 | 22 | 0 | 2 | 0 |
| rank/rank_perm.j | B / B / B | 1.278 | 1.3232 | 1.3232 | 7 | 7 | 0 | 0 | 0 |
| ssort/code/ssort.j | B / A1 / C | 1.5677 | 1 | 1.6309 | 6 | 0 | 6 | 0 | 0 |
| ssort/code/ssort1.j | E / B / B | 16.4118 | 1.5081 | 1.5081 | 7 | 6 | 0 | 1 | 0 |
| ssort/ssort1.j | B / A1 / B | 1.5146 | 1 | 1.5697 | 6 | 0 | 6 | 0 | 0 |
| treesort/treesort.j | A1 / A1 / B | 1 | 1 | 1.0144 | 0 | 0 | 1 | 0 | 0 |
