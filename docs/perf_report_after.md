# Face-Local teljesítmény-benchmark

Szintetikus adatbázisok, service-szintű mérések (a UI-szálon futó munka pontos mása). Idők ms-ban.

| skála | 100p / 1000img / 2000f (9 MB) | 500p / 5000img / 10000f (42 MB) | 1000p / 10000img / 20000f (84 MB) | 5000p / 50000img / 100000f (420 MB) |
|---|---|---|---|---|
| művelet | 100p_1k | 500p_5k | 1000p_10k | 5000p_50k |
|---|---|---|---|---|
| startup: init_db + migrations | 5 | 6 | 7 | 41 |
| sidebar refresh (old: full ORM graph) | 39 | 272 | 650 | 8898 |
| sidebar refresh (old + blobs deferred) | 43 | 290 | 610 | 8001 |
| sidebar refresh (new: aggregate queries) | 2 | 8 | 18 | 100 |
| persons tab list | 5 | 17 | 37 | 229 |
| search (name filter) | 3 | 9 | 17 | 82 |
| person dialog open (DB work) | 4 | 3 | 3 | 5 |
| settings open (DB counts) | 4 | 5 | 4 | 17 |
| JSON export | 75 | 322 | 730 | 6538 |
| CSV export | 42 | 279 | 570 | 6305 |
| DB backup (sqlite backup API) | 38 | 134 | 275 | 1858 |
| single person edit save | 2 | 2 | 2 | 17 |
