# Face-Local teljesítmény-benchmark

Szintetikus adatbázisok, service-szintű mérések (a UI-szálon futó munka pontos mása). Idők ms-ban.

| skála | 100p / 1000img / 2000f (9 MB) | 500p / 5000img / 10000f (43 MB) | 1000p / 10000img / 20000f (86 MB) | 5000p / 50000img / 100000f (431 MB) |
|---|---|---|---|---|
| művelet | 100p_1k | 500p_5k | 1000p_10k | 5000p_50k |
|---|---|---|---|---|
| startup: init_db + migrations | 8 | 27 | 16 | 55 |
| sidebar refresh (old: full ORM graph) | 61 | 413 | 742 | 4071 |
| sidebar refresh (new: aggregate queries) | 2 | 10 | 19 | 90 |
| persons tab list | 6 | 23 | 37 | 171 |
| search (name filter) | 4 | 12 | 18 | 83 |
| person dialog open (DB work) | 4 | 5 | 5 | 6 |
| settings open (DB counts) | 5 | 5 | 4 | 9 |
| JSON export | 199 | 243 | 272 | 1354 |
| CSV export | 69 | 136 | 246 | 952 |
| DB backup (sqlite backup API) | 98 | 380 | 963 | 3501 |
| single person edit save | 10 | 4 | 4 | 7 |
