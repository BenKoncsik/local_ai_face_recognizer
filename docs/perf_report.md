# Face-Local teljesítmény-benchmark

Szintetikus adatbázisok, service-szintű mérések (a UI-szálon futó munka pontos mása). Idők ms-ban.

| skála | 100p / 1000img / 2000f (9 MB) | 500p / 5000img / 10000f (42 MB) | 1000p / 10000img / 20000f (83 MB) | 5000p / 50000img / 100000f (415 MB) |
|---|---|---|---|---|
| művelet | 100p_1k | 500p_5k | 1000p_10k | 5000p_50k |
|---|---|---|---|---|
| startup: init_db + migrations | 5 | 6 | 7 | 37 |
| sidebar refresh (current, blobs loaded) | 38 | 291 | 611 | 17560 |
| sidebar refresh (blobs deferred) | 38 | 259 | 618 | 13935 |
| persons tab list | 9 | 48 | 87 | 8959 |
| search (name filter) | 6 | 25 | 49 | 4474 |
| person dialog open (DB work) | 4 | 3 | 3 | 6 |
| settings open (DB counts) | 5 | 3 | 5 | 31 |
| JSON export | 289 | 1404 | 2855 | 19735 |
| CSV export | 284 | 1355 | 2778 | 22092 |
| DB backup (sqlite backup API) | 29 | 118 | 301 | 3197 |
| single person edit save | 2 | 1 | 1 | 8 |
