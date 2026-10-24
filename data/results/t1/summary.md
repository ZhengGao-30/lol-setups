# T1 / Faker appendix (descriptive; series-block bootstrap 95% intervals)

## A3 Faker vs other starting mid laners (spawn-anchored pre-spawn snapshots s-60..s)

measure  flagged_mean  others_mean     diff       lo       hi  n_flagged  n_others
 in_pit         0.039        0.031    0.008   -0.000    0.018        432      2545
   rank         2.877        2.993   -0.116   -0.226   -0.027        432      2545
   dpit      5563.370     6032.727 -469.356 -709.075 -275.207        432      2545
  alone         0.631        0.622    0.009   -0.039    0.063        432      2545
   main         0.337        0.333    0.004   -0.042    0.049        432      2545
    deg         1.481        1.474    0.007   -0.023    0.041        432      2545
    btw         0.259        0.253    0.006   -0.011    0.023        432      2545

Faker setups: 432; all mid-lane setups: 2977.

## A5 behaviour by state

 group    state    n  commit_share  polarisation  mean_k
    T1 trailing   93         0.097         0.731   0.602
others trailing  652         0.107         0.724   0.598
    T1     even  211         0.256         0.460   1.308
others     even 1205         0.217         0.501   1.143
    T1  leading  117         0.291         0.274   1.573
others  leading  628         0.299         0.277   1.639

                              contrast  estimate     lo     hi
T1 trailing-minus-leading commit share    -0.194 -0.320 -0.064
                                others    -0.192 -0.236 -0.144
                       T1 minus others    -0.002 -0.133  0.149

## A8 T1 by event (same rules)

      event   n  commit_share  mean_k
   MSI 2025 197         0.203   1.142
Worlds 2025 224         0.254   1.299

## A6 fight-function shift (T1 outcome minus fitted Tullock probability)

 n_fights_with_T1  mean_residual     lo     hi
              399        -0.0024 -0.045 0.0456

## A7 Worlds 2025 final ledger

51 objective spawns across the five games; see A7_worlds2025_final_ledger.csv.

## A9 / A10

A9 (Worlds 2023/24, different objective rules): the control videos were not processed by the tracker in this version; not available. A4 (role effect u^M) is not estimated; A3 gives the descriptive position measures instead. A1/A2: see results/fingerprints.

| narrative | measure here | status |
|---|---|---|
| 'T1 does not panic when behind' | A5 trailing-minus-leading commitment contrast, T1 vs others | measurable |
| 'Faker controls the map' | A3 position measures (arrival rank, block membership, MST centrality) | measurable |
| 'T1 wins fights at equal numbers' | A6 residual of the fight function | measurable, descriptive |
| champion-pool depth, draft mind games, communication | no champion / audio data in this study | not measurable |