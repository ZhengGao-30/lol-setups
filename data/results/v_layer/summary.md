# V-layer descriptive results (2-fps minimap tracks; every spawn; s-120..s+60 every 15 s)

Team-setups: 2994; games: 164; series: 60; bootstrap: 1000 series-block resamples; risk set after the spawn; cells with n < 30 are 'n/e'. Primary pit radius 1500.

Team-setups per objective group:

group | team-setups
--- | ---
ATAKHAN | 328
BARON_2+ | 118
BARON_NASHOR | 320
DRAGON | 322
DRAGON_2+ | 1204
ELDER | 52
HORDE | 322
RIFTHERALD | 328

Position sources (share of alive positions): detected / held (covered) / pinned (V5 minute):

group | q_det | q_hold | q_pin
--- | --- | --- | ---
ATAKHAN | 0.706 | 0.000 | 0.079
BARON_2+ | 0.681 | 0.027 | 0.000
BARON_NASHOR | 0.692 | 0.008 | 0.044
DRAGON | 0.704 | 0.000 | 0.216
DRAGON_2+ | 0.757 | 0.006 | 0.008
ELDER | 0.666 | 0.033 | 0.000
HORDE | 0.712 | 0.000 | 0.190
RIFTHERALD | 0.722 | 0.000 | 0.120

## 1-2. Commitment curves and trailing - leading contrasts

Headline cells (sign agrees at R in (1000, 1500, 2000), BH-FDR q < 0.05; shown at R = 1500):

group | offset_s | stat | trailing_minus_leading | lo | hi | q_bh
--- | --- | --- | --- | --- | --- | ---
DRAGON | 90 | P(self >= 3) | -0.096 | -0.170 | -0.022 | 0.025
DRAGON | 105 | mean k | -0.386 | -0.659 | -0.114 | 0.014
HORDE | 15 | P(k=0) | 0.262 | 0.069 | 0.467 | 0.043
HORDE | 15 | mean k | -0.361 | -0.636 | -0.117 | 0.034
RIFTHERALD | -75 | P(k=0) | 0.123 | 0.034 | 0.210 | 0.034
RIFTHERALD | -75 | mean k | -0.154 | -0.274 | -0.048 | 0.025
ATAKHAN | 15 | P(k=0) | 0.291 | 0.143 | 0.442 | 0.000
ATAKHAN | 15 | mean k | -0.456 | -0.753 | -0.169 | 0.014
ATAKHAN | 30 | P(k=0) | 0.312 | 0.147 | 0.462 | 0.000
ATAKHAN | 30 | mean k | -0.525 | -0.845 | -0.178 | 0.034
ATAKHAN | 45 | P(k=0) | 0.304 | 0.150 | 0.465 | 0.000
ATAKHAN | 45 | mean k | -0.667 | -1.017 | -0.324 | 0.000
ATAKHAN | 75 | P(k=0) | 0.291 | 0.120 | 0.456 | 0.000
ATAKHAN | 90 | P(k=0) | 0.320 | 0.152 | 0.462 | 0.000
ATAKHAN | 90 | mean k | -0.660 | -0.980 | -0.341 | 0.000
ATAKHAN | 105 | P(k=0) | 0.372 | 0.200 | 0.540 | 0.000
ATAKHAN | 105 | mean k | -0.837 | -1.278 | -0.432 | 0.000
BARON_NASHOR | 30 | P(k=0) | 0.333 | 0.181 | 0.493 | 0.000
BARON_NASHOR | 30 | mean k | -0.538 | -0.926 | -0.205 | 0.000
BARON_NASHOR | 45 | P(k=0) | 0.348 | 0.187 | 0.500 | 0.000
BARON_NASHOR | 45 | mean k | -0.507 | -0.790 | -0.206 | 0.000
BARON_NASHOR | 60 | P(k=0) | 0.213 | 0.113 | 0.323 | 0.000
BARON_NASHOR | 60 | mean k | -0.377 | -0.580 | -0.161 | 0.014
BARON_NASHOR | 75 | P(k=0) | 0.271 | 0.105 | 0.415 | 0.014
BARON_NASHOR | 90 | P(k=0) | 0.273 | 0.127 | 0.404 | 0.034
BARON_NASHOR | 90 | mean k | -0.473 | -0.695 | -0.212 | 0.000
BARON_NASHOR | 105 | P(k=0) | 0.300 | 0.106 | 0.480 | 0.025
BARON_NASHOR | 105 | mean k | -0.560 | -0.907 | -0.222 | 0.000
BARON_NASHOR | 120 | P(k=0) | 0.273 | 0.116 | 0.442 | 0.000
BARON_NASHOR | 120 | mean k | -0.591 | -1.067 | -0.204 | 0.014
BARON_NASHOR | 135 | P(k=0) | 0.316 | 0.132 | 0.500 | 0.000
BARON_NASHOR | 135 | mean k | -0.632 | -1.030 | -0.216 | 0.025
BARON_NASHOR | 150 | P(k=0) | 0.344 | 0.147 | 0.536 | 0.000
BARON_NASHOR | 150 | mean k | -0.625 | -1.031 | -0.188 | 0.025
DRAGON_2+ | -60 | mean k | -0.074 | -0.132 | -0.018 | 0.043
DRAGON_2+ | 0 | P(k=0) | 0.222 | 0.157 | 0.286 | 0.000
DRAGON_2+ | 0 | mean k | -0.369 | -0.490 | -0.254 | 0.000
DRAGON_2+ | 15 | P(k=0) | 0.337 | 0.249 | 0.415 | 0.000
DRAGON_2+ | 15 | mean k | -0.682 | -0.848 | -0.498 | 0.000
DRAGON_2+ | 15 | P(self >= 3) | -0.109 | -0.157 | -0.055 | 0.000
DRAGON_2+ | 30 | P(k=0) | 0.282 | 0.188 | 0.372 | 0.000
DRAGON_2+ | 30 | mean k | -0.586 | -0.758 | -0.401 | 0.000
DRAGON_2+ | 30 | P(self >= 3) | -0.092 | -0.152 | -0.034 | 0.000
DRAGON_2+ | 45 | P(k=0) | 0.345 | 0.229 | 0.446 | 0.000
DRAGON_2+ | 45 | mean k | -0.706 | -0.905 | -0.504 | 0.000
DRAGON_2+ | 45 | P(self >= 3) | -0.092 | -0.150 | -0.042 | 0.014
DRAGON_2+ | 60 | P(k=0) | 0.247 | 0.108 | 0.389 | 0.000
DRAGON_2+ | 60 | mean k | -0.534 | -0.827 | -0.237 | 0.000
DRAGON_2+ | 75 | P(k=0) | 0.350 | 0.190 | 0.515 | 0.014
DRAGON_2+ | 75 | mean k | -0.750 | -1.054 | -0.421 | 0.000
BARON_2+ | -45 | P(k=0) | 0.175 | 0.073 | 0.289 | 0.000
BARON_2+ | -45 | mean k | -0.275 | -0.455 | -0.119 | 0.000
BARON_2+ | 15 | P(k=0) | 0.364 | 0.154 | 0.595 | 0.000

Sensitivity without mostly-held/interpolated snapshots (q_hold + q_interp > 0.5 dropped; 5151 of 37082 rows): 104 of 166 headline cells remain headlines; sign kept in 135.

## 3. Arrival race: first team with >= 3 at the pit (R = 1500) vs taker within 180 s

group | n | race_winner_took | neither
--- | --- | --- | ---
ATAKHAN | 106 | 0.509 | 0.255
BARON_2+ | 33 | 0.636 | 0.242
BARON_NASHOR | 80 | 0.675 | 0.250
DRAGON | 36 | 0.806 | 0.111
DRAGON_2+ | 264 | 0.795 | 0.019
ELDER | 13 | 0.538 | 0.231
HORDE | 16 | 0.750 | 0.000
RIFTHERALD | 62 | 0.774 | 0.048
