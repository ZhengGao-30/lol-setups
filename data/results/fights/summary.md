# Fight laws (V layer)

Series-block bootstrap 95% intervals.

model | counts | param | est | lo | hi | n
--- | --- | --- | --- | --- | --- | ---
S7 next death | tracked + V5 participants | c | -0.945 | -1.162 | -0.715 | 3877
S7 next death | tracked + V5 participants | b_n (square law -1, linear 0) | -2.327 | -2.700 | -1.959 | 3877
S7 next death | tracked + V5 participants | b_g | -4.652 | -5.901 | -3.430 | 3877
Tullock fight | tracked + V5 participants | gamma_n | 4.719 | 4.245 | 5.272 | 2139
Tullock fight | tracked + V5 participants | gamma_g | 11.276 | 9.636 | 13.097 | 2139
Tullock fight | tracked + V5 participants | gamma_side | -0.059 | -0.199 | 0.080 | 2139
Tullock fight | tracked + V5 participants | gamma_n / gamma_g | 0.419 | 0.359 | 0.499 | 2139
S7 next death | tracked only | c | -0.188 | -0.400 | 0.019 | 2683
S7 next death | tracked only | b_n (square law -1, linear 0) | -0.566 | -0.839 | -0.286 | 2683
S7 next death | tracked only | b_g | -7.034 | -8.138 | -6.002 | 2683
Tullock fight | tracked only | gamma_n | 0.795 | 0.630 | 0.952 | 1481
Tullock fight | tracked only | gamma_g | 10.936 | 9.205 | 12.830 | 1481
Tullock fight | tracked only | gamma_side | -0.029 | -0.154 | 0.093 | 1481
Tullock fight | tracked only | gamma_n / gamma_g | 0.073 | 0.054 | 0.095 | 1481

P(next death on the larger side) by local numbers (cells with n >= 30) vs the square-law value:

ratio | n | p_victim_larger | square_law
--- | --- | --- | ---
2v1 | 402 | 0.132 | 0.333
3v1 | 447 | 0.043 | 0.250
3v2 | 411 | 0.139 | 0.400
4v1 | 373 | 0.008 | 0.200
4v2 | 426 | 0.052 | 0.333
4v3 | 452 | 0.148 | 0.429
5v1 | 223 | 0.004 | 0.167
5v2 | 358 | 0.006 | 0.286
5v3 | 408 | 0.037 | 0.375
5v4 | 377 | 0.183 | 0.444
