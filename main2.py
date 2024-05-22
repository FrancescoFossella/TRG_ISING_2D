from code.physical_analysis import analysis_H, analysis_U

import numpy as np

N = 10
betas = np.linspace(0.1, 1, 10)
truncation = 0.5

H = analysis_H(N, betas, truncation)
U = analysis_U(N, betas, truncation)

