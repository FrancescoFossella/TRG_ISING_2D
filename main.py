from code.TRG import TRG

import numpy as np

trg = TRG(2, 1, truncate=10)

Z = trg.solve()

print(Z)
