from code.TRG import TRG

trg = TRG(4, 1)

tn = TRG(4, 1).initialize()

U, V = trg.svd(tn.tensors[0], orientation="left", truncate=2)

print(U.shape, V.shape)
