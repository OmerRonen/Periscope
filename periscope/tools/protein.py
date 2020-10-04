from periscope.utils.protein import Protein

protein, chain = '2mx7', 'A'
self = Protein(protein, chain)

print(self.dm)
print(self.dm.shape)
print(self.str_seq)