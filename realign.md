## Re-alignment method  
Our pipeline requires us to align the templates to our target protein so that we can feed distance matrices with similar dimensions for a technical standpoint.  
The basic idea was to take the homologous found in the msa (hhblits output) use SIFTS mapping to find the matching pdb structure if exist, then we find the overlapping part between the pdb sequence and uniprot sequence and align the distance matrix accordingly.  
For example let's look at 2w1p, chain A:
![](data/figures/example_aln.png)  
What I missed was that the aligned reference is a partial sequence from the pdb sequence
![](data/figures/aln_2.png)  
In this case the reference would be excluded but nontheless this demonstrates the point and modeller used the same data.  

Looking at how our methods predictions compared with modeller it's easy to understand how the additional information would benefit our model:
![](data/figures/2w1pA_analysis.png?raw=true)  
Those contact on the bottom left of the modeller predictions were in fact in the reference:  
![](data/figures/cmap_2w2e.png?raw=true)  
What I did was to run clustalo on the pdb sequence vs the target sequence with the aligned uniprot sequence being the profile to make sure the resulting alignment is not different from hhblits.
This modification was only done to the test data, ideally I would like to retrain with this as well.


