# Deep Homology-Based Protein Contact-Map Prediction

We present here Periscope, our method for homology modelling using a new method for contact map prediction.

## Requirements
The following programs should be installed and added to PATH environment variable:
- HHBlits (3.0.3)
- Evfold
- CCMpred  

Additionaly all packages in the `requiremetns.txt` file should be installed

# Interface
To predict a contact map for protein `5bu3D` using our method you need to clone to this repository
```python
python3 -m periscope.tools.predict nips_model "5bu3D" -o  "5bu3D.csv"
```