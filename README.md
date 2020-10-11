# Deep Homology-Based Protein Contact-Map Prediction

We present here Periscope, our method for homology modelling using a new method for contact map prediction.

## Requirements
The following programs should be installed and added to PATH environment variable:
- HHBlits (3.0.3)
- Evfold
- CCMpred  
- Clustal Omega (1.2.4)

Additionaly all packages in the `requiremetns.txt` file should be installed

# Interface

## Predictions
To predict a contact map for protein `5bu3D` using our method you need to clone to this repository and use:
```python
python3 -m periscope.tools.predict nips_model "5bu3D" -o  "5bu3D.csv"
```

## Training
To train a new model (named MODEL_NAME) use
```python
python3 -m periscope.tools.trainer -n MODEL_NAME -t 
```
You can also specify the hyperparameters through the `params.yaml` file on this repository
