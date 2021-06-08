# Deep Homology-Based Protein Contact-Map Prediction

We present here Periscope, our method for homology modelling using a new method for contact map prediction.

## Requirements
The following programs should be installed and added to PATH environment variable:
- HHBlits (3.0.3)
- Evfold
- CCMpred  
- Clustal Omega (1.2.4)

Additionaly all packages in the `requirements.txt` file should be installed

# Interface

## Predictions
To predict a contact map for protein `5bu3D` using our method you need to clone to this repository and use:
```python
python3 -m periscope.tools.predict prscope_bins_no_templates "5bu3D" -o  "5bu3D.csv"
```
The file `5bu3D.csv` will contain the predicted contact probabilities.

## Training
To train a new model (named MODEL_NAME) use
```python
python3 -m periscope.tools.trainer -n MODEL_NAME -t 
```
You can also specify the hyperparameters through the `params.yaml` file on this repository.  
The command above will train a model called "MODEL_NAME" over our training dataset (8905 proteins) for `20` epoches with learning rate `0.0001`. The model would be saved in the models folder on this repository.  
Please note that training the model over the full training set would require generating the data which may take a long time and requires a lot of memory.  

For any questions please feel free to contact `omer.ronen@mail.huji.ac.il`
