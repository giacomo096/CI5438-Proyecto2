import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():

    #Data import
    df = pd.read_csv('../iris.csv')

    inputs = df.drop('species', axis=1)
    outputs = df.species

    print(inputs)
    print(outputs)
 
main()