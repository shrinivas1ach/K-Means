import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

with open(r'C:\Users\SHRIRAM\Desktop\project\Iris.csv', 'r') as Irisfile:
    rows = csv.reader(Irisfile)
    for a in rows:
        print(a)
