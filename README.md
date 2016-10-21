# SPN-Z

### Python Sum-Prouct Network Library

SPN-Z is the first Open Source SPN library that is GPU-supported. It's not complete yet, but is already promising!

#### Dependencies:
- Numpy
- Tensorflow

## Why SPN-Z?
2 Words: Speed and Accuracy.
#### Speed:
How fast are we? We compared ourselves to the best published speed results we could find in this paper. On average we are 10x faster, and on bigger datasets even faster.

Raw Speed Comparisons on 6 Data Sets From the Paper

![lol](https://dl.dropboxusercontent.com/u/61478139/Screen%20Shot%202016-10-21%20at%204.46.29%20PM.png)

Normalized (Adam is set to 1, the rest are scared accordingly)

![lol](https://dl.dropboxusercontent.com/u/61478139/Screen%20Shot%202016-10-21%20at%204.46.36%20PM.png)

PGD and CCCP are from the Paper.
Adam is built on the same SPN with our library.

#### Accuracy:
Since this is built ontop of tensorflow, we have access to many optimizers for gradient descent. Furthermore you can customize minibatch size and visualize loss, validation loss, and testing loss live. 
Our gradient descent outperformed the paper's gradient descent on every occasion. The paper demonstrates that gradient descent is not viable for SPNs, we show that the use of parameterized learning rates says otherwise.

Log Loss using PGD (paper), Adam (SPN-Z), CCCP (paper)

![lol](https://dl.dropboxusercontent.com/u/61478139/Screen%20Shot%202016-10-21%20at%204.46.24%20PM.png)

Comparing PGD to Adam based on how far they are from CCCP

![lol](https://dl.dropboxusercontent.com/u/61478139/Screen%20Shot%202016-10-21%20at%204.46.32%20PM.png)

### How to use:
Will be updated when there is better functionality.
How to run an example from one of the datasets above.
```
from spn_lib.SPN2 import SPN
model = SPN()
model.make_model_from_file("spn_models/ntlcs.spncc.spn.txt")
model.start_session()
model.add_data("Data/nltcs.ts.data", "train")
model.add_data("Data/nltcs.valid.data", "valid")
model.add_data("Data/nltcs.test.data", "test")
model.train(50)
model.evaluate(model.data.test)
```
#### To Do Feature List:
Build Your Own SPN
Random Generate SPN
CCCP Optimizer
BMM Optimizer
Online Learning
Structural Learning
Continuous Variables
