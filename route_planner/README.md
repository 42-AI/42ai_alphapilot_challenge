## Navigation

### 1. Training

- **Models:**

The model architectures are implemented in the `code/models` folder. We managed to obtain satisfying results using an optimized version of the architecture detailed in `rnn.py`.
- **Training the models:**

We trained our best performing models using the talos package to optimize parameters for a custom GRU-based RNN using
`code/train.py`.

We also tried performing reinforcement learning using a Policy-Gradient method adapted for continuous outputs. We did not manage to obtain good results using this method, but you can try for yourself running using
`code/reinforcement_learning.py`

### 2. Run/Predict
You can run our model on the flightgoggles simulation using the `runner.py` file. 
