# Long Short Term Memory Networks

The distinguishing feature of LSTM is Gated control mechanism which controls the flow of information from the previous state and the current input to the current state. 

## Forget Gate Layer 

Forget Gate Layer $\sigma_{f}$  looks at the current input $x_{t}$ and previous hidden layer $h_{t-1}$ and provides a value from 0 to 1, which control the information flow from previous cell state $C_{t-1}$.

$f_{t} = \sigma( W_{f}.(h_{t-1}, x_{t}) + b_{f})$

## Input Gate Layer

Input Gate Layer $\sigma_{i}$ decides which values we will update. A $tanh$ layer creates a vector of new candidate values to replace the old values (Think about adding new pronouns to replace old subject's pronoun). 

$i_{t} = \sigma( W_{i}.(h_{t-1}, x_{t}) + b_{i})$ <br />
$\tilde{C}_{t} = \tanh( W_{i}.(h_{t-1}, x_{t}) + b_{C})$

$\tilde{C}_{t}$ represents the update to the cell state. 

Now we lose the unnecessary information from the previous cell state $C_{t-1}$ by multiplying it with $f_{t}$ and the add the result with the new information : product of $\tilde{C}_{t}$ and $i_{t}$.

$C_{t} = f_{t}C_{t-1} + i_{t}\tilde{C}_{t}$





