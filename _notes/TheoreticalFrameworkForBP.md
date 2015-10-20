# Hints

* Lagrangian function is the sum of two terms
1. Objective Function : Cost [ Squared Outpur Error ]
2. Constraint that defines the network dynamics

* B : Langrange Multiplier
* C : Squared Output Error function
* p : belongs to [1,P] examples
* k : belongs to [1,N] layers
* X(k) : state of kth layer
* F(k) = F( W(k).X(k-1) ) : Activation function of kth layer
* W(k) : Weight matrix connecting (k-1)th layer to kth layer
* X(0) = I (Inputer layer)
* X(N) : Output Layer

* Differentiating LF and equating to zero should provide the local minima of the cost function while satifying the constraints of the network. Diff w.r.t. X,W,B gives rise to 3 conditions.

1. Forward dynamics
2. Backward dnyamics
3. Weight Update rule
