## =============== random initialization ===============
randInitializeWeights <- function(L_in, L_out) {
  theta <- matrix(0, L_out, 1 + L_in);
  epsilon_init <- 0.12;
  theta <- matrix(rnorm(L_out*(L_in + 1)),
              L_out, (L_in + 1))*2*epsilon_init - epsilon_init;
  return(theta);
}
## =============== sigmoid function ===============
sigmoid <- function(z) {
  g <- 1 / (1 + exp(-z));
  return(g);
}
## =============== sigmoid gradient function ===============
sigmoidGradient <- function(z) {
  g <- matrix(0, dim(z));
  g <- sigmoid(z);
  g <- g * (1 -g);
  return(g);
}
## =============== cost J and gradient ===============
nnCostFunction <- function(nn_params,
                           input_layer_size,
                           hidden_layer_size,
                           num_labels,
                           X, y, lambda) {
  Theta1 <- matrix(nn_params[1:(hidden_layer_size *
                   (input_layer_size + 1))],
                   hidden_layer_size,
                   (input_layer_size + 1));
  Theta2 <- matrix(nn_params[-1:-(hidden_layer_size *
                   (input_layer_size + 1))],
                   num_labels,
                   (hidden_layer_size + 1));
  m <- dim(X)[1];
  J <- 0;
  Theta1_grad <- matrix(0, nrow(Theta1), ncol(Theta1));
  Theta2_grad <- matrix(0, nrow(Theta2), ncol(Theta2));

  for (i in 1:m) {
    y_label <- rep(0, num_labels);
    y_label[y[i] == as.integer(levels(factor(y)))] <- 1;
  ## forward propagation
    # first feed
    a1 <- X[i, ];
    a1 <- c(1, a1);
    # first hidden layer
    z2 <- Theta1 %*% a1;
    a2 <- sigmoid(z2);
    a2 <- c(1, a2);
    # output layer
    z3 <- Theta2 %*% a2;
    a3 <- sigmoid(z3);
    # add to cost function
    J <- J + sum(-y_label * log(a3) - (1 - y_label) * log(1 - a3));
  ## backward propagation
    delta3 <- a3 - y_label;
    delta2 <- (t(Theta2) %*% delta3)[-1] * sigmoidGradient(z2);

    Theta1_grad <- Theta1_grad + (delta2 %*% a1);
    Theta2_grad <- Theta2_grad + (delta3 %*% a2);    
  }
  
  J <- J / m;
  Theta1_grad <- Theta1_grad / m;
  Theta2_grad <- Theta2_grad / m;

  # J regulization
  reg_theta1 <- Theta1[, -1];
  reg_theta2 <- Theta2[, -1];
  J <- J + (lambda/(2*m)) * (sum(reg_theta1^2) + sum(reg_theta2^2));
  # gradient regulization
  Theta1_grad[, -1] <- Theta1_grad[, -1] + (lambda/m) * Theta1[, -1];
  Theta2_grad[, -1] <- Theta2_grad[, -1] + (lambda/m) * Theta2[, -1];

  # unroll gradients
  grad <- c(c(Theta1_grad), c(Theta2_grad));

  return(list(J = J, grad = grad));
}
## =============== cost J function for optimization ===============
costFunction <- function(nn_params,
                           input_layer_size,
                           hidden_layer_size,
                           num_labels,
                           X, y, lambda) {
  costJ <- nnCostFunction(nn_params = nn_params,
                          input_layer_size = input_layer_size,
                          hidden_layer_size = hidden_layer_size,
                          num_labels = num_labels,
                          X = X, y = y, lambda = lambda)$J;
  return(costJ);
}
## ========== cost J gradient function for optimization ==========
gradFunction <- function(nn_params,
                           input_layer_size,
                           hidden_layer_size,
                           num_labels,
                           X, y, lambda) {
  grad <- nnCostFunction(nn_params = nn_params,
                          input_layer_size = input_layer_size,
                          hidden_layer_size = hidden_layer_size,
                          num_labels = num_labels,
                          X = X, y = y, lambda = lambda)$grad;
  return(grad);
}
#################### end of utility functions ####################

## ================ train nerual network ===============
nnet_train <- function(X, y, hidden_layer_size = 25,
                 lambda = 1, maxit = 50) {
  m <- nrow(X);
  input_layer_size <- ncol(X);
  num_labels <- length(levels(factor(y)));
  # ================ Initializing Pameters ================
  initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels);
  initial_nn_params = c(c(initial_Theta1), c(initial_Theta2));

  # =================== Training NN ===================
  train_results <- optim(par = initial_nn_params,
                       fn = costFunction,
                       input_layer_size = input_layer_size,
                       hidden_layer_size = hidden_layer_size,
                       num_labels = num_labels,
                       X = X, y = y, lambda = lambda,
                       gr = gradFunction,
                       method = "L-BFGS-B",
                       control = list(maxit = maxit, trace = TRUE, REPORT = 1));
  nn_params <- train_results$par;
  Theta1 <- matrix(nn_params[1:(hidden_layer_size*(input_layer_size+1))],
                   hidden_layer_size, (input_layer_size + 1));
  Theta2 <- matrix(nn_params[-1:-(hidden_layer_size*(input_layer_size+1))],
                   num_labels, (hidden_layer_size + 1));
  ## ================= show accuracy =================
  pred = nnet_predict(Theta1, Theta2, X);
  cat("\nTraining Set Accuracy: ", sum(pred == y)/length(pred)*100,
      "%\n", sep = "");
  ## =============== return thetas ===============
  return(list(Theta1 = Theta1, Theta2 = Theta2));
}

## =============== nerual network predict ===============
nnet_predict <- function(Theta1, Theta2, X) {
  m <- nrow(X);
  num_labels <- nrow(Theta2);
  p = rep(0, m);

  h1 <- sigmoid(cbind(rep(1, m), X) %*% t(Theta1));
  h2 <- sigmoid(cbind(rep(1, m), h1) %*% t(Theta2));
  p = apply(h2, 1, which.max);
  return(p);
}

################ example ###############
