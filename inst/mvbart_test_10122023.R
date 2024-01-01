# install.packages("devtools")
# devtools::install_github("MateusMaiaDS/mvnbart2")

library(mvnbart)

# simulate data ####

# true regression functions
f_true_C <- function(X){
  10 * sin(2 * pi * X)
}

f_true_Q <- function(X){
  as.numeric(
    10 * cos(2 * pi * X)
  )
}

# true covariance matrix for residuals
sigma_c <- 5
sigma_q <- 1
rho <- -0.5
scale_c <- 10
scale_q <- 0.1
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 200

data_train <- data.frame(X = runif(N, 0, 1))

data_train$C <- NA
data_train$EC <- NA
data_train$Q <- NA
data_train$EQ <- NA

for (i in 1:N){
  resid <- Sigma_chol %*% rnorm(2)
  data_train$EC[i] <- f_true_C(data_train$X[i]) * scale_c
  data_train$C[i] <- (f_true_C(data_train$X[i]) + resid[1]) * scale_c
  data_train$EQ[i] <- f_true_Q(data_train$X[i]) * scale_q
  data_train$Q[i] <- (f_true_Q(data_train$X[i]) + resid[2]) * scale_q
}

# mvnbart
mvnBart_wrapper <- function(x_train, x_test, c_train, q_train){
  mvnBart_fit <- mvnbart::mvnbart(x_train = x_train,
                                  c_train = c_train, q_train = q_train,
                                  x_test = x_test, scale_bool = FALSE,
                                  n_tree = 100,
                                  n_mcmc = 5000,
                                  n_burn = 1000)
  
  sigma_c <- mvnBart_fit$tau_c_post^(-1/2) 
  sigma_q <- mvnBart_fit$tau_q_post^(-1/2)  
  rho = mvnBart_fit$rho_post
  
  return(
    list(
      c_hat_test = mvnBart_fit$c_hat_test,
      q_hat_test = mvnBart_fit$q_hat_test,
      sigma_c = mvnBart_fit$tau_c_post^(-1/2),
      sigma_q = mvnBart_fit$tau_q_post^(-1/2),
      rho = mvnBart_fit$rho_post
    )
  )
}

mvbart_mod_old <- mvnBart_wrapper(data.frame(X = data_train$X), data.frame(X = data_train$X), data_train$C, data_train$Q)

plot(data_train$X, apply(mvbart_mod_old$c_hat_test, 1, mean))
points(data_train$X, data_train$EC, col = "red")

plot(data_train$X, apply(mvbart_mod_old$q_hat_test, 1, mean))
points(data_train$X, data_train$EQ, col = "red")
