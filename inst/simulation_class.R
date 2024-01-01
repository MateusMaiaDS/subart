library(BART)
# library(mvnbart3)
rm(list=ls())
devtools::load_all()
library(purrr)
# simulate data ####
set.seed(42)
# true regression functions
f_true_C <- function(X){
     as.numeric(
          2*(cos(X[1]))  - 15*X[1]
     )
}

f_true_Q <- function(X){
     as.numeric(
          5*(3 * X[1]) + sin(pi*X[2])
     )
}

# true covariance matrix for residuals
sigma_c <- 1
sigma_q <- 1
rho <- 0.6
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 500

data_train <- data.frame(X1 = rep(NA, N))
data_train$X1 <- runif(N, -1, 1)
data_train$X2 <- runif(N, -1, 1)
data_train$X3 <- runif(N, -1, 1)
data_train$X4 <- runif(N, -1, 1)

data_train$C <- NA
data_train$EC <- NA
data_train$Q <- NA
data_train$EQ <- NA

for (i in 1:N){
     resid <- Sigma_chol %*% rnorm(2)
     data_train$EC[i] <- f_true_C(data_train[i,1:4])
     data_train$C[i] <- (f_true_C(data_train[i,1:4]) + resid[1]) * 1
     data_train$EQ[i] <- f_true_Q(data_train[i,1:4])
     data_train$Q[i] <- (f_true_Q(data_train[i,1:4]) + resid[2]) * 1
}

data_test <- data.frame(X1 = rep(NA, N))
data_test$X1 <- runif(N, -1, 1)
data_test$X2 <- runif(N, -1, 1)
data_test$X3 <- runif(N, -1, 1)
data_test$X4 <- runif(N, -1, 1)

data_test$C <- NA
data_test$EC <- NA
data_test$Q <- NA
data_test$EQ <- NA

for (i in 1:N){
     resid <- Sigma_chol %*% rnorm(2)
     data_test$EC[i] <- f_true_C(data_test[i,1:4])
     data_test$C[i] <- (f_true_C(data_test[i,1:4]) + resid[1]) * 1
     data_test$EQ[i] <- f_true_Q(data_test[i,1:4])
     data_test$Q[i] <- (f_true_Q(data_test[i,1:4]) + resid[2]) * 1
}

# Getting y_mat element
data_train <- dplyr::arrange(data_train,X1)
data_test <- dplyr::arrange(data_test,X1)
data_train$C <- ifelse(data_train$C>0,1,0)
data_train$EC <- ifelse(data_train$EC>0,1,0)
data_train$Q <- ifelse(data_train$Q>0,1,0)
data_train$EQ <- ifelse(data_train$EQ>0,1,0)

data_test$C <- ifelse(data_test$C>0,1,0)
data_test$EC <- ifelse(data_test$EC>0,1,0)
data_test$Q <- ifelse(data_test$Q>0,1,0)
data_test$EQ <- ifelse(data_test$EQ>0,1,0)

y_mat <- cbind(data_train$C,data_train$Q)

x_train <- data_train[,1:4]
x_test <- data_test[,1:4]
colnames(y_mat) <- c("C","Q")

bart_mod <- mvnbart4(x_train = x_train,y_mat = y_mat,Sigma_init = Sigma,
                     n_mcmc = 1000,n_burn = 0,df = 2,
                     x_test = x_test,n_tree = 50,
                     node_min_size = 5,m = 200,update_Sigma = FALSE)

# bart_mod$sigmas_post %>% plot(type = "l")
# table(bart_mod$y_hat_test_mean_class[,1],data_test$C)
# table(bart_mod$y_hat_test_mean_class[,2],data_test$Q)
bart_mod$sigmas_post %>% plot(type = "l")
