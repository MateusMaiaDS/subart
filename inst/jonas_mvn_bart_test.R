# install.packages("devtools")
rm(list=ls())
# devtools::install_github("MateusMaiaDS/mvnbart3")
# devtools::load_all()

library(BART)
library(mvnbart3)

# simulate data ####

# true regression functions
f_true_C <- function(X){
  as.numeric(
    (cos(2*X[1]) + 2 * X[2]^2 * X[3])
  )
}

f_true_Q <- function(X){
  as.numeric(
    3 * X[1] * X[4]^3 + 2 * X[2]^2
  )
}

# true covariance matrix for residuals
sigma_c <- 1
sigma_q <- 1
rho <- 0.9
scale_c <- 0.001
scale_q <- 1
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 50

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
  data_train$EC[i] <- f_true_C(data_train[i,1:4]) * scale_c
  data_train$C[i] <- (f_true_C(data_train[i,1:4]) + resid[1]) * scale_c
  data_train$EQ[i] <- f_true_Q(data_train[i,1:4]) * scale_q
  data_train$Q[i] <- (f_true_Q(data_train[i,1:4]) + resid[2]) * scale_q
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
  data_test$C[i] <- (f_true_C(data_test[i,1:4]) + resid[1]) * scale_c
  data_test$EQ[i] <- f_true_Q(data_test[i,1:4])
  data_test$Q[i] <- (f_true_Q(data_test[i,1:4]) + resid[2]) * scale_q
}

x_train <- data_train[,1:4]
x_test <- x_train
y_mat <- as.matrix(data_train[,c(5,7)])
colnames(y_mat) <- c("C","Q")

mvbart_mod <- mvnbart3(x_train = x_train,
                                 y_mat = y_mat,
                                 x_test = x_test,
                                 n_tree = 100,
                                 n_mcmc = 2500,
                                 n_burn = 500,
                                 conditional_bool = TRUE)



pairs(
  data.frame(rho = mvbart_mod$Sigma_post[1,2,]/sqrt(mvbart_mod$Sigma_post[1,1,]*mvbart_mod$Sigma_post[2,2,]),
             sigma_c = sqrt(mvbart_mod$Sigma_post[1,1,]),
             sigma_q = sqrt(mvbart_mod$Sigma_post[2,2,])
             )
  )

plot(mvbart_mod$y_hat_mean[,1], data_train$EC)
plot(mvbart_mod$y_hat_mean[,2], data_train$EQ)

mean_sigma_c <- sqrt(mean(mvbart_mod$Sigma_post[1,1,]))
mean_sigma_q <- sqrt(mean(mvbart_mod$Sigma_post[2,2,]))
mean_rho <- mean(mvbart_mod$Sigma_post[1,2,]/(mean_sigma_c*mean_sigma_q))

mean_sigma_c
mean_sigma_q
mean_rho

plot(mvbart_mod$Sigma_post[1,1,], type = "l")
plot(mvbart_mod$Sigma_post[1,2,], type = "l")
plot(mvbart_mod$Sigma_post[2,2,])
plot(mvbart_mod$y_hat[1,1,])
