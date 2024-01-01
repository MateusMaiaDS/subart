library(BART)
# library(mvnbart3)
rm(list=ls())
load_all()
# simulate data ####

# true regression functions
f_true_C <- function(X){
     as.numeric(
          5*(cos(2*X[1]))
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
rho <- 0.5
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 250

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

y_mat <- cbind(data_train$C,data_train$Q)

x_train <- data_train[,1:4]
x_test <- data_test[,1:4]
colnames(y_mat) <- c("C","Q")
#
mvbart_mod <- mvnbart3(x_train = x_train,
                   y_mat = y_mat,
                   x_test = x_test,
                   n_tree = 50,
                   n_mcmc = 2500,df = 3,
                   n_burn = 500,Sigma_init = Sigma,
                   update_Sigma =  TRUE)

# load_all("/Users/mateusmaia/Documents/mvnbart2")
# mvbart_mod2 <- mvnbart(x_train = x_train,
#                        c_train = y_mat[,1],q_train = y_mat[,2],scale_bool = FALSE,
#                        x_test = x_test,
#                        n_tree = 200,
#                        n_mcmc = 2500,df = 3,
#                        n_burn = 500)

par(mfrow = c(3,1))
plot(sqrt(mvbart_mod$Sigma_post[1,1,]),type = "l", ylab = expression(sigma[c]), main =expression(sigma[c]))
plot(sqrt(mvbart_mod$Sigma_post[2,2,]),type = "l", ylab = expression(sigma[c]), main =expression(sigma[c]))
plot(mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,])), type = "l",
     ylab = expression(rho),main =expression(rho))

sqrt(mvbart_mod$Sigma_post_mean[1,1])
sqrt(mvbart_mod$Sigma_post_mean[2,2])
mvbart_mod$Sigma_post_mean[1,2]/(sqrt(mvbart_mod$Sigma_post_mean[1,1])*mvbart_mod$Sigma_post_mean[2,2])

# Plotting the model
par(mfrow = c(2,2))
plot(x_train$X1,y_mat[,1],main = expression(C))
points(x_train$X1,mvbart_mod$y_hat_mean[,1], pch = 20)
plot(x_train$X1,y_mat[,2], main = expression(Q))
points(x_train$X1,mvbart_mod$y_hat_mean[,2], pch = 20)

# Comparing with the BART predictions
c_bart <- dbarts::bart(x.train = x_train,y.train = y_mat[,1],x.test = x_test,ntree = 1)
q_bart <- dbarts::bart(x.train = x_train,y.train = y_mat[,2],x.test = x_test,ntree = 1)

plot(x_train$X1,y_mat[,1], main = expression(C))
points(x_train$X1,c_bart$yhat.train.mean, pch = 20)
# points(x_train$X1, cos(2*x_train$X1), col = "blue")

plot(x_train$X1,y_mat[,2], main = expression(Q))
points(x_train$X1,q_bart$yhat.train.mean, pch = 20)
# points(x_train$X1, (3*x_train$X1), col = "blue")


# Extra experiments for mvBART2
# plot(x_train$X1,y_mat[,1], main = expression(C))
# lines(x_train$X1,mvbart_mod2$c_hat %>% rowMeans(), pch = 20)
# lines(x_train$X1, cos(2*x_train$X1), col = "blue")
#
# plot(x_train$X1,y_mat[,2], main = expression(Q))
# lines(x_train$X1,mvbart_mod2$q_hat %>% rowMeans(), pch = 20)
# lines(x_train$X1, 3*x_train$X1, col = "blue")


sqrt(mvbart_mod$Sigma_post_mean[1,1])
sqrt(mvbart_mod$Sigma_post_mean[2,2])
mvbart_mod$Sigma_post_mean[1,2]/(sqrt(mvbart_mod$Sigma_post_mean[1,1])*mvbart_mod$Sigma_post_mean[2,2])
