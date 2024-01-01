library(BART)
# library(mvnbart3)
rm(list=ls())
devtools::load_all()
# simulate data ####
set.seed(42)
# true regression function for E[Z|X]
f_true <- function(X){
        as.numeric(-2 + sin(pi*X[1]*X[2]) + 2*(X[3] - 0.5)^2 + X[4] + X[5])
        # c(0)
}

# simulation parameters
n <- 1000
p <- 10

# true correlation matrix for Z
rho <- 0.3
Sigma <- matrix(c(1,rho,rho,1), nrow = 2)
Sigma_chol <- t(chol(Sigma))



# simulate data
X_train <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
X_train <- as.data.frame(X_train)
Y_train <- data.frame(index = 1:n)
Y_train$Z1 <- NA
Y_train$EZ1 <- NA
Y_train$Y1 <- NA
Y_train$P1 <- NA
Y_train$Z2 <- NA
Y_train$EZ2 <- NA
Y_train$Y2 <- NA
Y_train$P2 <- NA

for (i in 1:n){
        resid <- Sigma_chol %*% rnorm(2)
        Y_train$EZ1[i] <- f_true(X_train[i,1:5])
        Y_train$Z1[i] <- f_true(X_train[i,1:5]) + resid[1]
        Y_train$Y1[i] <- (Y_train$Z1[i] > 0)
        Y_train$P1[i] <- pnorm(Y_train$EZ1[i])
        Y_train$EZ2[i] <- f_true(X_train[i,6:10])
        Y_train$Z2[i] <- f_true(X_train[i,6:10]) + resid[2]
        Y_train$Y2[i] <- (Y_train$Z2[i] > 0)
        Y_train$Y2[i] <- (Y_train$Z2[i] > 0)
        Y_train$P2[i] <- pnorm(Y_train$EZ2[i])
}

hist(Y_train$P1)

X_test <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
X_test <- as.data.frame(X_test)
Y_test <- data.frame(index = 1:n)
Y_test$Z1 <- NA
Y_test$EZ1 <- NA
Y_test$Y1 <- NA
Y_test$P1 <- NA
Y_test$Z2 <- NA
Y_test$EZ2 <- NA
Y_test$Y2 <- NA
Y_test$P2 <- NA

for (i in 1:n){
        resid <- Sigma_chol %*% rnorm(2)
        Y_test$EZ1[i] <- f_true(X_test[i,1:5])
        Y_test$Z1[i] <- f_true(X_test[i,1:5]) + resid[1]
        Y_test$Y1[i] <- (Y_test$Z1[i] > 0)
        Y_test$P1[i] <- pnorm(Y_test$EZ1[i])
        Y_test$EZ2[i] <- f_true(X_test[i,6:10])
        Y_test$Z2[i] <- f_true(X_test[i,6:10]) + resid[2]
        Y_test$Y2[i] <- (Y_test$Z2[i] > 0)
        Y_test$Y2[i] <- (Y_test$Z2[i] > 0)
        Y_test$P2[i] <- pnorm(Y_test$EZ2[i])
}

# Getting y_mat element


x_train <- X_train[,1:10]
x_test <- X_test[,1:10]
y_train <- cbind(as.numeric(Y_train$Y1),as.numeric(Y_train$Y2))
colnames(y_train) <- c("C","Q")
table(y_train[,1])
table(y_train[,2])

bart_mod <- mvnbart4(x_train = x_train,y_mat = y_train,
                     Sigma_init = diag(nrow = NCOL(y_train)),
                      n_mcmc = 2000,n_burn = 0,df = 2,
                      x_test = x_test,n_tree = 50,
                      node_min_size = 2,m = n,update_Sigma = TRUE)

bart_mod$y_hat[1,1,] %>% hist
bart_mod$sigmas_post %>% plot(type = "l")
bart_mod$sigmas_post %>% mean
