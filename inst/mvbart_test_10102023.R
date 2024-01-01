# install.packages("devtools")
# devtools::install_github("MateusMaiaDS/mvnbart3", force = TRUE)
rm(list=ls())
library(ggplot2)
library(bayesplot)
library(mvnbart3)
# devtools::load_all()
# simulate data ####

# true regression functions
f_true_C <- function(X){
  as.numeric(
    (10*sin(pi*X[1]*X[2]))
  )
}

f_true_Q <- function(X){
  as.numeric(
    (10*sin(pi*X[1]))
  )
}

# true covariance matrix for residuals
sigma_c <- 1
sigma_q <- 1
rho <- -0.5
scale_c <- 1
scale_q <- 1
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 250

data_train <- data.frame(X1 = runif(N, 0, 1),
                         X2 = runif(N, 0, 1))

data_train$C <- NA
data_train$EC <- NA
data_train$Q <- NA
data_train$EQ <- NA

for (i in 1:N){
  resid <- Sigma_chol %*% rnorm(2)
  data_train$EC[i] <- f_true_C(data_train[i,1:2]) * scale_c
  data_train$C[i] <- (f_true_C(data_train[i,1:2]) + resid[1]) * scale_c
  data_train$EQ[i] <- f_true_Q(data_train[i,1:2]) * scale_q
  data_train$Q[i] <- (f_true_Q(data_train[i,1:2]) + resid[2]) * scale_q
}


x_train <- data.frame(X1 = data_train$X1,
                      X2 = data_train$X2)
x_test <- data.frame(X1 = runif(1000, 0, 1),
                     X2 = runif(1000, 0, 1))
y_mat <- as.matrix(data_train[,c("C","Q")])
colnames(y_mat) <- c("C","Q")

mvbart_mod <- mvnbart3::mvnbart3(x_train = x_train,
                                 y_mat = y_mat,
                                 x_test = x_test,
                                 n_tree = 200,
                                 n_mcmc = 5000,
                                 n_burn = 1000)


# check inference for standard deviations and correlation
df_plot <-  data.frame(rho = mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,])),
                       sigma_c = sqrt(mvbart_mod$Sigma_post[1,1,]),
                       sigma_q = sqrt(mvbart_mod$Sigma_post[2,2,]),
                       y_hat = mvbart_mod$y_hat[1,1,]
)

par(mfrow = c(3,1))
plot(df_plot$sigma_c, type = "l", ylab= expression(sigma[c]),main =expression(sigma[c]) )
plot(df_plot$sigma_q, type = "l",ylab= expression(sigma[q]), main = expression(sigma[q]))
plot(df_plot$rho, type = "l", ylab = expression(rho), main  = expression(rho))


bart_mod <- dbarts::bart(x.train = x_train,y.train = y_mat[,1],x.test = x_test)
bart_mod$sigma %>% plot(type = "l")

