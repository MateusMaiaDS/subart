rm(list=ls())
library(subart)
library(expm)
library(scales)

# install.packages("devtools")
# devtools::install_github("MateusMaiaDS/subart",ref ="feat/binary_missing")

# univariate ####
# simulation setup
f <- function(x){
     c(
          5*sin(4*pi*x^2),
          5*sin(2*pi)^2
     )
}

p <- function(x){
     c(
          #ifelse(x > 0.6 & x < 0.8, 1, 0),
          pnorm(-2 + 2*x),
          0
     )
}

n <- 100
x <- seq(0, 1, length.out = n)

Sigma <- matrix(
     c(1, 0.5, 0.5, 1),
     ncol = 2
)
Sigma_sqrt <- sqrtm(Sigma)

# simulate data
y_hat <- matrix(NA, nrow = n, ncol = 2)
y <- matrix(NA, nrow = n, ncol = 2)
y_obs <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y_hat[i,] <- f(x[i])
     y[i,] <- y_hat[i,] + Sigma_sqrt %*% rnorm(2)
     for (j in 1:2){
          if (runif(1) < p(x[i])[j]) {y_obs[i,j] <- NA}
          else {y_obs[i,j] <- y[i,j]}
     }
}

# fit with imputation
n_mcmc <- 1500
fit <- subart(x_train = as.data.frame(x),
              y_mat = y_obs,
              x_test = as.data.frame(x),
              varimportance = FALSE)

y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
y2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y1_pred_band[i,1] <- quantile(fit$y_hat[i,1,], 0.05)
     y1_pred_band[i,2] <- quantile(fit$y_hat[i,1,], 0.95)
     y2_pred_band[i,1] <- quantile(fit$y_hat[i,2,], 0.05)
     y2_pred_band[i,2] <- quantile(fit$y_hat[i,2,], 0.95)
}

plot(x, y_hat[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), pch = 16, ylim = c(-15,15))
lines(x, fit$y_hat_mean[,1], col = "blue")
lines(x, y1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, y1_pred_band[,2], , col = "blue", lty = "dashed")

plot(y[,1], fit$y_hat_test_mean[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), ylim = c(-15,15))
lines(y[,1], y[,1])
y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
for ( i in 1:n ) {
     sim <- rnorm(n_mcmc, fit$y_hat_test[i,1,], sqrt(fit$Sigma_post[1,1,]))
     ci <- c(quantile(sim, 0.05), quantile(sim, 0.95))
     lines( c(y[i,1], y[i,1]) , ci, col = ifelse(is.na(y_obs[i,1]), alpha("red", 0.5), alpha("black", 0.5)))
}

# fit without imputation
NA_indicator <- rowSums(is.na(y_obs))
fit <- subart(x_train = data.frame(x = x[NA_indicator == 0]),
              y_mat = y_obs[NA_indicator == 0,],
              x_test = data.frame(x = x),
              varimportance = FALSE,
              n_mcmc = 1000,
              numcut = length(x[NA_indicator==0]))

y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
y2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y1_pred_band[i,1] <- quantile(fit$y_hat_test[i,1,], 0.05)
     y1_pred_band[i,2] <- quantile(fit$y_hat_test[i,1,], 0.95)
     y2_pred_band[i,1] <- quantile(fit$y_hat_test[i,2,], 0.05)
     y2_pred_band[i,2] <- quantile(fit$y_hat_test[i,2,], 0.95)
}

plot(x, y_hat[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), pch = 16, ylim = c(-5,25))
lines(x, fit$y_hat_test_mean[,1], col = "blue")
lines(x, y1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, y1_pred_band[,2], , col = "blue", lty = "dashed")


# multivariate ####
# simulation setup
f <- function(x){
     as.numeric(
          c(
               10*sin(pi*x[1]*x[2]),
               20*(x[1] - 0.5)^2 + 10 * x[2]
          )
     )
}

p <- function(x){
     as.numeric(
          c(
               #ifelse(x > 0.6 & x < 0.8, 1, 0),
               pnorm(as.numeric(-2 + x[1] + x[2])),
               0
          )
     )
}

n <- 20
x <- expand.grid(seq(0, 1, length.out = n),seq(0, 1, length.out = n))

Sigma <- matrix(
     c(1, 0.5, 0.5, 1),
     ncol = 2
)
Sigma_sqrt <- sqrtm(Sigma)

# simulate data
y_hat <- matrix(NA, nrow = n^2, ncol = 2)
y <- matrix(NA, nrow = n^2, ncol = 2)
y_obs <- matrix(NA, nrow = n^2, ncol = 2)
for (i in 1:n^2){
     y_hat[i,] <- f(x[i,])
     y[i,] <- y_hat[i,] + Sigma_sqrt %*% rnorm(2)
     for (j in 1:2){
          if (runif(1) < p(x[i,])[j]) {y_obs[i,j] <- NA}
          else {y_obs[i,j] <- y[i,j]}
     }
}

# fit with imputation
n_mcmc <- 1500
fit <- subart(x_train = as.data.frame(x),
              y_mat = y_obs,
              x_test = as.data.frame(x),
              varimportance = FALSE)

y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
y2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y1_pred_band[i,1] <- quantile(fit$y_hat[i,1,], 0.05)
     y1_pred_band[i,2] <- quantile(fit$y_hat[i,1,], 0.95)
     y2_pred_band[i,1] <- quantile(fit$y_hat[i,2,], 0.05)
     y2_pred_band[i,2] <- quantile(fit$y_hat[i,2,], 0.95)
}

plot(y[,1], fit$y_hat_test_mean[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), ylim = c(-15,15))
lines(y[,1], y[,1])
y1_pred_band <- matrix(NA, nrow = n^2, ncol = 2)
for ( i in 1:n^2 ) {
     sim <- rnorm(n_mcmc, fit$y_hat_test[i,1,], sqrt(fit$Sigma_post[1,1,]))
     ci <- c(quantile(sim, 0.05), quantile(sim, 0.95))
     lines( c(y[i,1], y[i,1]) , ci, col = ifelse(is.na(y_obs[i,1]), alpha("red", 0.5), alpha("black", 0.5)))
}

# fit without imputation
NA_indicator <- rowSums(is.na(y_obs))
fit <- subart(x_train = data.frame(x = x[NA_indicator == 0]),
              y_mat = y_obs[NA_indicator == 0,],
              x_test = data.frame(x = x),
              varimportance = FALSE,
              n_mcmc = 1000,
              numcut = length(x[NA_indicator==0]))

y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
y2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y1_pred_band[i,1] <- quantile(fit$y_hat_test[i,1,], 0.05)
     y1_pred_band[i,2] <- quantile(fit$y_hat_test[i,1,], 0.95)
     y2_pred_band[i,1] <- quantile(fit$y_hat_test[i,2,], 0.05)
     y2_pred_band[i,2] <- quantile(fit$y_hat_test[i,2,], 0.95)
}

plot(x, y_hat[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), pch = 16, ylim = c(-5,25))
lines(x, fit$y_hat_test_mean[,1], col = "blue")
lines(x, y1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, y1_pred_band[,2], , col = "blue", lty = "dashed")
