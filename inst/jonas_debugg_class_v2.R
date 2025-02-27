library(subart)
library(expm)
library(scales)
rm(list=ls())
devtools::load_all()
# simulation setup
f <- function(x){
     c(
          #1 + 2 * sin(2*pi*x),
          1,
          cos(2*pi*x)
     )
}

p <- function(x){
     c(
          ifelse(x > 0.7 & x < 0.8 | x > 0.3 & x < 0.4, 1, 0),
          #pnorm(-2 + 2*x),
          0
     )
}

n <- 500
x <- seq(0, 1, length.out = n)

Sigma <- matrix(
     c(1, 0.5, 0.5, 1),
     ncol = 2
)
Sigma_sqrt <- sqrtm(Sigma)

# simulate data
z_hat <- matrix(NA, nrow = n, ncol = 2)
z <- matrix(NA, nrow = n, ncol = 2)
y <- matrix(NA, nrow = n, ncol = 2)
y_obs <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     z_hat[i,] <- f(x[i])
     z[i,] <- z_hat[i,] + Sigma_sqrt %*% rnorm(2)
     y[i,] <- as.numeric(z[i,] > 0)
     for (j in 1:2){
          if (runif(1) < p(x[i])[j]) {y_obs[i,j] <- NA}
          else {y_obs[i,j] <- y[i,j]}
     }
}

# fit with imputation
n_mcmc <- 1000
fit <- subart(x_train = as.data.frame(x),
              y_mat = y_obs,
              x_test = as.data.frame(x),
              varimportance = FALSE)

z1_pred_band <- matrix(NA, nrow = n, ncol = 2)
z2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     z1_pred_band[i,1] <- quantile(fit$y_hat[i,1,], 0.05)
     z1_pred_band[i,2] <- quantile(fit$y_hat[i,1,], 0.95)
     z2_pred_band[i,1] <- quantile(fit$y_hat[i,2,], 0.05)
     z2_pred_band[i,2] <- quantile(fit$y_hat[i,2,], 0.95)
}

plot(x, z_hat[,1], col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), pch = 16, ylim = c(-3,3))
lines(x, fit$y_hat_mean[,1], col = "blue")
lines(x, z1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, z1_pred_band[,2], , col = "blue", lty = "dashed")

plot(x, pnorm(z_hat[,1]), col = ifelse(is.na(y_obs[,1]), alpha("red", 0.5), alpha("black", 0.5)), pch = 16, ylim = c(0,1))
lines(x, pnorm(z1_pred_band[,1]), col = "blue", lty = "dashed")
lines(x, pnorm(z1_pred_band[,2]) , col = "blue", lty = "dashed")
