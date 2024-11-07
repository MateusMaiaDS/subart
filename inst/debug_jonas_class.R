devtools::load_all()
library(expm)
rm(list=ls())
# simulation setup
f <- function(x){
     c(
          sin(2*pi*x),
          cos(2*pi*x)
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

fit <- subart(x_train = as.data.frame(x),
              y_mat = y_obs,
              x_test = as.data.frame(x),
              varimportance = FALSE,
              n_mcmc = 1000,hier_prior_bool = TRUE)

y1_pred_band <- matrix(NA, nrow = n, ncol = 2)
y2_pred_band <- matrix(NA, nrow = n, ncol = 2)
for (i in 1:n){
     y1_pred_band[i,1] <- quantile(fit$y_hat_test[i,1,], 0.05)
     y1_pred_band[i,2] <- quantile(fit$y_hat_test[i,1,], 0.95)
     y2_pred_band[i,1] <- quantile(fit$y_hat_test[i,2,], 0.05)
     y2_pred_band[i,2] <- quantile(fit$y_hat_test[i,2,], 0.95)
}




plot(x, y_hat[,1], col = ifelse(is.na(y_obs[,1]), "red", "black"), ylim = c(-12, 12))
lines(x, fit$y_hat_test_mean[,1], col = "blue")
lines(x, y1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, y1_pred_band[,2], , col = "blue", lty = "dashed")


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

plot(x, z_hat[,1], col = ifelse(is.na(y_obs[,1]), "red", "black"))
lines(x, fit$y_hat_test_mean[,1], col = "blue")
lines(x, y1_pred_band[,1], col = "blue", lty = "dashed")
lines(x, y1_pred_band[,2], , col = "blue", lty = "dashed")

