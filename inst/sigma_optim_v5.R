set.seed(42)
# function to create correlation matrix ouf of correlation coefficients
makeSigma <- function(sigma, d){
     Sigma <- diag(d)

     indices <- expand.grid(1:d, 1:d)
     indices <- indices[indices[,1] < indices[2],]
     indices <- indices[order(indices[,1],indices[,2]),]

     for (i in 1:d){
          for (j in 1:d){
               Sigma[i,j] <- ifelse(
                    i == j,
                    1,
                    ifelse(
                         i < j,
                         sigma[which(indices[,1] == i & indices[,2] == j)],
                         sigma[which(indices[,1] == j & indices[,2] == i)]
                    )
               )
          }
     }
     return(Sigma)
}

# target function
# logpost <- function(sigma,d sigma0, sigma){

logpost <- function(sigma){
     Sigma <- makeSigma(sigma, d)
     return(
          - n/2 * log(det(Sigma))
          - 1/2 * sum(apply(resid, 1, function(y) y %*% solve(Sigma) %*% y))
          - 1/2 * (sigma - sigma0) %*% solve(Sigma0) %*% (sigma - sigma0)
     )
}

logpost(sigma0,2,sigma0)
log_mvn_post_cor_sample(resid, resid-resid,sigma0,d,sigma0)

# full function to get metropolis draw
sigma_draw <- function(d, n, residuals, sigma0, Sigma0, df, init, maxit){
     fit <- optim(
          par = init,
          fn = logpost,
          method = "BFGS",
          hessian = TRUE,
          control = list(fnscale = -1,
                         maxit = maxit),
     )
     mu <- fit$par
     S <- -1 * solve(fit$hessian)
     y <- t(chol(S)) %*% rnorm((d^2 - d)/2)
     u <- rchisq(1, df)
     return(
          list(
               mu = mu,
               draw = sqrt(df/u) * y + mu
          )
     )
}

# simulate residuals
n <- 50
d <- 2

sigma_true <- c(0.5) # must be of length (d^2 - d)/2
Sigma_true <- makeSigma(sigma_true, d)
det(Sigma_true) # must be > 0
Sigma_true_chol <- t(chol(Sigma_true))
resid <- matrix(NA, nrow = n, ncol = d)

for (i in 1:n){
     resid[i,] <- Sigma_true_chol %*% rnorm(d)
}


residuals <- resid
# define priors
sigma0 <- rep(0, (d^2 - d)/2)
Sigma0 <- 1 * diag((d^2 - d)/2)
maxit <- 3
df <- 5
init <- sigma0

sigma_draw(2,sigma0,resid,resid-resid,5)

# test sampler
sigma_draw(d = d,
           n = n,
           residuals = resid,
           sigma0 = sigma0,
           Sigma0 = Sigma0,
           df = 5,
           init = sigma0,
           maxit = 3)

# Sorting values
fit <- optim(
     par = init,
     fn = logpost,
     method = "BFGS",
     hessian = TRUE,
     control = list(fnscale = -1,
                    maxit = maxit),
)

fit

aux <- rosen_bfgs(d,sigma0,resid,resid-resid)

log_mvn_post_cor_sample(resid,
                        resid,
                        sigma0,d,
                        Sigma0)

# Loading the Rcpp sigma sampler
aux <- sigma_sampler(2000,
                        d,
                        sigma0,
                        resid,
                        resid-resid,
                        5)

# test mcmc
n_mcmc <- 500
sigma_samples <- matrix(NA, ncol = (d*d-d)/2, nrow = n_mcmc)
sigma_samples[1,] <- sigma0
init <- sigma0
for (i in 2:n_mcmc){
     proposal <- sigma_draw(d = d,
                            n = n,
                            residuals = resid,
                            sigma0 = sigma0,
                            Sigma0 = Sigma0,
                            df = 5,
                            init = init,
                            maxit = 3)

     draw <- as.vector(proposal$draw)

     switch_prob <- ifelse(
          det(makeSigma(draw, d)) > 0 & max(abs(draw)) < 1,
          exp(logpost(draw) - logpost(sigma_samples[i-1,])),
          0
     )
     u <- runif(1)
     if (u < switch_prob ) {
          sigma_samples[i,] <- draw
     } else {
          sigma_samples[i,] <- sigma_samples[i-1,]
     }
     init <- proposal$mu
}

plot(sigma_samples[,1], type = "l")
