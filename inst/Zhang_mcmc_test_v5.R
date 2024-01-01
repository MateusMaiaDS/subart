library(ggplot2)
library(GGally)
library(bayesplot)
library(gridExtra)
library(CholWishart)
library(progress)

# functions
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

makeSigmaInv <- function(Sigma, d){

  indices <- expand.grid(1:d, 1:d)
  indices <- indices[indices[,1] < indices[2],]
  indices <- indices[order(indices[,1],indices[,2]),]

  sigma <- rep(NA, (d^2 - d)/2)
  for (i in 1:((d^2 - d)/2)){
    sigma[i] <- Sigma[indices[i,1], indices[i,2]]
  }

  return(sigma)
}

log_prior_dens <- function(R, D, nu){
  d <- ncol(D)
  W <- sqrt(D) %*% R %*% sqrt(D)
  return(
    dWishart(W, nu + d - 1, diag(d), log = TRUE) + ((d-1)/2) * sum(log(diag(D)))
  )
}

log_posterior_dens <- function(R, D, nu, Z, sample_prior){
  if (sample_prior == TRUE) {return(log_prior_dens(R,D,nu))}
  else{
    return(
      log_prior_dens(R,D,nu)
      - n/2 * log(det(R))
      - 1/2 * sum(apply(Z, 1, function(y) y %*% solve(R) %*% y))
    )
  }
}

log_proposal_dens <- function(R_star, D_star, R, D, m){
  W_star <- sqrt(D_star) %*% R_star %*% sqrt(D_star)
  W <- sqrt(D) %*% R %*% sqrt(D)
  return(
    dInvWishart(W_star, m, m * W, log = TRUE) + ((d-1)/2) * sum(log(diag(D_star)))
  )
}

# wishart_loglikelihood(X = Sigma_true,nu = 3,Sigma = Sigma_true)

# simulate data
n <- 50
d <- 2
sigma_true <- c(0.1) # must be of length (d^2 - d)/2
Sigma_true <- makeSigma(sigma_true, d)
min(eigen(Sigma_true)$value) # must be > 0
Sigma_true_chol <- t(chol(Sigma_true))
Z <- matrix(NA, nrow = n, ncol = d)

for (i in 1:n){
  Z[i,] <- Sigma_true_chol %*% rnorm(d)
}

dWishart(x = Sigma_true,df = 3,Sigma = Sigma_true,log = TRUE)

A <- rWishart(1, 10, diag(4))[, , 1]
dWishart(x = A, df = 10, Sigma = diag(4L), log = TRUE)
wishart_loglikelihood(X = A,nu = 10,Sigma = diag(4L))
dInvWishart(x = A, df = 10, Sigma = diag(4L), log = TRUE)
iwishart_loglikelihood(A,diag(4L),10)
LaplacesDemon::dinvwishart(Sigma = A,nu = 10,S = diag(4L),log = TRUE)


LaplacesDemon::dinvwishart(Sigma = W_star,nu = 200,S = 200*W,log = TRUE)
iwishart_loglikelihood(W_star,S = 200*W,200)

rInvWishart(n = 1,df = 200,Sigma = 200*Sigma_true)
# sampler ####
# if you want to sample from the prior, set sample_prior <- TRUE
d <- 2
n_mcmc <- 10000
sample_prior <- FALSE
nu <- 2
m <- 200 # degrees of freedom for proposal distribution; must be > d. Large m leads to smaller jumps and higher acceptance rates
pb <- progress_bar$new(total = n_mcmc)
R_samples <- list()
D_samples <- list()
R_samples[[1]] <- diag(d)
D_samples[[1]] <- diag(d)
acceptance <- rep(NA, n_mcmc)
acceptance[1] <- 1

for (i in 2:n_mcmc){
  pb$tick()
  R_current <- R_samples[[i-1]]
  D_current <- D_samples[[i-1]]
  W_current <- sqrt(D_samples[[i-1]]) %*% R_samples[[i-1]] %*% sqrt(D_samples[[i-1]])
  W_proposal <- rInvWishart(1, m, m * W_current)[,,1]
  D_proposal <- diag(diag(W_proposal))
  R_proposal <- solve(sqrt(D_proposal)) %*% W_proposal %*% solve(sqrt(D_proposal))

  alpha <- min(
    exp(
      (log_posterior_dens(R_proposal, D_proposal, nu, Z, sample_prior) - log_posterior_dens(R_current, D_current, nu, Z, sample_prior)) +
      (log_proposal_dens(R_current, D_current, R_proposal, D_proposal, m) - log_proposal_dens(R_proposal, D_proposal, R_current, D_current, m))
    ),
    1
  )
  u <- runif(1, 0, 1)
  if (u < alpha) {
    R_samples[[i]] <- R_proposal
    D_samples[[i]] <- D_proposal
  acceptance[i] <- 1
  } else {
    R_samples[[i]] <- R_samples[[i-1]]
    D_samples[[i]] <- D_samples[[i-1]]
    acceptance[i] <- 0
  }
}


# check results

mean(acceptance)

R_samples_df <- data.frame(
  index = 1:n_mcmc,
  r1 = NA,
  r2 = NA,
  r3 = NA
  )

for (i in 1:n_mcmc){
  R_samples_df[i,] <- c(i,makeSigmaInv(R_samples[[i]], d))
}

grid.arrange(
  ggplot(R_samples_df) +
    geom_line(aes(index, r1)) +
    scale_y_continuous(limits = c(-1,1)),
  ggplot(R_samples_df) +
    geom_line(aes(index, r2)) +
    scale_y_continuous(limits = c(-1,1)),
  ggplot(R_samples_df) +
    geom_line(aes(index, r3)) +
    scale_y_continuous(limits = c(-1,1)),
  ncol = 1
)

ggpairs(R_samples_df[((n_mcmc/2)+1):n_mcmc,-1])


