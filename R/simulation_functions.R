#' Simulation settings for the Friedman #1 regression case
#'
#' This function generates data according to the Friedman #1 scenario from
#' Esser et al. (2024, arXiv:2404.02228). The p = 10 predictors are generated
#' independently from a uniform distribution on [0, 1]. Not all predictors are
#' used in each response, but all predictors are included in the model fitting.
#'
#' The three responses are generated as:
#' \deqn{
#' y_1 = 10 \sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2
#' }
#' \deqn{
#' y_2 = 8 x_4 + 20 \sin(\pi x_1)
#' }
#' \deqn{
#' y_3 = 10 x_5 - 5 x_2 - 5 x_4
#' }
#'
#' @param n Sample size
#' @param p Number of covariates (default 10)
#' @param mvn_dim Dimension \eqn{d} of the multivariate response (2 or 3)
#' @param Sigma Covariance matrix or function to generate it; default is \code{NULL}.
#'   Example values for d = 2: \eqn{\sigma_1 = 1, \sigma_2 = 10, \rho_{12} = 0.75};
#'   for d = 3: \eqn{\sigma = (1, 2.5, 5), \rho = (0.8, 0.5, 0.25)}
#'
#' @return A data frame with n rows, p predictor columns, and mvn_dim outcome columns
#' @export

sim_mvn_friedman1 <- function(n, p, mvn_dim,Sigma = NULL){


     # Setting some default values for Sigma
     if(mvn_dim==3){

          if(is.null(Sigma)){

               sigma1 <- 1
               sigma2 <- 2.5
               sigma3 <- 5
               rho12 <- 0.8
               rho13 <- 0.5
               rho23 <- 0.25

               Sigma <- diag(c(sigma1^2,sigma2^2,sigma3^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <- rho12*sigma1*sigma2
               Sigma[1,3] <- Sigma[3,1] <- rho13*sigma1*sigma3
               Sigma[2,3] <- Sigma[3,2] <- rho23*sigma2*sigma3
          }

          determinant(Sigma)$modulus[1]
          eigen(Sigma)$values
     } else {
          sigma1 <- 1
          sigma2 <- 10
          rho12 <- 0.75 # Original is 0.75
          if(is.null(Sigma)){
               Sigma <- diag(c(sigma1^2,sigma2^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <-sigma1*sigma2*rho12
          }
          determinant(Sigma)$modulus[1]
          eigen(Sigma)$values

     }

     # Verifying if it is a valid Sigma.
     if(NROW(Sigma)!=mvn_dim | NCOL(Sigma)!=mvn_dim){
          stop(paste0("Insert a valid Sigma matrix for the ",mvn_dim,"-d case."))
     }
     # Verifying if is semi-positive-define
     if(!all(eigen(Sigma)$values>0)){
          stop("Insert a positive-semidefined matrix")
     }

     # Generate the x matrix
     x <- matrix(stats::runif(p*n), ncol = p)
     y1 <- 10*sin(x[,1]*x[,2]*pi) + 20*(x[,3]-0.5)^2
     y2 <- 8*x[,4] + 20*sin(x[,1]*pi)

     # Adding the only if p=3
     if(mvn_dim==3){
            y3 <- 10* x[, 5] - 5 * x[, 2] - 5 * x[,4]
     }

     y <- matrix(0,nrow = n,ncol = mvn_dim)
     if(mvn_dim==3){
          y_true <- cbind(y1,y2,y3)
          for(i in 1:n){
               y[i,] <- y_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
          }

     } else if(mvn_dim==2){
          y_true <- cbind(y1,y2)
          for(i in 1:n){
               y[i,] <- y_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
          }
     }

     # Return a list with all the quantities
     return(list( x = data.frame(x) ,
                  y = y,
                  y_true = y_true,
                  Sigma = Sigma))
}

#' Simulation setting for the Classification case (Friedman #2)
#'
#' This function generates data according to the Friedman #2 scenario from
#' Esser et al. (2024, arXiv:2404.02228). The p = 10 predictors are generated
#' independently from a uniform distribution on [0, 1].
#'
#' The latent variables for each response are generated as:
#' \deqn{
#' z_1 = \sin(\pi x_1 x_2) + x_3^3
#' }
#' \deqn{
#' z_2 = -1 + 2 x_1 x_4 + \exp(x_5)
#' }
#' \deqn{
#' z_3 = 0.5(x_2 + x_4) + x_5
#' }
#' Correlated noise \eqn{\varepsilon_i \sim MVN_d(0, \Sigma)} is added to each
#' latent variable. The binary outcomes are then obtained via the probit link function.
#'
#' The correlation matrix \eqn{\Sigma} can be specified according to Table 2 in
#' the paper for d = 2 or d = 3. The dimension \eqn{d = 2} uses the first two responses.
#'
#' @param n Sample size
#' @param p Number of covariates (default 10)
#' @param mvn_dim Dimension \eqn{d} of the multivariate latent variable (2 or 3)
#' @param Sigma Covariance matrix or function to generate it; default is \code{NULL}.
#'   For d = 2, use \eqn{\sigma_1 = 1, \sigma_2 = 10, \rho_{12} = 0.75};
#'   for d = 3, \eqn{\sigma = (1, 2.5, 5), \rho = (0.8, 0.5, 0.25)}.
#'
#' @return A data frame with n rows, p predictor columns, and mvn_dim binary outcome columns
#' @export

sim_mvn_friedman2 <- function(n, p, mvn_dim,Sigma = NULL){


     if (mvn_dim == 3) {
          sigma1 <- 1
          sigma2 <- 1
          sigma3 <- 1
          rho12 <- 0.8
          rho13 <- 0.5
          rho23 <- 0.25
          if(is.null(Sigma)){
               Sigma <- diag(c(sigma1^2, sigma2^2, sigma3^2), nrow = mvn_dim)
               Sigma[1, 2] <- Sigma[2, 1] <- rho12 * sigma1 * sigma2
               Sigma[1, 3] <- Sigma[3, 1] <- rho13 * sigma1 * sigma3
               Sigma[2, 3] <- Sigma[3, 2] <- rho23 * sigma2 * sigma3
          }
          determinant(Sigma)$modulus[1]
          eigen(Sigma)$values
     }
     else {
          sigma1 <- 1
          sigma2 <- 1
          rho12 <- 0.75
          if(is.null(Sigma)){
               Sigma <- diag(c(sigma1^2, sigma2^2), nrow = mvn_dim)
               Sigma[1, 2] <- Sigma[2, 1] <- sigma1 * sigma2 * rho12
          }
          determinant(Sigma)$modulus[1]
          eigen(Sigma)$values
     }
     if (NROW(Sigma) != mvn_dim | NCOL(Sigma) != mvn_dim) {
          stop(paste0("Insert a valid Sigma matrix for the ", mvn_dim,
                      "-d case."))
     }
     if (!all(eigen(Sigma)$values > 0)) {
          stop("Insert a positive-semidefined matrix")
     }
     x <- matrix(stats::runif(p * n, min = -1, max = 1), ncol = p)
     z1 <- sin(x[, 1] * x[, 2] * pi) + x[, 3]^3
     z2 <- -1 + 2 * x[,1]*x[,4] + exp(x[,5])
     if (mvn_dim == 3) {
          z3 <- 0.5 * x[,2] + 0.5 * x[,4] + x[,5]
     }
     y <- matrix(0, nrow = n, ncol = mvn_dim)
     y_true <- matrix(0, nrow = n, ncol = mvn_dim)
     z <- matrix(0, nrow = n, ncol = mvn_dim)
     p_true <- matrix(0, nrow = n, ncol = mvn_dim)
     if (mvn_dim == 3) {
          z_true <- cbind(z1, z2, z3)
          for (i in 1:n) {
               z[i, ] <- z_true[i, ] + mvnfast::rmvn(n = 1, mu = rep(0,
                                                                     mvn_dim), sigma = Sigma)
               y[i, ] <- (z[i, ] > 0)
               y_true[i, ] <- (z_true[i, ] > 0)
               p_true[i, ] <- stats::pnorm(z_true[i, ])
          }
     }
     else if (mvn_dim == 2) {
          z_true <- cbind(z1, z2)
          for (i in 1:n) {
               z[i, ] <- z_true[i, ] + mvnfast::rmvn(n = 1, mu = rep(0,
                                                                     mvn_dim), sigma = Sigma)
               y[i, ] <- (z[i, ] > 0)
               y_true[i, ] <- (z_true[i, ] > 0)
               p_true[i, ] <- stats::pnorm(z_true[i, ])
          }
     }
     return(list(x = data.frame(x), y = y, z = z, z_true = z_true, y_true = y_true,
                 p_true = p_true, Sigma = Sigma))

}

