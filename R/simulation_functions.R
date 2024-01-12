#' Simulation setting for the Regression case
#'
#' @param n sample size
#' @param p number of covariates
#' @param mvn_dim dimension \eqn{d} for the multivariate
#' @param Sigma Sigma function, default is \code{NULL}
#'
#' @return a simulated data
#' @export
#'
#' @examples
sim_mvn_friedman1 <- function(n, p, mvn_dim,Sigma = NULL){


     # Setting some default values for Sigma
     if(mvn_dim==3){
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
          determinant(Sigma)$modulus[1]
          eigen(Sigma)$values
     } else {
          sigma1 <- 1
          sigma2 <- 10
          rho12 <- 0.75 # Original is 0.75
          Sigma <- diag(c(sigma1^2,sigma2^2),nrow = mvn_dim)
          Sigma[1,2] <- Sigma[2,1] <-sigma1*sigma2*rho12
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
     x <- matrix(runif(p*n), ncol = p)
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

#' Simulation setting for the Regression Friedman 2
#'
#' @param n sample size
#' @param p number of covariates
#' @param mvn_dim dimension \eqn{d} for the multivariate
#' @param Sigma Sigma function, default is \code{NULL}
#'
#' @return a simulated data
#' @export
#'
#' @examples
sim_mvn_friedman2 <- function(n, p, mvn_dim,Sigma = NULL){


     # Setting the Sigma if Sigma is set for NULL
     if(is.null(Sigma)){
          if(mvn_dim==3){
               sigma1 <- 1
               sigma2 <- 125
               sigma3 <- 0.1
               rho12 <- 0.8
               rho13 <- 0.5
               rho23 <- 0.25
               Sigma <- diag(c(sigma1^2,sigma2^2,sigma3^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <- rho12*sigma1*sigma2
               Sigma[1,3] <- Sigma[3,1] <- rho13*sigma1*sigma3
               Sigma[2,3] <- Sigma[3,2] <- rho23*sigma2*sigma3
               determinant(Sigma)$modulus[1]
               eigen(Sigma)$values
          } else {
               sigma1 <- 1
               sigma2 <- 125
               rho12 <- 0.75
               Sigma <- diag(c(sigma1^2,sigma2^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <-sigma1*sigma2*rho12
               determinant(Sigma)$modulus[1]
               eigen(Sigma)$values

          }
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
     x1 <- matrix(runif(5*n), ncol = 5)
     y1 <- 10*sin(x1[,1]*x1[,2]*pi) + 20*(x1[,3]-0.5)^2 + 10*x1[,4] + 5*x1[,5]

     x2 <- cbind(runif(n = n,max = 100),
                 runif(n = n,min = 40*pi,max = 560*pi),
                 runif(n = n,min = 0,max = 1),
                 runif(n = n,min = 1,max = 11))

     y2 <- (x2[,1]^2 + (x2[,2]*x2[,3]-(1/(x2[,2]*x2[,4])))^(2))^(0.5)

     # Adding the only if p=3
     if(mvn_dim==3){
          y3 <- atan((x2[,2]*x2[,3]-1/(x2[,2]*x2[,4]))/x2[,1])
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

     # Generating the remaining covariates variables
     if(p > 9 ){
          x_noise <- matrix(runif(n = n*(p-9)),ncol = (p-9))
          x <- cbind(x1,x2,x_noise)
     } else {
          x <- cbind(x1,x2)
     }


     # Return a list with all the quantities
     return(list( x = data.frame(x) ,
                  y = y,
                  y_true = y_true,
                  Sigma = Sigma))
}

#' Simulation setting for the Classification case
#'
#' @param n sample size
#' @param p number of covariates
#' @param mvn_dim dimension \eqn{d} for the multivariate
#' @param Sigma Sigma function, default is \code{NULL}
#'
#' @return a simulated data
#' @export
#'
#' @examples
sim_class_mvn_friedman1 <- function(n, p, mvn_dim,Sigma = NULL){


        if (mvn_dim == 3) {
                sigma1 <- 1
                sigma2 <- 1
                sigma3 <- 1
                rho12 <- 0.8
                rho13 <- 0.5
                rho23 <- 0.25
                Sigma <- diag(c(sigma1^2, sigma2^2, sigma3^2), nrow = mvn_dim)
                Sigma[1, 2] <- Sigma[2, 1] <- rho12 * sigma1 * sigma2
                Sigma[1, 3] <- Sigma[3, 1] <- rho13 * sigma1 * sigma3
                Sigma[2, 3] <- Sigma[3, 2] <- rho23 * sigma2 * sigma3
                determinant(Sigma)$modulus[1]
                eigen(Sigma)$values
        }
        else {
                sigma1 <- 1
                sigma2 <- 1
                rho12 <- 0.75
                Sigma <- diag(c(sigma1^2, sigma2^2), nrow = mvn_dim)
                Sigma[1, 2] <- Sigma[2, 1] <- sigma1 * sigma2 * rho12
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
        x <- matrix(runif(p * n, min = -1, max = 1), ncol = p)
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
                        p_true[i, ] <- pnorm(z_true[i, ])
                }
        }
        else if (mvn_dim == 2) {
                z_true <- cbind(z1, z2)
                for (i in 1:n) {
                        z[i, ] <- z_true[i, ] + mvnfast::rmvn(n = 1, mu = rep(0,
                                                                              mvn_dim), sigma = Sigma)
                        y[i, ] <- (z[i, ] > 0)
                        y_true[i, ] <- (z_true[i, ] > 0)
                        p_true[i, ] <- pnorm(z_true[i, ])
                }
        }
        return(list(x = data.frame(x), y = y, z = z, z_true = z_true, y_true = y_true,
                    p_true = p_true, Sigma = Sigma))

}

#' Simulation setting for the Classification case
#'
#' @param n sample size
#' @param p number of covariates
#' @param mvn_dim dimension \eqn{d} for the multivariate
#' @param Sigma Sigma function, default is \code{NULL}
#'
#' @return a simulated data
#' @export
#'
#' @examples
sim_class_mvn_friedman2 <- function(n, p, mvn_dim,Sigma = NULL){


     # Setting the Sigma if Sigma is set for NULL
     if(is.null(Sigma)){
          if(mvn_dim==3){
               sigma1 <- 1
               sigma2 <- 1
               sigma3 <- 1
               rho12 <- 0.8
               rho13 <- 0.5
               rho23 <- 0.25
               Sigma <- diag(c(sigma1^2,sigma2^2,sigma3^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <- rho12*sigma1*sigma2
               Sigma[1,3] <- Sigma[3,1] <- rho13*sigma1*sigma3
               Sigma[2,3] <- Sigma[3,2] <- rho23*sigma2*sigma3
               determinant(Sigma)$modulus[1]
               eigen(Sigma)$values
          } else {
               sigma1 <- 1
               sigma2 <- 1
               rho12 <- 0.75
               Sigma <- diag(c(sigma1^2,sigma2^2),nrow = mvn_dim)
               Sigma[1,2] <- Sigma[2,1] <-sigma1*sigma2*rho12
               determinant(Sigma)$modulus[1]
               eigen(Sigma)$values

          }
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
     x1 <- matrix(runif(5*n), ncol = 5)
     z1 <- 10*sin(x1[,1]*x1[,2]*pi) - 20*(x1[,3]-0.5)^2 - 10*x1[,4] + 5*x1[,5]

     x2 <- cbind(runif(n = n,max = 100),
                 runif(n = n,min = 40*pi,max = 560*pi),
                 runif(n = n,min = 0,max = 1),
                 runif(n = n,min = 1,max = 11))

     z2 <- cos(x2[,1]^2 + (x2[,2]*x2[,3]-(1/(x2[,2]*x2[,4]))))

     # Adding the only if p=3
     if(mvn_dim==3){
          z3 <- sin((x2[,2]*x2[,3]-1/(x2[,2]*x2[,4]))/x2[,1])
     }

     y <- matrix(0,nrow = n,ncol = mvn_dim)
     y_true <- matrix(0,nrow = n, ncol = mvn_dim)
     z <- matrix(0, nrow = n, ncol = mvn_dim)
     p_true <- matrix(0, nrow = n, ncol = mvn_dim)

     if(mvn_dim==3){
          z_true <- cbind(z1,z2,z3)
          for(i in 1:n){
               z[i,] <- z_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
               y[i,] <- (z[i,]>0)
               y_true[i,] <- (z_true[i,]>0)
               p_[i,] <- pnorm(z[i,])
          }

     } else if(mvn_dim==2){
          z_true <- cbind(z1,z2)
          for(i in 1:n){
               z[i,] <- z_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
               y[i,] <- (z[i,]>0)
               y_true[i,] <- (z_true[i,]>0)
               p_true[i,] <- pnorm(z_true[i,])
          }
     }

     # Generating the remaining covariates variables
     if(p > 9 ){
          x_noise <- matrix(runif(n = n*(p-9)),ncol = (p-9))
          x <- cbind(x1,x2,x_noise)
     } else {
          x <- cbind(x1,x2)
     }


     # Return a list with all the quantities
     return(list( x = data.frame(x) ,
                  y = y,
                  z_true = z_true,
                  y_true = y_true,
                  p_true = p_true,
                  Sigma = Sigma))
}

