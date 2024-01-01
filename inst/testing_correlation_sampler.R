rm(list=ls())
# Loading the mian function
Rcpp::sourceCpp("src/demo.cpp")

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

# Simluating Residuals
n <- 500
d <- 3

sigma_true <- c(0.8,0.3,-0.1) # must be of length (d^2 - d)/2

Sigma_true <- makeSigma(sigma_true, d)
det(Sigma_true) # must be > 0
Sigma_true_chol <- t(chol(Sigma_true))
resid <- matrix(NA, nrow = n, ncol = d)

for (i in 1:n){
     resid[i,] <- Sigma_true_chol %*% rnorm(d)
}


# Setting the initial values
sigma0 <- rep(0.0, (d^2 - d)/2)
Sigma0 <- makeSigma(sigma0,(d*d-d)*0.5)
diag(Sigma0) <- rep(1, nrow(Sigma0))
df_ <- 5

#Just to auxiliar I gonna define y_mat as the residuals and y_hat as the zero matrix
y_mat_ <- resid
y_hat_ <- matrix(0,nrow = nrow(y_mat_),ncol = ncol(y_mat_))
n_mcmc <- 2000

# Hacky way to set an initial value?
if(d==3){
     sigma_init <- c(cor(resid[,1],resid[,2]),cor(resid[,3],resid[,1]),cor(resid[,2],resid[,3]))
} else{
     sigma_init <- sigma0
}

sigma_post <- sigma_sampler(nmcmc = n_mcmc,
              d = d,
              sigma_0 = sigma0,
              # sigma_init_optim = sigma_init,
              y_mat = y_mat_,
              y_hat = y_hat_,df = df_,Sigma_0 = Sigma0)

# Plottin for the 2d case
if(d==2){
     par(mfrow= c(1,1))
     plot(c(sigma_post),type = "l")
     abline(h = sigma_true, lty = 2, col = "blue", lwd = 2)
} else {
     par(mfrow = c(nrow(sigma_post),1))
     for(i in 1:nrow(sigma_post)){
          exp_ <- expression(sigma(i))
          plot(sigma_post[i,], type = "l", ylab = expression(sigma))
          abline(h = sigma_true[i], lty = 2, col = "blue", lwd = 2)
     }
}


