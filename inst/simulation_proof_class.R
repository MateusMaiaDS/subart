devtools::load_all()
set.seed(42)
# test for binary outcomes ####
p <- 10
n <- 1000
mvn_dim <- 2
if(mvn_dim==3){
     rho12 <- 0.8
     rho13 <- 0.5
     rho23 <- 0.25
     Sigma <- diag(mvn_dim)
     Sigma[1,2] <- Sigma[2,1] <- rho12
     Sigma[1,3] <- Sigma[3,1] <- rho13
     Sigma[2,3] <- Sigma[3,2] <- rho23
     determinant(Sigma)$modulus[1]
     eigen(Sigma)$values
} else {
     Sigma <- diag(nrow = mvn_dim)
     Sigma[1,2] <- Sigma[2,1] <- 0.75
     determinant(Sigma)$modulus[1]
     eigen(Sigma)$values

}

sim_mvn_binary <- function(n, p, mvn_dim,Sigma,seed = NULL){

     # Setting the seed

     # Generate the x matrix
     x <- matrix(runif(p*n), ncol = p)
     z1 <- 2*x[,1]*x[,2]
     z2 <- x[,4] + sin(x[,1]*2*pi)

     # Adding the only if p=3
     if(mvn_dim==3){
          z3 <- cos(2*pi*(x[,5] - x[,3]))
     }

     z <- matrix(0,nrow = n, ncol = mvn_dim)
     y <- matrix(0,nrow = n, ncol = mvn_dim)
     p <- matrix(0,nrow = n, ncol = mvn_dim)
     if(mvn_dim==3){
          z_true <- cbind(z1,z2,z3)
          for(i in 1:n){
               z[i,] <- z_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
               y[i,] <- (z[i,] > 0)
               p[i,] <- pnorm(z_true[i,])
          }

     } else if(mvn_dim==2){
          z_true <- cbind(z1,z2)
          for(i in 1:n){
               z[i,] <- z_true[i,] + mvnfast::rmvn(n = 1,mu = rep(0,mvn_dim),sigma = Sigma)
               y[i,] <- (z[i,] > 0)
               p[i,] <- pnorm(z_true[i,])
          }
     }

     # Return a list with all the quantities
     return(list( x = x ,
                  y = y,
                  z_true = z_true,
                  z = z,
                  p = p,
                  Sigma = Sigma))
}

sim_data <- sim_mvn_binary(n = n,p = 10,mvn_dim = mvn_dim,
                           Sigma = Sigma,seed = 42)
sim_new <- sim_mvn_binary(n = n,p = 10,mvn_dim = mvn_dim,
                          Sigma = Sigma,seed = 43)

# Transforming the elements into df
df_x <- as.data.frame(sim_data$x)
df_y <- sim_data$y
df_x_new <- as.data.frame(sim_new$x)

mod <- mvnbart4(x_train = df_x,
                y_mat = df_y,
                x_test = df_x_new,m = nrow(df_x),
                var_selection_bool = TRUE,tn_sampler = FALSE,
                df = 10,n_tree = 100)


# Visualzing the variable importance
par(mfrow=c(1,mvn_dim))
for( y_j_plot in 1:mvn_dim){
     total_count <- apply(mod$var_importance[,,y_j_plot],2,sum)
     norm_count <- total_count/sum(total_count)
     names(norm_count) <- paste0("x.",1:ncol(df_x))
     sort <- sort(norm_count,decreasing = TRUE)
     barplot(sort,main = paste0("Var importance for y.",y_j_plot),las = 2)
}

# Diagonistics of the prediction over the test set
par(mfrow = c(2,mvn_dim))

for( y_j_plot in 1:mvn_dim){
     plot(sim_data$z_true[,y_j_plot],mod$y_hat_mean[,y_j_plot], pch = 20, main = paste0("y.",y_j_plot, " train pred"),
          xlab = "y.true.train" , ylab = "y.hat.train", col = ggplot2::alpha("black",0.2))
     abline(a = 0,b = 1,col = "blue", lty = 'dashed', lwd = 1.5)
}
for( y_j_plot in 1:mvn_dim){
     plot(sim_new$z_true[,y_j_plot],mod$y_hat_test_mean[,y_j_plot], pch = 20, main = paste0("y.",y_j_plot, " test pred"),
          xlab = "y.true.test" , ylab = "y.hat.test", col = ggplot2::alpha("black",0.2))
     abline(a = 0,b = 1,col = "blue", lty = 'dashed', lwd = 1.5)
}



# For the 2-dvariate case
if(mvn_dim == 2) {
     par(mfrow=c(1,3))
     plot(sqrt(mod$Sigma_post[1,1,]), main = expression(sigma[1]), type = 'l', ylab = expression(sigma[1]),xlab = "MCMC iter")
     abline(h = sqrt(Sigma[1,1]), lty = 'dashed', col = 'blue')
     plot(sqrt(mod$Sigma_post[2,2,]), main = expression(sigma[2]), type = 'l', ylab = expression(sigma[2]),xlab = "MCMC iter")
     abline(h = sqrt(Sigma[2,2]), lty = 'dashed', col = 'blue')
     plot(mod$Sigma_post[1,2,]/(sqrt(mod$Sigma_post[1,1,])*sqrt(mod$Sigma_post[2,2,])), main = expression(rho), type = 'l', ylab = expression(rho),xlab = "MCMC iter")
     abline(h = rho12, lty = 'dashed', col = 'blue')
} else if(mvn_dim ==3 ){
     par(mfrow=c(2,3))
     plot(sqrt(mod$Sigma_post[1,1,]), main = expression(sigma[1]), type = 'l', ylab = expression(sigma[1]),xlab = "MCMC iter")
     abline(h = sqrt(Sigma[1,1]), lty = 'dashed', col = 'blue')
     plot(sqrt(mod$Sigma_post[2,2,]), main = expression(sigma[2]), type = 'l', ylab = expression(sigma[2]),xlab = "MCMC iter")
     abline(h = sqrt(Sigma[2,2]), lty = 'dashed', col = 'blue')
     plot(sqrt(mod$Sigma_post[3,3,]), main = expression(sigma[3]), type = 'l', ylab = expression(sigma[3]),xlab = "MCMC iter")
     abline(h = sqrt(Sigma[3,3]), lty = 'dashed', col = 'blue')

     plot(mod$Sigma_post[1,2,]/(sqrt(mod$Sigma_post[1,1,])*sqrt(mod$Sigma_post[2,2,])), main = expression(rho[12]), type = 'l', ylab = expression(rho[12]),xlab = "MCMC iter")
     abline(h = Sigma[1,2]/(sqrt(Sigma[1,1])*sqrt(Sigma[2,2])), lty = 'dashed', col = 'blue')
     plot(mod$Sigma_post[1,3,]/(sqrt(mod$Sigma_post[1,1,])*sqrt(mod$Sigma_post[3,3,])), main = expression(rho[13]), type = 'l', ylab = expression(rho[13]),xlab = "MCMC iter")
     abline(h = Sigma[1,3]/(sqrt(Sigma[1,1])*sqrt(Sigma[3,3])), lty = 'dashed', col = 'blue')
     plot(mod$Sigma_post[2,3,]/(sqrt(mod$Sigma_post[3,3,])*sqrt(mod$Sigma_post[2,2,])), type = 'l', ylab = expression(rho[23]),xlab = "MCMC iter")
     abline(h = Sigma[2,3]/(sqrt(Sigma[2,2])*sqrt(Sigma[3,3])), lty = 'dashed', col = 'blue')

}
