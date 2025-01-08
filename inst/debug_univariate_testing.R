rm(list=ls())
# Loading the model and testing in the univariate case
devtools::load_all()
library(dbarts)
set.seed(42)

# Generating the univariate sample
j <- 1
n <- 250
n_tree <- 100

# Generating the simulation friedman1
sim_data <- sim_mvn_friedman1(n = n,p = 10,mvn_dim = 2)
sim_data_test <- sim_mvn_friedman1(n = n,p = 10,mvn_dim = 2)
x_train <- sim_data$x
y_train <- sim_data$y[,,drop=FALSE]
true_sigma <- sim_data$Sigma[j,j]
x_test <- sim_data_test$x

# Adjusting a subart model
subart_mod <- subart(x_train = x_train,
                     y_mat = y_train,
                     x_test = x_test,
                     n_tree = n_tree,hier_prior_bool = TRUE,
                     n_burn = 0)
bart_mod <- dbarts::bart(x.train = x_train,y.train = y_train[,1],
                         x.test = x_test,ntree = n_tree,ndpost = 2000,nskip = 0)

par(mfrow=c(1,2))
plot(sqrt(subart_mod$all_Sigma_post[1,1,]),
     type='l',
     ylim = range(bart_mod$sigma,sqrt(subart_mod$all_Sigma_post)),
     main=paste0("Friedman1 || j:",j," n: ", n, "n_tree: ",n_tree),
     ylab = expression(sigma))
lines(bart_mod$sigma, col = 'blue')
abline(h = true_sigma, col = "red", lty = 2, lwd = 2)
post_index <- 1001:2000
plot(colMeans(bart_mod$yhat.test[post_index,]),apply(subart_mod$y_hat_test[,1,post_index,drop=FALSE],c(1,2),mean),
     ylab =  "subart predictions",
     xlab = "bart predictions", main =" Comparing posterior mean of pred.")
abline(a=0,b = 1)
plot(colMeans(bart_mod$yhat.test[post_index,]),sim_data_test$y_true[,1,drop=FALSE],
     ylab =  "true",
     xlab = "bart predictions", main =" Comparing posterior mean of pred.")
abline(a=0,b = 1, col = 'red', lty=2,lwd=2)
plot(apply(subart_mod$y_hat_test[,1,post_index,drop=FALSE],c(1,2),mean),sim_data_test$y_true[,1,drop=FALSE],
     ylab =  "true",
     xlab = "subart predictions", main =" Comparing posterior mean of pred.")
abline(a=0,b = 1, col = 'red', lty=2,lwd=2)


# # Running all arguments as default
# run_arguments <- function(){
#      y_mat <- y_train
#      x_test <- x_train
#      n_tree = 50
#      node_min_size = 5
#      n_mcmc = 2000
#      n_burn = 500
#      alpha = 0.95
#      beta = 2
#      nu = 3
#      sigquant = 0.9
#      kappa = 2
#      numcut = 100L # Defining the grid of split rules
#      usequants = FALSE
#      m = 20 # Degrees of freed for the classification setting.
#      varimportance = TRUE
#      hier_prior_bool = TRUE # Use a hierachical prior or not;
#      specify_variables = NULL # Specify variables for each dimension (j) by name or index for.
#      diagnostic = TRUE
# }
