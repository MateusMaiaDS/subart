# Loading the main functions
devtools::load_all()
rm(list=ls())
# Loading libraries
library(purrr)

# Loading the simulation files
set.seed(42)
n <- 250
sim_data <- sim_mvn_friedman1(n = n,p = 10,mvn_dim = 2)

random_index <- sample(1:nrow(sim_data$x),
                       size = 0.25*nrow(sim_data$x))
y_missing <- sim_data$y
y_missing[random_index,1] <- NA

# Storing X, Y
x_train_scale <- X <- sim_data$x
y_mat_scale <- y <- sim_data$y_true

running_all_arguments <- FALSE
if(running_all_arguments){
     # Running main arguments
     # x_train <- X
     # y_mat <- y_missing
     # x_test <- X
     n_tree = 10
     node_min_size = 5
     n_mcmc = 2000
     n_burn = 500
     alpha = 0.95
     beta = 2
     nu = 3
     sigquant = 0.9
     kappa = 2
     numcut = 100L # Defining the grid of split rules
     usequants = FALSE
     m = 20 # Degrees of freed for the classification setting.
     varimportance = TRUE
     hier_prior_bool = FALSE # Use a hierachical prior or not;
     specify_variables = NULL # Specify variables for each dimension (j) by name or index for.
     diagnostic = TRUE
     specify_variables = NULL # Specify variables for each dimension (j) by name or index for.,
     diagnostic = TRUE # Calculates the Effective Sample size for the covariance and correlation parameters

     x_train = as.data.frame(x)
     y_mat = y_obs
     x_test = as.data.frame(x)
     varimportance = FALSE
     n_mcmc = 1000
     numcut <- 100
}
# # Running subart model
subart_ig <- subart::subart(x_train = X,y_mat = y_missing,
                            x_test = X,n_tree = 50,
                            hier_prior_bool = TRUE)

# subart_t <- subart::subart(x_train = X,y_mat = y,
#                            x_test = X,n_tree = 50,
#                            hier_prior_bool = TRUE)

plot(y_mat_scale[,1],subart_ig$y_hat_mean[,1])
