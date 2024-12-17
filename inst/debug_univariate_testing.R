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
sim_data <- sim_mvn_friedman1(n = 250,p = 10,mvn_dim = 2,Sigma = diag(nrow=2))
x_train <- sim_data$x
y_train <- sim_data$y[,j,drop=FALSE]
true_sigma <- sim_data$Sigma[j,j]

# Adjusting a subart model
subart_mod <- subart(x_train = x_train,
                     y_mat = y_train,
                     x_test = x_train,
                     n_tree = n_tree,hier_prior_bool = TRUE)
bart_mod <- dbarts::bart(x.train = x_train,y.train = y_train,x.test = x_train,ntree = n_tree,ndpost = 2000,nskip = 0)
sqrt(subart_mod$all_Sigma_post[1,1,]) %>% plot(type='l',ylim = range(bart_mod$sigma,sqrt(subart_mod$all_Sigma_post)))
lines(bart_mod$sigma, col = 'blue')

# Running all arguments as default
run_arguments <- function(){
     y_mat <- y_train
     x_test <- x_train
     n_tree = 50
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
     hier_prior_bool = TRUE # Use a hierachical prior or not;
     specify_variables = NULL # Specify variables for each dimension (j) by name or index for.
     diagnostic = TRUE
}
