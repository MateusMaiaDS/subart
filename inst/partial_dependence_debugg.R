devtools::load_all()
rm(list=ls())

n <- 1000
sim_data <- sim_mvn_friedman1(n = n,p = 5,mvn_dim = 2,Sigma = matrix(c(1,0.25,0.25,1),nrow=  2))
# x_train <- sim_data$x
# y_train <- sim_data$y
# variable_index <- 3
# use_quantiles <- FALSE
# n_points <- 10


aux <- partial_dependance_plot(variable_index = 3,use_quantiles = TRUE,n_points = 15,
                               x_train = sim_data$x,y_train = sim_data$y,
                               n_tree = 50)
