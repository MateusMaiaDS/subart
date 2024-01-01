# rm(list=ls())
library(purrr)
library(devtools)

devtools::load_all()
# source("inst/test_simulations.R")
# x_train <- data_train %>% dplyr::select(dplyr::starts_with("X"))
# x_test <- data_test %>% dplyr::select(dplyr::starts_with("X"))
# y_mat <- cbind(data_train$C,data_train$Q)
# colnames(y_mat) <- c("C","Q")
# x_train <- x_test <-
n_tree = 50
n_mcmc = 2000
n_burn = 500
alpha = 0.95
beta = 2
df = 3
sigquant = 0.9
kappa = 2
# Hyperparam for tau_b and tau_b_0
numcut <- 100L
usequants <- FALSE
node_min_size <- 5
Sigma_init <- NULL
scale_y <- TRUE
update_Sigma <- TRUE
varimportance <- TRUE
specify_variables <- list(c(1,2,3),
                          c(1,4))
x_train <- df_x
y_mat <- df_y
x_test <- df_x_new
# probit_bart <- mvnbart4(x_train = x_train,y_mat = y_mat,x_test = x_test,n_tree = 100,Sigma_init = Sigma)
# table(probit_bart$y_hat_mean_class[,1],data_train$C)
# table(probit_bart$y_hat_mean_class[,2],data_train$Q)
