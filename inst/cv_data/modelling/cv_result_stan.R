# Generating the model results
rm(list=ls())
library(doParallel)
devtools::load_all()
set.seed(42)
n_ <- 250
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 2
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman2" # For this can be either 'friedman1' or 'friedman2'


# Printing whcih model is being generated
cat("n_", n_, "p_" , p_, "tree", n_tree_, "mvn_dim", mvn_dim_, "task", task_, "sim " , sim_)

# It was run to test at first
n_rep <- 100
cv_ <- vector("list",length = n_rep)

if(task_ == "regression" & sim_ == "friedman1"){
     for(rep in 1:n_rep){
          cv_[[rep]]$train <- sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
          cv_[[rep]]$test <- sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
     }
} else if (task_ == "regression" & sim_ == "friedman2"){
     for(rep in 1:n_rep){
          cv_[[rep]]$train <- sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
          cv_[[rep]]$test <- sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
     }
}

# define model - this should be done outside the simulation loop
stan_code_regression <- "
        data {
          int<lower=1> K;
          int<lower=1> J;
          int<lower=0> N;
          array[N] vector[J] x_train;
          array[N] vector[J] x_test;
          array[N] vector[K] y;
        }
        parameters {
          matrix[K, J] beta;
          cholesky_factor_corr[K] L_Omega;
          vector<lower=0>[K] L_sigma;
        }
        model {
          array[N] vector[K] mu;
          matrix[K, K] L_Sigma;

          for (n in 1:N) {
            mu[n] = beta * x_train[n];

          }

          L_Sigma = diag_pre_multiply(L_sigma, L_Omega);

          to_vector(beta) ~ normal(0, 5);
          L_Omega ~ lkj_corr_cholesky(4);
          L_sigma ~ cauchy(0, 2.5);

          y ~ multi_normal_cholesky(mu, L_Sigma);
        }
        generated quantities{
          array[N] vector[K] y_hat_train;
          array[N] vector[K] y_hat_test;
          for (n in 1:N) {
            y_hat_train[n] = beta * x_train[n];
            y_hat_test[n] = beta * x_test[n];
          }
          matrix[K, K] Sigma;
          Sigma = multiply_lower_tri_self_transpose(diag_pre_multiply(L_sigma, L_Omega));
        }
        "

# compile model - this should be done outside the simulation loop
stan_model_regression <- stan_model(model_code = stan_code_regression)


# Running inside the function
i <- 1
cv_element_ <- cv_[[i]]
