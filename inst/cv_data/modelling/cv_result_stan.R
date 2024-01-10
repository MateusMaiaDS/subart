# Generating the model results
rm(list=ls())
library(doParallel)
library(rstan)
devtools::load_all()
set.seed(42)
n_ <- 1000
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 2
task_ <- "classification" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'


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
} else if(task_ == "classification" & sim_ == "friedman1"){
        for(rep in 1:n_rep){
                cv_[[rep]]$train <- sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
                cv_[[rep]]$test <- sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
        }
} else {
        stop ("Insert a valid task and simulation")
}

# define model - this should be done outside the simulation loop
if(task_ == "regression"){
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
} else if(task_ == "classification"){
        stan_code_regression <- "
        functions {
          int sum2d(array[,] int a) {
            int s = 0;
            for (i in 1:size(a)) {
              s += sum(a[i]);
            }
            return s;
          }
        }
        data {
          int<lower=1> K;
          int<lower=1> D;
          int<lower=0> N;
          array[N] vector[K] x_train;
          array[N] vector[K] x_test;
          array[N, D] int<lower=0, upper=1> y;
        }
        transformed data {
          int<lower=0> N_pos;
          array[sum2d(y)] int<lower=1, upper=N> n_pos;
          array[size(n_pos)] int<lower=1, upper=D> d_pos;
          int<lower=0> N_neg;
          array[(N * D) - size(n_pos)] int<lower=1, upper=N> n_neg;
          array[size(n_neg)] int<lower=1, upper=D> d_neg;

          N_pos = size(n_pos);
          N_neg = size(n_neg);
          {
            int i;
            int j;
            i = 1;
            j = 1;
            for (n in 1:N) {
              for (d in 1:D) {
                if (y[n, d] == 1) {
                  n_pos[i] = n;
                  d_pos[i] = d;
                  i += 1;
                } else {
                  n_neg[j] = n;
                  d_neg[j] = d;
                  j += 1;
                }
              }
            }
          }
        }
        parameters {
          matrix[D, K] beta;
          cholesky_factor_corr[D] L_Omega;
          vector<lower=0>[N_pos] z_pos;
          vector<upper=0>[N_neg] z_neg;
        }
        transformed parameters {
          array[N] vector[D] z;
          for (n in 1:N_pos) {
            z[n_pos[n], d_pos[n]] = z_pos[n];
          }
          for (n in 1:N_neg) {
            z[n_neg[n], d_neg[n]] = z_neg[n];
          }
        }
        model {
          L_Omega ~ lkj_corr_cholesky(4);
          to_vector(beta) ~ normal(0, 5);
          {
            array[N] vector[D] beta_x;
            for (n in 1:N) {
              beta_x[n] = beta * x_train[n];
            }
            z ~ multi_normal_cholesky(beta_x, L_Omega);
          }
        }
        generated quantities {
          array[N] vector[D] z_hat_train;
          array[N] vector[D] z_hat_test;
          for (n in 1:N) {
            z_hat_train[n] = beta * x_train[n];
            z_hat_test[n] = beta * x_test[n];
          }
          corr_matrix[D] Omega;
          Omega = multiply_lower_tri_self_transpose(L_Omega);
        }
        "
}
# compile model - this should be done outside the simulation loop
stan_model_regression <- stan_model(model_code = stan_code_regression)


# Running inside the function

result <- vector("list",n_rep)
source("inst/cv_data/modelling/cv_functions.R")

for(i in 1:n_rep){
        result[[i]] <- stan_mvn(cv_element_ = cv_[[i]],
                                mvn_dim_ = mvn_dim_,
                                n_tree_ = n_tree_,
                                n_ = n_,p_ = p_,
                                i = i,stan_model_regression = stan_model_regression,
                                task_ = task_)
        cat("Running stan model iteration... ", i, "\n")
        cat("n_", n_, "p_" , p_, "tree", n_tree_, "mvn_dim", mvn_dim_, "task", task_, "sim " , sim_,"\n")

}


# # Saving the results
# saveRDS(object = result, file = paste0("inst/cv_data/",task_,"/result/STAN_",sim_,"_",task_,"_n_",n_,"_p_",p_,
#                                        "_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))
