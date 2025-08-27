library(rstan)

stan_code_classification <- "
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

# compile model - this should be done outside the simulation loop
stan_model_regression <- stan_model(model_code = stan_code_classification)

# ========================== #
# ========================== #
# ========================== #
