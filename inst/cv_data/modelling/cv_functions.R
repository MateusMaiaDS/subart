# Creating the function to be used in paralllel and generate the results
cv_matrix <- function(cv_element_,
                      n_tree_,
                      mvn_dim_,
                      n_,
                      p_,
                      i){

     # LOADING LIBRARIES
     library(dbarts)

     # Getting the data elements
     x_train <- cv_element_$train$x
     y_train <- cv_element_$train$y
     x_test <- cv_element_$test$x
     y_test <- cv_element_$test$y
     y_true_train <- cv_element_$train$y_true
     y_true_test <- cv_element_$test$y_true

     # True Sigma element
     Sigma_ <- cv_element_$train$Sigma

     # Creating a list with multiple models for
     bart_models <- vector("list",mvn_dim_)

     # Generating the crossvalidaiton
     comparison_metrics <- data.frame(metric = NULL,
                                      value = NULL,
                                      model = NULL,
                                      mvn_dim = NULL,
                                      fold = NULL)
     # Generating the BART model
     for(i_ in 1:mvn_dim_){
          bart_models[[i_]] <- bart(x.train = x_train,y.train = y_train[,i_],
                                             x.test = x_test,ntree = n_tree_,
                                             ndpost = 2000,nskip = 500,)


          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                     value =  rmse(x = bart_models[[i_]]$yhat.train.mean,
                                                                                   y = y_true_train[,i_]),
                                                                     model = "BART",
                                                                     fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                     value =  rmse(x = bart_models[[i_]]$yhat.test.mean,
                                                                                   y = y_true_test[,i_]),
                                                                     model = "BART",
                                                                     fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                     value =  rmse(x = bart_models[[i_]]$yhat.test.mean,
                                                                                   y = y_true_test[,i_]),
                                                                     model = "BART",
                                                                     fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                     value = crps(y = y_true_train[,i_],
                                                                                  means = bart_models[[i_]]$yhat.train.mean,
                                                                                  sds = rep(mean(bart_models[[i_]]$sigma),nrow(y_true_train)))$CRPS,
                                                                     model = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                     value = crps(y = y_true_test[,i_],
                                                                                  means = bart_models[[i_]]$yhat.test.mean,
                                                                                  sds = rep(mean(bart_models[[i_]]$sigma),nrow(y_true_test)))$CRPS,
                                                                     model = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                     value = pi_coverage(y = y_test[,i_],
                                                                                         y_hat_post = t(bart_models[[i_]]$yhat.test),
                                                                                         sd_post = bart_models[[i_]]$sigma,
                                                                                         prob = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                     value = pi_coverage(y = y_train[,i_],
                                                                                         y_hat_post = t(bart_models[[i_]]$yhat.train),
                                                                                         sd_post = bart_models[[i_]]$sigma,
                                                                                         prob = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                     value = ci_coverage(y_ = y_true_test[,i_],
                                                                                         y_hat_ = bart_models[[i_]]$yhat.test.mean,
                                                                                         sd_ = mean(bart_models[[i_]]$sigma),
                                                                                         prob_ = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                     value = ci_coverage(y_ = y_true_train[,i_],
                                                                                         y_hat_ = bart_models[[i_]]$yhat.train.mean,
                                                                                         sd_ = mean(bart_models[[i_]]$sigma),
                                                                                         prob_ = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                                     value = cr_coverage(f_true = y_true_train[,i_],
                                                                                         f_post = t(bart_models[[i_]]$yhat.train),
                                                                                         prob = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                                     value = cr_coverage(f_true = y_true_test[,i_],
                                                                                         f_post = t(bart_models[[i_]]$yhat.test),
                                                                                         prob = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))





     }

     # Doing the same for the MVN-BART
     mvbart_mod <- mvnbart(x_train = x_train,y_mat = y_train,x_test = x_test,
                           n_tree = n_tree_,n_mcmc = 2500,n_burn = 500,df = 10)

     # Generating the mvnBART model
     for(i_ in 1:mvn_dim_){

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                        value =  rmse(x = mvbart_mod$y_hat_mean[,i_],
                                                                                      y = y_true_train[,i_]),
                                                                        model = "mvBART",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                        value =  rmse(x = mvbart_mod$y_hat_test_mean[,i_],
                                                                                      y = y_true_test[,i_]),
                                                                        model = "mvBART",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                        value = crps(y = y_true_train[,i_],
                                                                                     means = mvbart_mod$y_hat_mean[,i_],
                                                                                     sds = rep(mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),nrow(y_true_train)))$CRPS,
                                                                        model = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                        value = crps(y = y_true_test[,i_],
                                                                                     means = mvbart_mod$y_hat_test_mean[,i_],
                                                                                     sds = rep(mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),nrow(y_true_test)))$CRPS,
                                                                        model = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                        value = pi_coverage(y = y_test[,i_],
                                                                                            y_hat_post = (mvbart_mod$y_hat_test[,i_,]),
                                                                                            sd_post = sqrt(mvbart_mod$Sigma_post[i_,i_,]),
                                                                                            prob = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                        value = pi_coverage(y = y_train[,i_],
                                                                                            y_hat_post =  (mvbart_mod$y_hat[,i_,]),
                                                                                            sd_post = sqrt(mvbart_mod$Sigma_post[i_,i_,]),
                                                                                            prob = 0.5,n_mcmc_replications = 100),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))


             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                        value = ci_coverage(y_ = y_true_test[,i_],
                                                                                            y_hat_ = mvbart_mod$y_hat_test_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                        value = ci_coverage(y_ = y_true_train[,i_],
                                                                                            y_hat_ =  mvbart_mod$y_hat_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                                        value = cr_coverage(f_true = y_true_test[,i_],
                                                                                            f_post = (mvbart_mod$y_hat_test[,i_,]),
                                                                                            prob = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                                        value = cr_coverage(f_true = y_true_train[,i_],
                                                                                            f_post =  (mvbart_mod$y_hat[,i_,]),
                                                                                            prob = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))




     }

     # Fittng the Fitting the Linear SUR
     if(mvn_dim_==2){

             # Recreate a data.frame in the shape of the single dataset.
             colnames(y_train) <- colnames(y_test) <- paste0("y.",1:mvn_dim_)
             train_data <- cbind(x_train,y_train)
             test_data <- cbind(x_test,y_test)
             eq1 <- as.formula( paste0("y.1 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
             eq2 <- as.formula( paste0("y.2 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
             eqSystem <- list( y.1 = eq1, y.2 = eq2)

     } else {


             # Recreate a data.frame in the shape of the single dataset.
             colnames(y_train) <- colnames(y_test) <- paste0("y.",1:mvn_dim_)
             train_data <- cbind(x_train,y_train)
             eq1 <- as.formula( paste0("y.1 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
             eq2 <- as.formula( paste0("y.2 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
             eq3 <- as.formula( paste0("y.3 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )

             eqSystem <- list( y.1 = eq1, y.2 = eq2, y.3 = eq3)


     }

     # Loading package
     library(systemfit)
     # Doing the predictions with the SUR model
     sur_mod <- systemfit(eqSystem, method = "SUR", data =  train_data)
     surmod_test_predict <- predict(sur_mod,x_test)
     surmod_train_predict <- predict(sur_mod,x_train)

     # Generating the BART model
     for(i_ in 1:mvn_dim_){

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                        value =  rmse(x = surmod_train_predict[,i_],
                                                                                      y = y_true_train[,i_]),
                                                                        model = "SUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                        value =  rmse(x = surmod_test_predict[,i_],
                                                                                      y = y_true_test[,i_]),
                                                                        model = "SUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                        value = crps(y = y_true_train[,i_],
                                                                                     means = surmod_train_predict[,i_],
                                                                                     sds = rep(sqrt(sur_mod$residCovEst[i_,i_]),nrow(y_train)))$CRPS,
                                                                        model = "SUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                        value = crps(y = y_true_test[,i_],
                                                                                     means = surmod_test_predict[,i_],
                                                                                     sds = rep(sqrt(sur_mod$residCovEst[i_,i_]),nrow(y_test)))$CRPS,
                                                                        model = "SUR", fold = i,
                                                                        mvn_dim = i_))


             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                        value = ci_coverage(y_ = y_true_train[,i_],
                                                                                            y_hat_ = surmod_train_predict[,i_],
                                                                                            sd_ = sqrt(sur_mod$residCovEst[i_,i_]),
                                                                                            prob_ = 0.5),
                                                                        model  = "SUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                        value = ci_coverage(y_ = y_true_test[,i_],
                                                                                            y_hat_ = surmod_test_predict[,i_],
                                                                                            sd_ = sqrt(sur_mod$residCovEst[i_,i_]),
                                                                                            prob_ = 0.5),
                                                                        model  = "SUR", fold = i,
                                                                        mvn_dim = i_))


     }



     # Return the cross-validation matrix
     return(comparison_metrics)

}


# Creating the function for the STAN code
# Generating the model results
stan_mvn <- function(cv_element_,
                     n_tree_,
                     mvn_dim_,
                     n_,
                     p_,
                     i){

        # Stan SUR ####
        library(rstan)

        # Getting the data elements
        x_train <- cv_element_$train$x
        y_train <- cv_element_$train$y
        x_test <- cv_element_$test$x
        y_test <- cv_element_$test$y
        y_true_train <- cv_element_$train$y_true
        y_true_test <- cv_element_$test$y_true

        # True Sigma element
        Sigma_ <- cv_element_$train$Sigma

        # Generating the crossvalidaiton
        comparison_metrics <- data.frame(metric = NULL,
                                         value = NULL,
                                         model = NULL,
                                         mvn_dim = NULL,
                                         fold = NULL)

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

        # everything from now on should be done inside the simulation loop
        # define data - here I guess the first 1 in each line should be replaced by some replication index
        stan_data <- list(
                x_train = x_train,
                x_test = x_test,
                y = y_train,
                N = n_,
                J = p_,
                K = mvn_dim_
        )

        # run sampler
        stan_fit_regression <- sampling(
                object = stan_model_regression,
                data = stan_data,
                pars = c("y_hat_train","y_hat_test","Sigma"),
                include = TRUE,
                chains = 1,
                iter = 2500,
                warmup = 500
        )

        # Extracting some posterior samples
        stan_samples_regression <- rstan::extract(stan_fit_regression)
        stan_samples_regression$y_hat_train_mean <- apply(stan_samples_regression$y_hat_train, c(2:3), mean)
        stan_samples_regression$y_hat_test_mean <- apply(stan_samples_regression$y_hat_test, c(2:3), mean)


      # Generating the bayesian-SUR model
      for(i_ in 1:mvn_dim_){

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                        value =  rmse(x = stan_samples_regression$y_hat_train_mean[,i_],
                                                                                      y = y_true_train[,i_]),
                                                                        model = "bayesSUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                        value =  rmse(x = stan_samples_regression$y_hat_test_mean[,i_],
                                                                                      y = y_true_test[,i_]),
                                                                        model = "bayesSUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                        value = crps(y = y_true_train[,i_],
                                                                                     means = stan_samples_regression$y_hat_train_mean[,i_],
                                                                                     sds = mean(sqrt(stan_samples_regression$Sigma[,i_,i_])))$CRPS,
                                                                        model = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                        value = crps(y = y_true_test[,i_],
                                                                                     means =  stan_samples_regression$y_hat_test_mean[,i_],
                                                                                     sds = mean(sqrt(stan_samples_regression$Sigma[,i_,i_])))$CRPS,
                                                                        model = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                        value = pi_coverage(y = y_test[,i_],
                                                                                            y_hat_post = t(stan_samples_regression$y_hat_test[,,i]),
                                                                                            sd_post = sqrt(stan_samples_regression$Sigma[,i_,i_]),
                                                                                            prob = 0.5),
                                                                        model  = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                        value = pi_coverage(y = y_train[,i_],
                                                                                            y_hat_post =  t(stan_samples_regression$y_hat_train[,,i]),
                                                                                            sd_post = sqrt(stan_samples_regression$Sigma[,i_,i_]),
                                                                                            prob = 0.5),
                                                                        model  = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))


             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                        value = ci_coverage(y_ = y_true_test[,i_],
                                                                                            y_hat_ = stan_samples_regression$y_hat_test_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
                                                                        model  = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                        value = ci_coverage(y_ = y_true_train[,i_],
                                                                                            y_hat_ =  stan_samples_regression$y_hat_train_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
                                                                        model  = "bayesSUR", fold = i,
                                                                        mvn_dim = i_))

     }


     return(comparison_metrics)

}



