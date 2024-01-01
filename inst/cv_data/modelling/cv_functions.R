# Creating the function to be used in paralllel and generate the results
cv_matrix <- function(cv_element_,
                      ntree_,
                      mvn_dim_,
                      n_,
                      p_,
                      i){

     # Getting the data elements
     x_train <- cv_element_$train$x
     y_train <- cv_element_$train$y
     x_test <- cv_element_$test$x
     y_test <- cv_element_$test$y
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
          bart_models[[i_]] <- dbarts::bart(x.train = x_train,y.train = y_train[,i_],
                                             x.test = x_test,ntree = ntree_,
                                             ndpost = 2000,nskip = 500)
          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                     value =  rmse(x = bart_models[[i_]]$yhat.train.mean,
                                                                                   y = y_train[,i_]),
                                                                     model = "BART",
                                                                     fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                     value =  rmse(x = bart_models[[i_]]$yhat.test.mean,
                                                                                   y = y_test[,i_]),
                                                                     model = "BART",
                                                                     fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                     value = crps(y = y_train[,i_],
                                                                                  means = bart_models[[i_]]$yhat.train.mean,
                                                                                  sds = rep(mean(bart_models[[i_]]$sigma)))$CRPS,
                                                                     model = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                     value = crps(y = y_test[,i_],
                                                                                  means = bart_models[[i_]]$yhat.test.mean,
                                                                                  sds = rep(mean(bart_models[[i_]]$sigma)))$CRPS,
                                                                     model = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                     value = pi_coverage(y = y_test[,i_],
                                                                                         y_hat_post = bart_models[[i_]]$yhat.test,
                                                                                         sd_post = bart_models[[i_]]$sigma,
                                                                                         prob = 0.5,n_mcmc_replications = 100),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                     value = pi_coverage(y = y_train[,i_],
                                                                                         y_hat_post = bart_models[[i_]]$yhat.train,
                                                                                         sd_post = bart_models[[i_]]$sigma,
                                                                                         prob = 0.5,n_mcmc_replications = 100),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                     value = ci_coverage(y_ = y_test[,i_],
                                                                                         y_hat_ = bart_models[[i_]]$yhat.test.mean,
                                                                                         sd_ = mean(bart_models[[i_]]$sigma),
                                                                                         prob_ = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))

          comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                     value = ci_coverage(y_ = y_train[,i_],
                                                                                         y_hat_ = bart_models[[i_]]$yhat.train.mean,
                                                                                         sd_ = mean(bart_models[[i_]]$sigma),
                                                                                         prob_ = 0.5),
                                                                     model  = "BART", fold = i,
                                                                     mvn_dim = i_))


     }

     # Doing the same for the MVN-BART
     mvbart_mod <- mvnbart(x_train = x_train,y_mat = y_train,x_test = x_test,
                           n_tree = ntree_,n_mcmc = 2500,n_burn = 500,df = 10)

     # Generating the BART model
     for(i_ in 1:mvn_dim_){

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                        value =  rmse(x = mvbart_mod$y_hat_mean[,i_],
                                                                                      y = y_train[,i_]),
                                                                        model = "mvBART",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                        value =  rmse(x = mvbart_mod$y_hat_test_mean[,i_],
                                                                                      y = y_test[,i_]),
                                                                        model = "mvBART",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                        value = crps(y = y_train[,i_],
                                                                                     means = mvbart_mod$y_hat_mean[,i_],
                                                                                     sds = sqrt(mvbart_mod$Sigma_post[i_,i_,]))$CRPS,
                                                                        model = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                        value = crps(y = y_test[,i_],
                                                                                     means = mvbart_mod$y_hat_test_mean[,i_],
                                                                                     sds = sqrt(mvbart_mod$Sigma_post[i_,i_,]))$CRPS,
                                                                        model = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                        value = pi_coverage(y = y_test[,i_],
                                                                                            y_hat_post = t(mvbart_mod$y_hat_test[,i_,]),
                                                                                            sd_post = sqrt(mvbart_mod$Sigma_post[i_,i_,]),
                                                                                            prob = 0.5,n_mcmc_replications = 100),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                        value = pi_coverage(y = y_train[,i_],
                                                                                            y_hat_post =  t(mvbart_mod$y_hat[,i_,]),
                                                                                            sd_post = sqrt(mvbart_mod$Sigma_post[i_,i_,]),
                                                                                            prob = 0.5,n_mcmc_replications = 100),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))


             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                        value = ci_coverage(y_ = y_test[,i_],
                                                                                            y_hat_ = mvbart_mod$y_hat_test_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
                                                                        model  = "mvBART", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                        value = ci_coverage(y_ = y_train[,i_],
                                                                                            y_hat_ =  mvbart_mod$y_hat_mean[,i_],
                                                                                            sd_ = mean(sqrt(mvbart_mod$Sigma_post[i_,i_,])),
                                                                                            prob_ = 0.5),
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
                                                                                      y = y_train[,i_]),
                                                                        model = "SUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                        value =  rmse(x = surmod_test_predict[,i_],
                                                                                      y = y_test[,i_]),
                                                                        model = "SUR",
                                                                        fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                        value = crps(y = y_train[,i_],
                                                                                     means = surmod_train_predict[,i_],
                                                                                     sds = rep(sqrt(sur_mod$residCovEst[i_,i_]),nrow(y_train)))$CRPS,
                                                                        model = "SUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                        value = crps(y = y_test[,i_],
                                                                                     means = surmod_test_predict[,i_],
                                                                                     sds = rep(sqrt(sur_mod$residCovEst[i_,i_]),nrow(y_test)))$CRPS,
                                                                        model = "SUR", fold = i,
                                                                        mvn_dim = i_))

             # Creating a auxiliar for the PI (skip this for the SUR)
             # npost_ <- nrow(bart_models[[i_]]$yhat.test)
             # rep_y_hat_train <- matrix(NA, nrow = nrow(y_mat), ncol = npost_)
             # rep_y_hat_test <- matrix(NA, nrow = nrow(x_test), ncol = npost_)
             #
             # for(jj_ in 1:npost_){
             #         rep_y_hat_train[,jj_] <- surmod_train_predict[,i_]
             #         rep_y_hat_test[,jj_] <- surmod_test_predict[,i_]
             #
             # }
             # comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
             #                                                            value = pi_coverage(y = y_test[,i_],
             #                                                                                y_hat_post = t(rep_y_hat_test),
             #                                                                                sd_post = rep(sqrt(sur_mod$residCovEst[i_,i_]),npost_),    ),
             #                                                                                prob = 0.5,n_mcmc_replications = 100,only_post = TRUE),
             #                                                            model  = "SUR", fold = i,
             #                                                            mvn_dim = i_)
             #
             # comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
             #                                                            value = pi_coverage(y = y_train[,i_],
             #                                                                                y_hat_post = t(rep_y_hat),
             #                                                                                sd_post = sqrt(mvbart_mod$Sigma_post[i_,i_,]),
             #                                                                                prob = 0.5,n_mcmc_replications = 100,only_post = TRUE),
             #                                                            model  = "SUR", fold = i,
             #                                                            mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                        value = ci_coverage(y_ = y_train[,i_],
                                                                                            y_hat_ = surmod_train_predict[,i_],
                                                                                            sd_ = sqrt(sur_mod$residCovEst[i_,i_]),
                                                                                            prob_ = 0.5),
                                                                        model  = "SUR", fold = i,
                                                                        mvn_dim = i_))

             comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                        value = ci_coverage(y_ = y_test[,i_],
                                                                                            y_hat_ = surmod_test_predict[,i_],
                                                                                            sd_ = sqrt(sur_mod$residCovEst[i_,i_]),
                                                                                            prob_ = 0.5),
                                                                        model  = "SUR", fold = i,
                                                                        mvn_dim = i_))


     }

     # Return the cross-validation matrix
     return(comparison_metrics)

}
