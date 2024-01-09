# Creating the function to be used in paralllel and generate the results
cv_matrix <- function(cv_element_,
                      n_tree_,
                      mvn_dim_,
                      n_,
                      p_,
                      i,
                      task_){

     # LOADING LIBRARIES
     library(dbarts)

     # Getting the data elements
     x_train <- cv_element_$train$x
     y_train <- cv_element_$train$y
     x_test <- cv_element_$test$x
     y_test <- cv_element_$test$y
     y_true_train <- cv_element_$train$y_true
     y_true_test <- cv_element_$test$y_true

     if(task_ == "classification"){
       z_true_train <- cv_element_$train$z_true
       z_true_test <- cv_element_$test$z_true
       z_train <- cv_element_$train$z
       z_test <- cv_element_$test$z
       p_true_train <- pnorm(cv_element_$train$z_true)
       p_true_test <- pnorm(cv_element_$test$z_true)
     }

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

     # Creating the data.frame for the correlation parameters
     correlation_metrics <- data.frame(metric = NULL,
                                       value = NULL,
                                       model = NULL,
                                       mvn_dim = NULL,
                                       param_index = NULL,
                                       fold = NULL)

     # Generating the BART model
     for(i_ in 1:mvn_dim_){
          bart_models[[i_]] <- bart(x.train = x_train,y.train = y_train[,i_],
                                             x.test = x_test,ntree = n_tree_,
                                             ndpost = 2000,nskip = 500)

          # Storing different metrics depending on the task
          if(task_=="regression"){
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
          } else if( task_ == "classification") {

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_train",
                                                                       value = logloss(y_true = y_true_train[,i_],
                                                                                       y_hat = colMeans(pnorm(bart_models[[i_]]$yhat.train))),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_test",
                                                                       value = logloss(y_true = y_true_test[,i_],
                                                                                       y_hat = colMeans(pnorm(bart_models[[i_]]$yhat.test))),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_train",
                                                                       value = brierscore(y_true = y_true_train[,i_],
                                                                                       y_hat = colMeans(pnorm(bart_models[[i_]]$yhat.train))),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_test",
                                                                       value = brierscore(y_true = y_true_test[,i_],
                                                                                       y_hat = colMeans(pnorm(bart_models[[i_]]$yhat.test))),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))


            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_train",
                                                                       value = acc(y_true = y_true_train[,i_],
                                                                                  y_hat = ifelse(colMeans(pnorm(bart_models[[i_]]$yhat.train))>0.5,1,0)),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_test",
                                                                       value = acc(y_true = y_true_test[,i_],
                                                                                          y_hat = ifelse(colMeans(pnorm(bart_models[[i_]]$yhat.test))>0.5,1,0)),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_train",
                                                                       value = mcc(y_true = y_true_train[,i_],
                                                                                   y_hat = ifelse(colMeans(pnorm(bart_models[[i_]]$yhat.train))>0.5,1,0)),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_test",
                                                                       value = mcc(y_true = y_true_test[,i_],
                                                                                   y_hat = ifelse(colMeans(pnorm(bart_models[[i_]]$yhat.test))>0.5,1,0)),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_train",
                                                                       value = cr_coverage(f_true = z_true_train[,i_],
                                                                                           f_post = t(bart_models[[i_]]$yhat.train),
                                                                                           prob = 0.5),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_test",
                                                                       value = cr_coverage(f_true = z_true_test[,i_],
                                                                                           f_post = t(bart_models[[i_]]$yhat.test),
                                                                                           prob = 0.5),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_train",
                                                                       value = cr_coverage(f_true = p_true_train[,i_],
                                                                                           f_post = t(pnorm(bart_models[[i_]]$yhat.train)),
                                                                                           prob = 0.5),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_test",
                                                                       value = cr_coverage(f_true = p_true_test[,i_],
                                                                                           f_post = t(pnorm(bart_models[[i_]]$yhat.test)),
                                                                                           prob = 0.5),
                                                                       model = "BART", fold = i,
                                                                       mvn_dim = i_))

          } else {
            stop("Insert a valid task.")
        }


     }

     # Doing the same for the MVN-BART
     if(task_ == "regression"){
       mvbart_mod <- mvnbart(x_train = x_train,y_mat = y_train,x_test = x_test,
                             n_tree = n_tree_,n_mcmc = 2500,n_burn = 500,df = 10)
     } else if(task_ == "classification"){
       mvbart_mod <- mvnbart(x_train = x_train,y_mat = y_train,x_test = x_test,m = nrow(x_train),
                             n_tree = n_tree_,n_mcmc = 2500,n_burn = 500,df = 2)
     }

     # Generating metrics accordingly to the task
     if(task_ == "regression"){

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


             if(mvn_dim_== 2) {

               # Doing for the correlation parameters
               rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
               rho_post <- mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,]))
               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = rho_,
                                                                                               f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho12",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                           value = rmse(x = mean(rho_post),y = rho_),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho12",
                                                                           fold = i))

               # Doing for the main sigma parameters
               for(jj_ in 1:mvn_dim_){
                 sigma_ <- sqrt(Sigma_[jj_,jj_])
                 sigma_post <- sqrt(mvbart_mod$Sigma_post[jj_,jj_,])
                 correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                              value = cr_coverage(f_true = sigma_,
                                                                                                  f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                                  prob = 0.5),
                                                                              model = "mvBART",
                                                                              mvn_dim = mvn_dim_,
                                                                              param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                              fold = i))

                 correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                              value = rmse(x = mean(sigma_post),y = sigma_),
                                                                              model = 'mvBART',
                                                                              mvn_dim = mvn_dim_,
                                                                              param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                              fold = i))
               }

             } else if(mvn_dim_== 3 ) {

               # Comparing the true values for the \rho12, \rho13, and \rho23
               rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
               rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
               rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
               rho_12_post <- mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,]))
               rho_13_post <- mvbart_mod$Sigma_post[1,3,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[3,3,]))
               rho_23_post <- mvbart_mod$Sigma_post[2,3,]/(sqrt(mvbart_mod$Sigma_post[2,2,])*sqrt(mvbart_mod$Sigma_post[3,3,]))


               # Storing the correlations
               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = rho_12,
                                                                                               f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho12",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                           value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho12",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = rho_13,
                                                                                               f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho13",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                           value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho13",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = rho_23,
                                                                                               f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho23",
                                                                           fold = i))

               correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                           value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                           model = "mvBART",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = "rho23",
                                                                           fold = i))

              # Getting the metrics for the true sigmas
               for(jj_ in 1:mvn_dim_){

                 sigma_ <- sqrt(Sigma_[jj_,jj_])
                 sigma_post <- sqrt(mvbart_mod$Sigma_post[jj_,jj_,])
                 correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                              value = cr_coverage(f_true = sigma_,
                                                                                                  f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                                  prob = 0.5),
                                                                              model = "mvBART",
                                                                              mvn_dim = mvn_dim_,
                                                                              param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                              fold = i))

                 correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                              value = rmse(x = mean(sigma_post),y = sigma_),
                                                                              model = 'mvBART',
                                                                              mvn_dim = mvn_dim_,
                                                                              param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                              fold = i))
               }



             }

     } else if (task_ == "classification"){ # Considering metrics regarding the classification context

       for( i_ in 1:mvn_dim_){ # Generating the metrics regarding each dimension
           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_train",
                                                                      value = logloss(y_true = y_true_train[,i_],
                                                                                      y_hat = rowMeans(pnorm(mvbart_mod$y_hat[,i_,]))),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_test",
                                                                      value = logloss(y_true = y_true_test[,i_],
                                                                                      y_hat = rowMeans(pnorm(mvbart_mod$y_hat_test[,i_,]))),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_train",
                                                                      value = brierscore(y_true = y_true_train[,i_],
                                                                                         y_hat = rowMeans(pnorm(mvbart_mod$y_hat[,i_,]))),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_test",
                                                                      value = brierscore(y_true = y_true_test[,i_],
                                                                                         y_hat = rowMeans(pnorm(mvbart_mod$y_hat_test[,i_,]))),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))


           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_train",
                                                                      value = acc(y_true = y_true_train[,i_],
                                                                                  y_hat = ifelse(rowMeans(pnorm(mvbart_mod$y_hat[,i_,]))>0.5,1,0)),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_test",
                                                                      value = acc(y_true = y_true_test[,i_],
                                                                                  y_hat = ifelse(rowMeans(pnorm(mvbart_mod$y_hat_test[,i_,]))>0.5,1,0)),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_train",
                                                                      value = mcc(y_true = y_true_train[,i_],
                                                                                  y_hat = ifelse(rowMeans(pnorm(mvbart_mod$y_hat[,i_,]))>0.5,1,0)),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_test",
                                                                      value = mcc(y_true = y_true_test[,i_],
                                                                                  y_hat = ifelse(rowMeans(pnorm(mvbart_mod$y_hat_test[,i_,]))>0.5,1,0)),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_train",
                                                                      value = cr_coverage(f_true = z_true_train[,i_],
                                                                                          f_post = (mvbart_mod$y_hat[,i_,]),
                                                                                          prob = 0.5),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_test",
                                                                      value = cr_coverage(f_true = z_true_test[,i_],
                                                                                          f_post = (mvbart_mod$y_hat_test[,i_,]),
                                                                                          prob = 0.5),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_train",
                                                                      value = cr_coverage(f_true = p_true_train[,i_],
                                                                                          f_post = pnorm((mvbart_mod$y_hat[,i_,])),
                                                                                          prob = 0.5),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

           # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
           comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_test",
                                                                      value = cr_coverage(f_true = p_true_test[,i_],
                                                                                          f_post = pnorm((mvbart_mod$y_hat_test[,i_,])),
                                                                                          prob = 0.5),
                                                                      model = "mvBART", fold = i,
                                                                      mvn_dim = i_))

       } # Generating the metric regarding each mvn_component
       # Storing the correlation metrics
       if(mvn_dim_== 2) {

         # Doing for the correlation parameters
         rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
         rho_post <- mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,]))
         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = rho_,
                                                                                         f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho12",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                     value = rmse(x = mean(rho_post),y = rho_),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho12",
                                                                     fold = i))

       } else if(mvn_dim_== 3 ) {

         # Comparing the true values for the \rho12, \rho13, and \rho23
         rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
         rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
         rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
         rho_12_post <- mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[2,2,]))
         rho_13_post <- mvbart_mod$Sigma_post[1,3,]/(sqrt(mvbart_mod$Sigma_post[1,1,])*sqrt(mvbart_mod$Sigma_post[3,3,]))
         rho_23_post <- mvbart_mod$Sigma_post[2,3,]/(sqrt(mvbart_mod$Sigma_post[2,2,])*sqrt(mvbart_mod$Sigma_post[3,3,]))


         # Storing the correlations
         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = rho_12,
                                                                                         f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho12",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                     value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho12",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = rho_13,
                                                                                         f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho13",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                     value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho13",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = rho_23,
                                                                                         f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho23",
                                                                     fold = i))

         correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                     value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                     model = "mvBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = "rho23",
                                                                     fold = i))

       }

     }

     # The frequentist SUR should be used only for the regression approach.
     if(task_ == "regression"){
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


     } # End of the if to (task_=="regression") for the SUR


     # Return the cross-validation matrix
     return(list(comparison_metrics = comparison_metrics,
                 correlation_metrics = correlation_metrics))

}


# Creating the function for the STAN code
# Generating the model results
stan_mvn <- function(cv_element_,
                     n_tree_,
                     mvn_dim_,
                     n_,
                     p_,
                     i,
                     stan_model_regression,
                     task_){

        # Stan SUR ####
        library(rstan)

        # Getting the data elements
        x_train <- cv_element_$train$x
        y_train <- cv_element_$train$y
        x_test <- cv_element_$test$x
        y_test <- cv_element_$test$y
        y_true_train <- cv_element_$train$y_true
        y_true_test <- cv_element_$test$y_true


        # Getting the true model
        if(task_ == "classification"){
          z_true_train <- cv_element_$train$z_true
          z_true_test <- cv_element_$test$z_true
          z_train <- cv_element_$train$z
          z_test <- cv_element_$test$z
          p_true_train <- pnorm(cv_element_$train$z_true)
          p_true_test <- pnorm(cv_element_$test$z_true)
        }


        # True Sigma element
        Sigma_ <- cv_element_$train$Sigma

        # Generating the crossvalidaiton
        comparison_metrics <- data.frame(metric = NULL,
                                         value = NULL,
                                         model = NULL,
                                         mvn_dim = NULL,
                                         fold = NULL)

        # Creating the data.frame for the correlation parameters
        correlation_metrics <- data.frame(metric = NULL,
                                          value = NULL,
                                          model = NULL,
                                          mvn_dim = NULL,
                                          param_index = NULL,
                                          fold = NULL)

        # everything from now on should be done inside the simulation loop
        # define data - here I guess the first 1 in each line should be replaced by some replication index
        if(task_ == "regression"){
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

        } else if(task_ == "classification"){
          stan_data <- list(
            x_train = x_train,
            x_test = x_test,
            y = y_train,
            N = n_,
            D = mvn_dim_,
            K = p_
          )

          # run sampler
          stan_fit_class <- sampling(
            object = stan_model_regression,
            data = stan_data,
            pars = c("z_hat_train","z_hat_test","Omega"),
            include = TRUE,
            chains = 1,
            iter = 2500,
            warmup = 500
          )

        }



        # Extracting some posterior samples
        if(task_ == "regression"){
            stan_samples_regression <- rstan::extract(stan_fit_regression)
            stan_samples_regression$y_hat_train_mean <- apply(stan_samples_regression$y_hat_train, c(2:3), mean)
            stan_samples_regression$y_hat_test_mean <- apply(stan_samples_regression$y_hat_test, c(2:3), mean)
        } else if(task_ == "classification") {
            stan_samples_class <- rstan::extract(stan_fit_class)
            stan_samples_class$z_hat_train_mean <- apply(stan_samples_class$z_hat_train, c(2:3), mean)
            stan_samples_class$z_hat_test_mean <- apply(stan_samples_class$z_hat_test, c(2:3), mean)
            stan_samples_class$p_hat_train <- pnorm(stan_samples_class$z_hat_train)
            stan_samples_class$p_hat_test <- pnorm(stan_samples_class$z_hat_test)
            stan_samples_class$p_hat_train_mean <- apply(stan_samples_class$p_hat_train, c(2:3), mean)
            stan_samples_class$p_hat_test_mean <- apply(stan_samples_class$p_hat_test, c(2:3), mean)
        } else {
          stop("insert a valid task")
        }

      if(task_== "regression"){
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
                                                                                                y_hat_post = t(stan_samples_regression$y_hat_test[,,i_]),
                                                                                                sd_post = sqrt(stan_samples_regression$Sigma[,i_,i_]),
                                                                                                prob = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))

                 comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                            value = pi_coverage(y = y_train[,i_],
                                                                                                y_hat_post =  t(stan_samples_regression$y_hat_train[,,i_]),
                                                                                                sd_post = sqrt(stan_samples_regression$Sigma[,i_,i_]),
                                                                                                prob = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))


                 comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                            value = ci_coverage(y_ = y_true_test[,i_],
                                                                                                y_hat_ = stan_samples_regression$y_hat_test_mean[,i_],
                                                                                                sd_ = mean(sqrt(stan_samples_regression$Sigma[,i_,i_])),
                                                                                                prob_ = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))

                 comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                            value = ci_coverage(y_ = y_true_train[,i_],
                                                                                                y_hat_ =  stan_samples_regression$y_hat_train_mean[,i_],
                                                                                                sd_ = mean(sqrt(stan_samples_regression$Sigma[,i_,i_])),
                                                                                                prob_ = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))


                 comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                                            value = cr_coverage(f_true = y_true_test[,i_],
                                                                                                f_post = t(stan_samples_regression$y_hat_test[,,i_]),
                                                                                                prob = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))

                 comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                                            value = cr_coverage(f_true = y_true_train[,i_],
                                                                                                f_post = t(stan_samples_regression$y_hat_train[,,i_]),
                                                                                                prob = 0.5),
                                                                            model  = "bayesSUR", fold = i,
                                                                            mvn_dim = i_))

          }

          if(mvn_dim_== 2) {

            # Doing for the correlation parameters
            rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
            rho_post <- stan_samples_regression$Sigma[,1,2]/(sqrt(stan_samples_regression$Sigma[,1,1])*sqrt(stan_samples_regression$Sigma[,2,2]))
            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                        value = cr_coverage(f_true = rho_,
                                                                                            f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho12",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                        value = rmse(x = mean(rho_post),y = rho_),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho12",
                                                                        fold = i))

            # Doing for the main sigma parameters
            for(jj_ in 1:mvn_dim_){
              sigma_ <- sqrt(Sigma_[jj_,jj_])
              sigma_post <- sqrt(stan_samples_regression$Sigma[,jj_,jj_])
              correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = sigma_,
                                                                                               f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                               prob = 0.5),
                                                                           model = "bayesSUR",
                                                                           mvn_dim = 2,
                                                                           param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                           fold = i))

              correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                           value = rmse(x = mean(sigma_post),y = sigma_),
                                                                           model = 'bayesSUR',
                                                                           mvn_dim = 2,
                                                                           param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                           fold = i))
            }
          } else if (mvn_dim_==3){

            # Comparing the true values for the \rho12, \rho13, and \rho23
            rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
            rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
            rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
            rho_12_post <- stan_samples_regression$Sigma[,1,2]/(sqrt(stan_samples_regression$Sigma[,1,1])*sqrt(stan_samples_regression$Sigma[,2,2]))
            rho_13_post <- stan_samples_regression$Sigma[,1,3]/(sqrt(stan_samples_regression$Sigma[,1,1])*sqrt(stan_samples_regression$Sigma[,3,3]))
            rho_23_post <- stan_samples_regression$Sigma[,2,3]/(sqrt(stan_samples_regression$Sigma[,2,2])*sqrt(stan_samples_regression$Sigma[,3,3]))


            # Storing the correlations
            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                        value = cr_coverage(f_true = rho_12,
                                                                                            f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho12",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                        value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho12",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                        value = cr_coverage(f_true = rho_13,
                                                                                            f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho13",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                        value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho13",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                        value = cr_coverage(f_true = rho_23,
                                                                                            f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho23",
                                                                        fold = i))

            correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                        value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                        model = "bayesSUR",
                                                                        mvn_dim = mvn_dim_,
                                                                        param_index = "rho23",
                                                                        fold = i))

            # Getting the metrics for the true sigmas
            for(jj_ in 1:mvn_dim_){

              sigma_ <- sqrt(Sigma_[jj_,jj_])
              sigma_post <- sqrt(stan_samples_regression$Sigma[,jj_,jj_])
              correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                           value = cr_coverage(f_true = sigma_,
                                                                                               f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                               prob = 0.5),
                                                                           model = "bayesSUR",
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                           fold = i))

              correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                           value = rmse(x = mean(sigma_post),y = sigma_),
                                                                           model = 'bayesSUR',
                                                                           mvn_dim = mvn_dim_,
                                                                           param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                           fold = i))
            }
          }


      } else if (task_=="classification"){
          for(i_ in 1:mvn_dim_){
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_train",
                                                                       value = logloss(y_true = y_true_train[,i_],
                                                                                       y_hat = stan_samples_class$p_hat_train_mean[,i_]),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_test",
                                                                       value = logloss(y_true = y_true_test[,i_],
                                                                                       y_hat = stan_samples_class$p_hat_test_mean[,i_]),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_train",
                                                                       value = brierscore(y_true = y_true_train[,i_],
                                                                                          y_hat = stan_samples_class$p_hat_train_mean[,i_]),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_test",
                                                                       value = brierscore(y_true = y_true_test[,i_],
                                                                                          y_hat = stan_samples_class$p_hat_test_mean[,i_]),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))


            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_train",
                                                                       value = acc(y_true = y_true_train[,i_],
                                                                                   y_hat = ifelse(stan_samples_class$p_hat_train_mean[,i_]>0.5,1,0)),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_test",
                                                                       value = acc(y_true = y_true_test[,i_],
                                                                                   y_hat = ifelse(stan_samples_class$p_hat_test_mean[,i_]>0.5,1,0)),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_train",
                                                                       value = mcc(y_true = y_true_train[,i_],
                                                                                   y_hat = ifelse(stan_samples_class$p_hat_train_mean[,i_]>0.5,1,0)),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_test",
                                                                       value = mcc(y_true = y_true_test[,i_],
                                                                                   y_hat = ifelse(stan_samples_class$p_hat_test_mean[,i_]>0.5,1,0)),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_train",
                                                                       value = cr_coverage(f_true = z_true_train[,i_],
                                                                                           f_post = t(stan_samples_class$z_hat_train[,,i_]),
                                                                                           prob = 0.5),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_test",
                                                                       value = cr_coverage(f_true = z_true_test[,i_],
                                                                                           f_post = t(stan_samples_class$z_hat_test[,,i_]),
                                                                                           prob = 0.5),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_train",
                                                                       value = cr_coverage(f_true = p_true_train[,i_],
                                                                                           f_post = t(stan_samples_class$p_hat_train[,,i_]),
                                                                                           prob = 0.5),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
            comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_test",
                                                                       value = cr_coverage(f_true = p_true_test[,i_],
                                                                                           f_post = t(stan_samples_class$p_hat_test[,,i_]),
                                                                                           prob = 0.5),
                                                                       model = "bayesSUR", fold = i,
                                                                       mvn_dim = i_))

            }

            # Storing the correlation metrics
            if(mvn_dim_== 2) {

              # Doing for the correlation parameters
              rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
              rho_post <- stan_samples_class$Omega[,1,2]/(sqrt(stan_samples_class$Omega[,1,1])*sqrt(stan_samples_class$Omega[,2,2]))
              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                          value = cr_coverage(f_true = rho_,
                                                                                              f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho12",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                          value = rmse(x = mean(rho_post),y = rho_),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho12",
                                                                          fold = i))

            } else if(mvn_dim_== 3 ) {

              # Comparing the true values for the \rho12, \rho13, and \rho23
              rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
              rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
              rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
              rho_12_post <- stan_samples_class$Omega[,1,2]/(sqrt(stan_samples_class$Omega[,1,1])*sqrt(stan_samples_class$Omega[,2,2]))
              rho_13_post <- stan_samples_class$Omega[,1,3]/(sqrt(stan_samples_class$Omega[,1,1])*sqrt(stan_samples_class$Omega[,3,3]))
              rho_23_post <- stan_samples_class$Omega[,2,3]/(sqrt(stan_samples_class$Omega[,2,2])*sqrt(stan_samples_class$Omega[,3,3]))


              # Storing the correlations
              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                          value = cr_coverage(f_true = rho_12,
                                                                                              f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho12",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                          value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho12",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                          value = cr_coverage(f_true = rho_13,
                                                                                              f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho13",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                          value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho13",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                          value = cr_coverage(f_true = rho_23,
                                                                                              f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho23",
                                                                          fold = i))

              correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                          value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                          model = "bayesSUR",
                                                                          mvn_dim = mvn_dim_,
                                                                          param_index = "rho23",
                                                                          fold = i))

            }
    }

     return(list(comparison_metrics = comparison_metrics,
                 correlation_metrics = correlation_metrics))

}



