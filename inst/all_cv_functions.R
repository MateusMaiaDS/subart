## This file stores all CV functions for each one of the models: BART, subart, mvBART
cv_matrix_bart <- function(cv_element_,
                           n_tree_,
                           mvn_dim_,
                           n_,
                           p_,
                           i,
                           task_,
                           n_mcmc_,
                           n_burn_){
  
  
  library(subart)
  
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
  
  n_ <- nrow(x_train)
  crps_pred_post_sample_train_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  crps_pred_post_sample_test_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  
  
  # Generating the BART model
  for(i_ in 1:mvn_dim_){
    
    bart_models[[i_]] <- subart::subart(x_train = x_train,y_mat = y_train[,i_,drop = FALSE],
                                        x_test = x_test,n_tree = n_tree_,n_mcmc = n_mcmc_,
                                        n_burn = n_burn_)
    
    # Storing different metrics depending on the task
    if(task_=="regression"){
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                 value =  rmse(x = bart_models[[i_]]$y_hat_mean,
                                                                               y = y_true_train[,i_]),
                                                                 model = "BART",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                 value =  rmse(x = bart_models[[i_]]$y_hat_test_mean,
                                                                               y = y_true_test[,i_]),
                                                                 model = "BART",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      
      for(ii in 1:n_){
        # for(i_j in 1:mvn_dim_){
        crps_pred_post_sample_train_bart[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = bart_models[[i_]]$y_hat[ii,1,])
        crps_pred_post_sample_test_bart[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = bart_models[[i_]]$y_hat_test[ii,1,])
        # }
      }
      
      # scoringRules::crps_sample(y = y_true_train,bart_models[[i_]]$yhat.train)
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                 value = mean(crps_pred_post_sample_train_bart[,i_]),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                 value = mean(crps_pred_post_sample_test_bart[,i_]),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                 value = pi_coverage(y = y_test[,i_],
                                                                                     y_hat_post = (bart_models[[i_]]$y_hat_test[,1,]),
                                                                                     sd_post = sqrt(bart_models[[i_]]$Sigma_post[1,1,]),
                                                                                     prob = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                 value = pi_coverage(y = y_train[,i_],
                                                                                     y_hat_post = (bart_models[[i_]]$y_hat[,1,]),
                                                                                     sd_post = sqrt(bart_models[[i_]]$Sigma_post[1,1,]),
                                                                                     prob = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                 value = ci_coverage(y_ = y_true_test[,i_],
                                                                                     y_hat_ = bart_models[[i_]]$y_hat_test[,1,],
                                                                                     sd_ = mean(sqrt(bart_models[[i_]]$Sigma_post[1,1,])),
                                                                                     prob_ = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                 value = ci_coverage(y_ = y_true_train[,i_],
                                                                                     y_hat_ = bart_models[[i_]]$y_hat_mean[,1],
                                                                                     sd_ = mean(bart_models[[i_]]$sigma),
                                                                                     prob_ = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                                 value = cr_coverage(f_true = y_true_train[,i_],
                                                                                     f_post = (bart_models[[i_]]$y_hat[,1,]),
                                                                                     prob = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                                 value = cr_coverage(f_true = y_true_test[,i_],
                                                                                     f_post = (bart_models[[i_]]$y_hat_test[,1,]),
                                                                                     prob = 0.5),
                                                                 model  = "BART", fold = i,
                                                                 mvn_dim = i_))
    } else if( task_ == "classification") {
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_train",
                                                                 value = logloss(y_true = y_true_train[,i_],
                                                                                 y_hat = colMeans(pnorm(t(bart_models[[i_]]$y_hat[,1,])))),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_test",
                                                                 value = logloss(y_true = y_true_test[,i_],
                                                                                 y_hat = colMeans(pnorm(t(bart_models[[i_]]$y_hat_test[,1,])))),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_train",
                                                                 value = brierscore(y_true = y_true_train[,i_],
                                                                                    y_hat = colMeans(pnorm(t(bart_models[[i_]]$y_hat[,1,])))),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_test",
                                                                 value = brierscore(y_true = y_true_test[,i_],
                                                                                    y_hat = colMeans(pnorm(t(bart_models[[i_]]$y_hat_test[,1,])))),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_train",
                                                                 value = acc(y_true = y_true_train[,i_],
                                                                             y_hat = ifelse(colMeans(pnorm(t(bart_models[[i_]]$y_hat[,1,])))>0.5,1,0)),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_test",
                                                                 value = acc(y_true = y_true_test[,i_],
                                                                             y_hat = ifelse(colMeans(pnorm(t(bart_models[[i_]]$y_hat_test[,1,])))>0.5,1,0)),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_train",
                                                                 value = mcc(y_true = y_true_train[,i_],
                                                                             y_hat = ifelse(colMeans(pnorm(t(bart_models[[i_]]$y_hat[,1,])))>0.5,1,0)),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_test",
                                                                 value = mcc(y_true = y_true_test[,i_],
                                                                             y_hat = ifelse(colMeans(pnorm(t(bart_models[[i_]]$y_hat_test[,1,])))>0.5,1,0)),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_train",
                                                                 value = cr_coverage(f_true = z_true_train[,i_],
                                                                                     f_post = (bart_models[[i_]]$y_hat[,1,]),
                                                                                     prob = 0.5),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_test",
                                                                 value = cr_coverage(f_true = z_true_test[,i_],
                                                                                     f_post = (bart_models[[i_]]$y_hat_test[,1,]),
                                                                                     prob = 0.5),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_train",
                                                                 value = cr_coverage(f_true = p_true_train[,i_],
                                                                                     f_post = (pnorm(bart_models[[i_]]$y_hat[,1,])),
                                                                                     prob = 0.5),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_test",
                                                                 value = cr_coverage(f_true = p_true_test[,i_],
                                                                                     f_post = (pnorm(bart_models[[i_]]$y_hat_test[,1,])),
                                                                                     prob = 0.5),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
    } else {
      stop("Insert a valid task.")
    }
    
    
    # Doing for the main sigma parameters
    if(task_=="regression"){
      sigma_ <- sqrt(Sigma_[i_,i_])
      sigma_post <- sqrt(bart_models[[i_]]$Sigma_post[1,1,])
      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                   value = cr_coverage(f_true = sigma_,
                                                                                       f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                       prob = 0.5),
                                                                   model = "BART",
                                                                   mvn_dim = mvn_dim_,
                                                                   param_index = paste0("sigma",i_,i_,collapse = ""),
                                                                   fold = i))
      
      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                   value = rmse(x = mean(sigma_post),y = sigma_),
                                                                   model = 'BART',
                                                                   mvn_dim = mvn_dim_,
                                                                   param_index = paste0("sigma",i_,i_,collapse = ""),
                                                                   fold = i))
      
    }
  }
  
  
  sigma_list <- if(task_=="regression"){
    lapply(bart_models, function(x){sqrt(x$Sigma_post[1,1,])})
  } else {
    NULL
  }
  
  return(list(comparison_metrics = comparison_metrics,
              correlation_metrics = correlation_metrics,
              sigma = sigma_list))
  
}

cv_matrix_bart_old <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               task_,
                               n_mcmc_,
                               n_burn_){
  
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
  
  n_ <- nrow(x_train)
  crps_pred_post_sample_train_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  crps_pred_post_sample_test_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  
  
  # Generating the BART model
  for(i_ in 1:mvn_dim_){
    
    bart_models[[i_]] <- bart(x.train = x_train,y.train = y_train[,i_],
                              x.test = x_test,ntree = n_tree_,
                              ndpost = (n_mcmc_-n_burn_),nskip = n_burn_)
    
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
      
      
      for(ii in 1:n_){
        # for(i_j in 1:mvn_dim_){
        crps_pred_post_sample_train_bart[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = bart_models[[i_]]$yhat.train[,ii])
        crps_pred_post_sample_test_bart[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = bart_models[[i_]]$yhat.test[,ii])
        # }
      }
      
      # scoringRules::crps_sample(y = y_true_train,bart_models[[i_]]$yhat.train)
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                 value = mean(crps_pred_post_sample_train_bart[,i_]),
                                                                 model = "BART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                 value = mean(crps_pred_post_sample_test_bart[,i_]),
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
    
    
    # Doing for the main sigma parameters
    if(task_=="regression"){
      sigma_ <- sqrt(Sigma_[i_,i_])
      sigma_post <- bart_models[[i_]]$sigma
      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                   value = cr_coverage(f_true = sigma_,
                                                                                       f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                       prob = 0.5),
                                                                   model = "BART",
                                                                   mvn_dim = mvn_dim_,
                                                                   param_index = paste0("sigma",i_,i_,collapse = ""),
                                                                   fold = i))
      
      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                   value = rmse(x = mean(sigma_post),y = sigma_),
                                                                   model = 'BART',
                                                                   mvn_dim = mvn_dim_,
                                                                   param_index = paste0("sigma",i_,i_,collapse = ""),
                                                                   fold = i))
      
    }
  }
  
  
  sigma_list <- if(task_=="regression"){
    lapply(bart_models, function(x){x$sigma})
  } else {
    NULL
  }
  
  sigma_list <- lapply(bart_models, function(x){x$sigma})
  
  return(list(comparison_metrics = comparison_metrics,
              correlation_metrics = correlation_metrics,
              sigma = sigma_list))
  
}


## This file stores all CV functions for each one of the models: BART, subart, mvBART
cv_matrix_subart <- function(cv_element_,
                             n_tree_,
                             mvn_dim_,
                             n_,
                             p_,
                             i,
                             task_,
                             n_mcmc_,
                             n_burn_){
  
  # LOADING LIBRARIES
  library(subart)
  
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
  
  n_ <- nrow(x_train)
  crps_pred_post_sample_train_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  crps_pred_post_sample_test_bart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
  
  
  # Doing the same for the MVN-BART
  if(task_ == "regression"){
    subart_mod <- subart::subart(x_train = x_train,y_mat = y_train,x_test = x_test,
                                 n_tree = n_tree_,n_mcmc = n_mcmc_,n_burn = n_burn_) # Maybe need to change the df to 10
  } else if(task_ == "classification"){
    
    if(ncol(y_train)==2){
      subart_mod <- subart::subart(x_train = x_train,y_mat = y_train,x_test = x_test,m = nrow(x_train)/10,
                                   n_tree = n_tree_,n_mcmc = n_mcmc_,n_burn = n_burn_)
    } else {
      subart_mod <- subart::subart(x_train = x_train,y_mat = y_train,x_test = x_test,m = nrow(x_train)/2,
                                   n_tree = n_tree_,n_mcmc = n_mcmc_,n_burn = n_burn_)
    }
  }
  
  
  
  
  if(task_ == "regression"){
    # New calculation of CRPS
    crps_pred_post_sample_train_subart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    crps_pred_post_sample_test_subart <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    
    for(ii in 1:n_){
      
      for(i_ in 1:mvn_dim_){
        crps_pred_post_sample_train_subart[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = subart_mod$y_hat[ii,i_,])
        crps_pred_post_sample_test_subart[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = subart_mod$y_hat_test[ii,i_,])
      }
    }
  }
  
  # Generating metrics accordingly to the task
  if(task_ == "regression"){
    
    
    # Generating the mvnBART model
    for(i_ in 1:mvn_dim_){
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                 value =  rmse(x = subart_mod$y_hat_mean[,i_],
                                                                               y = y_true_train[,i_]),
                                                                 model = "suBART",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                 value =  rmse(x = subart_mod$y_hat_test_mean[,i_],
                                                                               y = y_true_test[,i_]),
                                                                 model = "suBART",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                 value = mean(crps_pred_post_sample_train_subart[,i_]),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                 value = mean(crps_pred_post_sample_test_subart[,i_]),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                 value = pi_coverage(y = y_test[,i_],
                                                                                     y_hat_post = (subart_mod$y_hat_test[,i_,]),
                                                                                     sd_post = sqrt(subart_mod$Sigma_post[i_,i_,]),
                                                                                     prob = 0.5),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                 value = pi_coverage(y = y_train[,i_],
                                                                                     y_hat_post =  (subart_mod$y_hat[,i_,]),
                                                                                     sd_post = sqrt(subart_mod$Sigma_post[i_,i_,]),
                                                                                     prob = 0.5,n_mcmc_replications = 100),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                 value = ci_coverage(y_ = y_true_test[,i_],
                                                                                     y_hat_ = subart_mod$y_hat_test_mean[,i_],
                                                                                     sd_ = mean(sqrt(subart_mod$Sigma_post[i_,i_,])),
                                                                                     prob_ = 0.5),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                 value = ci_coverage(y_ = y_true_train[,i_],
                                                                                     y_hat_ =  subart_mod$y_hat_mean[,i_],
                                                                                     sd_ = mean(sqrt(subart_mod$Sigma_post[i_,i_,])),
                                                                                     prob_ = 0.5),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                                 value = cr_coverage(f_true = y_true_test[,i_],
                                                                                     f_post = (subart_mod$y_hat_test[,i_,]),
                                                                                     prob = 0.5),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                                 value = cr_coverage(f_true = y_true_train[,i_],
                                                                                     f_post =  (subart_mod$y_hat[,i_,]),
                                                                                     prob = 0.5),
                                                                 model  = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
    }
    
    
    if(mvn_dim_== 2) {
      
      # Doing for the correlation parameters
      rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
      rho_post <- subart_mod$Sigma_post[1,2,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[2,2,]))
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_,
                                                                                      f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_post),y = rho_),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      # Doing for the main sigma parameters
      for(jj_ in 1:mvn_dim_){
        sigma_ <- sqrt(Sigma_[jj_,jj_])
        sigma_post <- sqrt(subart_mod$Sigma_post[jj_,jj_,])
        correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = sigma_,
                                                                                         f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                         prob = 0.5),
                                                                     model = "suBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                     fold = i))
        
        correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                     value = rmse(x = mean(sigma_post),y = sigma_),
                                                                     model = "suBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                     fold = i))
      }
      
    } else if(mvn_dim_== 3 ) {
      
      # Comparing the true values for the \rho12, \rho13, and \rho23
      rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
      rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
      rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
      rho_12_post <- subart_mod$Sigma_post[1,2,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[2,2,]))
      rho_13_post <- subart_mod$Sigma_post[1,3,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[3,3,]))
      rho_23_post <- subart_mod$Sigma_post[2,3,]/(sqrt(subart_mod$Sigma_post[2,2,])*sqrt(subart_mod$Sigma_post[3,3,]))
      
      
      # Storing the correlations
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_12,
                                                                                      f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_13,
                                                                                      f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho13",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho13",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_23,
                                                                                      f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho23",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho23",
                                                                  fold = i))
      
      # Getting the metrics for the true sigmas
      for(jj_ in 1:mvn_dim_){
        
        sigma_ <- sqrt(Sigma_[jj_,jj_])
        sigma_post <- sqrt(subart_mod$Sigma_post[jj_,jj_,])
        correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
                                                                     value = cr_coverage(f_true = sigma_,
                                                                                         f_post = matrix(sigma_post,ncol = length(sigma_post)),
                                                                                         prob = 0.5),
                                                                     model = "suBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                     fold = i))
        
        correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
                                                                     value = rmse(x = mean(sigma_post),y = sigma_),
                                                                     model = "suBART",
                                                                     mvn_dim = mvn_dim_,
                                                                     param_index = paste0("sigma",jj_,jj_,collapse = ""),
                                                                     fold = i))
      }
      
      
      
    }
    
  } else if (task_ == "classification"){ # Considering metrics regarding the classification context
    
    for( i_ in 1:mvn_dim_){ # Generating the metrics regarding each dimension
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_train",
                                                                 value = logloss(y_true = y_true_train[,i_],
                                                                                 y_hat = rowMeans(pnorm(subart_mod$y_hat[,i_,]))),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "logloss_test",
                                                                 value = logloss(y_true = y_true_test[,i_],
                                                                                 y_hat = rowMeans(pnorm(subart_mod$y_hat_test[,i_,]))),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_train",
                                                                 value = brierscore(y_true = y_true_train[,i_],
                                                                                    y_hat = rowMeans(pnorm(subart_mod$y_hat[,i_,]))),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "brier_test",
                                                                 value = brierscore(y_true = y_true_test[,i_],
                                                                                    y_hat = rowMeans(pnorm(subart_mod$y_hat_test[,i_,]))),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_train",
                                                                 value = acc(y_true = y_true_train[,i_],
                                                                             y_hat = ifelse(rowMeans(pnorm(subart_mod$y_hat[,i_,]))>0.5,1,0)),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "acc_test",
                                                                 value = acc(y_true = y_true_test[,i_],
                                                                             y_hat = ifelse(rowMeans(pnorm(subart_mod$y_hat_test[,i_,]))>0.5,1,0)),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_train",
                                                                 value = mcc(y_true = y_true_train[,i_],
                                                                             y_hat = ifelse(rowMeans(pnorm(subart_mod$y_hat[,i_,]))>0.5,1,0)),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "mcc_test",
                                                                 value = mcc(y_true = y_true_test[,i_],
                                                                             y_hat = ifelse(rowMeans(pnorm(subart_mod$y_hat_test[,i_,]))>0.5,1,0)),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_train",
                                                                 value = cr_coverage(f_true = z_true_train[,i_],
                                                                                     f_post = (subart_mod$y_hat[,i_,]),
                                                                                     prob = 0.5),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "z_cr_test",
                                                                 value = cr_coverage(f_true = z_true_test[,i_],
                                                                                     f_post = (subart_mod$y_hat_test[,i_,]),
                                                                                     prob = 0.5),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_train",
                                                                 value = cr_coverage(f_true = p_true_train[,i_],
                                                                                     f_post = pnorm((subart_mod$y_hat[,i_,])),
                                                                                     prob = 0.5),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
      # Calculating uncertainty metrics regarding Z (i.e: credible intervals)
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "p_cr_test",
                                                                 value = cr_coverage(f_true = p_true_test[,i_],
                                                                                     f_post = pnorm((subart_mod$y_hat_test[,i_,])),
                                                                                     prob = 0.5),
                                                                 model = "suBART", fold = i,
                                                                 mvn_dim = i_))
      
    } # Generating the metric regarding each mvn_component
    
    # Storing the correlation metrics
    if(mvn_dim_== 2) {
      
      # Doing for the correlation parameters
      rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
      rho_post <- subart_mod$Sigma_post[1,2,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[2,2,]))
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_,
                                                                                      f_post = matrix(rho_post,ncol = length(rho_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_post),y = rho_),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
    } else if(mvn_dim_== 3 ) {
      
      # Comparing the true values for the \rho12, \rho13, and \rho23
      rho_12 <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
      rho_13 <- Sigma_[1,3]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[3,3]))
      rho_23 <- Sigma_[2,3]/(sqrt(Sigma_[2,2])*sqrt(Sigma_[3,3]))
      rho_12_post <- subart_mod$Sigma_post[1,2,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[2,2,]))
      rho_13_post <- subart_mod$Sigma_post[1,3,]/(sqrt(subart_mod$Sigma_post[1,1,])*sqrt(subart_mod$Sigma_post[3,3,]))
      rho_23_post <- subart_mod$Sigma_post[2,3,]/(sqrt(subart_mod$Sigma_post[2,2,])*sqrt(subart_mod$Sigma_post[3,3,]))
      
      
      # Storing the correlations
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_12,
                                                                                      f_post = matrix(rho_12_post,ncol = length(rho_12_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_12_post),y = rho_12),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho12",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_13,
                                                                                      f_post = matrix(rho_13_post,ncol = length(rho_13_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho13",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_13_post),y = rho_13),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho13",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "cr_cov",
                                                                  value = cr_coverage(f_true = rho_23,
                                                                                      f_post = matrix(rho_23_post,ncol = length(rho_23_post)),prob = 0.5),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho23",
                                                                  fold = i))
      
      correlation_metrics <- rbind(correlation_metrics,data.frame(metric = "rmse",
                                                                  value = rmse(x = mean(rho_23_post),y = rho_23),
                                                                  model = "suBART",
                                                                  mvn_dim = mvn_dim_,
                                                                  param_index = "rho23",
                                                                  fold = i))
      
    }
    
  }
  
  return(list(comparison_metrics = comparison_metrics,
              correlation_metrics = correlation_metrics,
              sigma = subart_mod$all_Sigma_post))
}

## This file stores all CV functions for each one of the models: BART, subart, mvBART
cv_matrix_bayesSUR <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               task_,
                               n_mcmc_,
                               n_burn_){
  
  library(surbayes)
  
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
  
  n_ <- nrow(x_train)
  
  
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
      names(eqSystem) <- NULL
    } else {
      
      # Recreate a data.frame in the shape of the single dataset.
      colnames(y_train) <- colnames(y_test) <- paste0("y.",1:mvn_dim_)
      train_data <- cbind(x_train,y_train)
      eq1 <- as.formula( paste0("y.1 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
      eq2 <- as.formula( paste0("y.2 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
      eq3 <- as.formula( paste0("y.3 ~ ", paste0("X",1:NCOL(x_test),collapse = "+")) )
      eqSystem <- list( y.1 = eq1, y.2 = eq2, y.3 = eq3)
      names(eqSystem) <- NULL
      
      
    }
    
    # Loading package
    library(systemfit)
    library(surbayes)
    
    # Doing the predictions with the SUR model
    # sur_mod <- systemfit(eqSystem, method = "SUR", data =  train_data)
    sur_mod <- surbayes::sur_sample(formula.list = eqSystem,
                                    data = train_data,
                                    M = (n_mcmc_-n_burn_))
    
    # x111 <- x_test[1,,drop = FALSE]
    # y111 <- y_test[1,,drop = FALSE]
    
    # Doing for the first dimension
    surmod_test_predict <- predict(sur_mod,newdata = cbind(x_test,y_test),nsims = (n_mcmc_-n_burn_))
    surmod_train_predict <- predict(sur_mod,newdata = cbind(x_train,y_train),nsims = (n_mcmc_-n_burn_))
    
    # Storing only the first observations
    surmod_test_predict <- surmod_test_predict[1:nrow(x_test),,]
    surmod_train_predict <- surmod_train_predict[1:nrow(x_test),,]
    
    for(i_aux in 1:mvn_dim_){
      aux_counter <- c(0,1:(mvn_dim_-1))
      surmod_test_predict[,i_aux,] <- tcrossprod(as.matrix(cbind(1,x_test)),as.matrix(sur_mod$betadraw[,11*aux_counter[i_aux]+1:11]))
      surmod_train_predict[,i_aux,] <- tcrossprod(as.matrix(cbind(1,x_train)),as.matrix(sur_mod$betadraw[,11*aux_counter[i_aux]+1:11]))
      
    }
    
    
    
    # New calculation of CRPS
    n_ <- sur_mod$n
    crps_pred_post_sample_train <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    crps_pred_post_sample_test <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    crps_pred_post_sample_train_f <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    crps_pred_post_sample_test_f <- matrix(data = NA,nrow = n_,ncol = mvn_dim_)
    
    for(ii in 1:n_){
      
      for(i_ in 1:mvn_dim_){
        crps_pred_post_sample_train[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = surmod_train_predict[ii,i_,])
        crps_pred_post_sample_test[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = surmod_test_predict[ii,i_,])
        
      }
    }
    
    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist),dim = c(mvn_dim_,mvn_dim_,(n_mcmc_-n_burn_)))
    
    
    # Generating the BART model
    for(i_ in 1:mvn_dim_){
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                                 value =  rmse(x = apply(surmod_train_predict[,i_,],1,mean),
                                                                               y = y_true_train[,i_]),
                                                                 model = "bayesSUR",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                                 value =  rmse(x = apply(surmod_test_predict[,i_,],1,mean),
                                                                               y = y_true_test[,i_]),
                                                                 model = "bayesSUR",
                                                                 fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                                 value = mean(crps_pred_post_sample_train[,i_]),
                                                                 model = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                                 value = mean(crps_pred_post_sample_test[,i_]),
                                                                 model = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                                 value = pi_coverage(y = y_train[,i_],
                                                                                     y_hat_post = (surmod_train_predict[,i_,]),
                                                                                     sd_post = sqrt(sur_mod_sigma_array[i_,i_,]),
                                                                                     prob = 0.5),
                                                                 model  = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                                 value = pi_coverage(y = y_test[,i_],
                                                                                     y_hat_post = (surmod_test_predict[,i_,]),
                                                                                     sd_post = sqrt(sur_mod_sigma_array[i_,i_,]),
                                                                                     prob = 0.5),
                                                                 model  = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                                 value = ci_coverage(y_ = y_true_train[,i_],
                                                                                     y_hat_ = colMeans(surmod_train_predict[,i_,]),
                                                                                     sd_ = sqrt(unlist(lapply(sur_mod$Sigmalist,function(x){x[i_,i_]}))),
                                                                                     prob_ = 0.5),
                                                                 model  = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                                 value = ci_coverage(y_ = y_true_test[,i_],
                                                                                     y_hat_ = colMeans(surmod_test_predict[,i_,]),
                                                                                     sd_ = sqrt(unlist(lapply(sur_mod$Sigmalist,function(x){x[i_,i_]}))),
                                                                                     prob_ = 0.5),
                                                                 model  = "bayesSUR", fold = i,
                                                                 mvn_dim = i_))
      
      
    }
    
    
  } # End of the if to (task_=="regression") for the SUR
  
  # Storing the correlation metrics
  if(mvn_dim_== 2) {
    
    # Doing for the correlation parameters
    rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist),dim = c(2,2,2000))
    rho_post <- sur_mod_sigma_array[1,2,]/(sqrt(sur_mod_sigma_array[1,1,])*sqrt(sur_mod_sigma_array[2,2,]))
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
    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist),dim = c(3,3,2000))
    
    rho_12_post <- sur_mod_sigma_array[1,2,]/(sqrt(sur_mod_sigma_array[1,1,])*sqrt(sur_mod_sigma_array[2,2,]))
    rho_13_post <- sur_mod_sigma_array[1,3,]/(sqrt(sur_mod_sigma_array[1,1,])*sqrt(sur_mod_sigma_array[3,3,]))
    rho_23_post <- sur_mod_sigma_array[2,3,]/(sqrt(sur_mod_sigma_array[2,2,])*sqrt(sur_mod_sigma_array[3,3,]))
    
    
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
  
  
  if(mvn_dim_==2){
    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist),dim = c(2,2,2000))
  } else if (mvn_dim_==3) {
    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist),dim = c(3,3,2000))
  }
  
  # Doing for the main sigma parameters
  for(jj_ in 1:mvn_dim_){
    sigma_ <- sqrt(Sigma_[jj_,jj_])
    sigma_post <- sqrt(sur_mod_sigma_array[jj_,jj_,])
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
  
  
  
  # Return the cross-validation matrix
  return(list(comparison_metrics = comparison_metrics,
              correlation_metrics = correlation_metrics,
              sigma = sur_mod_sigma_array))
  
  
}

## This file stores all CV functions for each one of the models: BART, subart, mvBART
cv_matrix_skewBART <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               task_,
                               n_mcmc_,
                               n_burn_){
  
  library(skewBART)
  
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
  
  
  hypers <- skewBART::Hypers(as.matrix(x_train), as.matrix(y_train),
                             num_tree = n_tree_*ncol(y_train))
  
  opts <- skewBART::Opts(num_burn = n_burn_, num_save = (n_mcmc_-n_burn_), update_Sigma_mu = FALSE,
                         update_s = FALSE, update_alpha = FALSE)
  
  fitted_Multiskewbart <- skewBART::MultiskewBART(X = x_train, Y = y_train, test_X = x_test,
                                                  do_skew = FALSE,hypers = hypers,opts = opts)
  
  # =============
  
  # New calculation of CRPS
  n_mcmc_ <- dim(fitted_Multiskewbart$y_hat_train)[3]
  n_ <- nrow(fitted_Multiskewbart$y_hat_train)
  crps_pred_post_sample_train <- matrix(data = NA,nrow = n_,ncol = 2)
  crps_pred_post_sample_test <- matrix(data = NA,nrow = n_,ncol = 2)
  crps_pred_post_sample_train_f <- matrix(data = NA,nrow = n_,ncol = 2)
  crps_pred_post_sample_test_f <- matrix(data = NA,nrow = n_,ncol = 2)
  
  for(ii in 1:n_){
    
    for(i_ in 1:2){
      crps_pred_post_sample_train[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = fitted_Multiskewbart$y_hat_train[ii,i_,])
      crps_pred_post_sample_test[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = fitted_Multiskewbart$y_hat_test[ii,i_,])
      
      crps_pred_post_sample_train_f[ii, i_] <- scoringRules::crps_sample(y_true_train[ii,i_],dat = fitted_Multiskewbart$f_hat_train[ii,i_,])
      crps_pred_post_sample_test_f[ii,i_] <- scoringRules::crps_sample(y_true_test[ii,i_],dat = fitted_Multiskewbart$f_hat_test[ii,i_,])
    }
  }
  
  for(i_ in 1:2){
    
    # USING THE f as the output
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_train",
                                                               value =  rmse(x = fitted_Multiskewbart$f_hat_train_mean[,i_],
                                                                             y = y_true_train[,i_]),
                                                               model = "mvBART",
                                                               fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "rmse_test",
                                                               value =  rmse(x = fitted_Multiskewbart$f_hat_test_mean[,i_],
                                                                             y = y_true_test[,i_]),
                                                               model = "mvBART",
                                                               fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_train",
                                                               value = mean(crps_pred_post_sample_train_f[,i_]),
                                                               model = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "crps_test",
                                                               value = mean(crps_pred_post_sample_test_f[,i_]),
                                                               model = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_test",
                                                               value = pi_coverage(y = y_test[,i_],
                                                                                   y_hat_post = (fitted_Multiskewbart$f_hat_test[,i_,]),
                                                                                   sd_post = sqrt(fitted_Multiskewbart$Sigma[i_,i_,]),
                                                                                   prob = 0.5),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "pi_train",
                                                               value = pi_coverage(y = y_train[,i_],
                                                                                   y_hat_post =  (fitted_Multiskewbart$f_hat_train[,i_,]),
                                                                                   sd_post = sqrt(fitted_Multiskewbart$Sigma[i_,i_,]),
                                                                                   prob = 0.5,n_mcmc_replications = 100),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_test",
                                                               value = ci_coverage(y_ = y_true_test[,i_],
                                                                                   y_hat_ = fitted_Multiskewbart$f_hat_test_mean[,i_],
                                                                                   sd_ = mean(sqrt(fitted_Multiskewbart$Sigma[i_,i_,])),
                                                                                   prob_ = 0.5),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "ci_train",
                                                               value = ci_coverage(y_ = y_true_train[,i_],
                                                                                   y_hat_ =  fitted_Multiskewbart$f_hat_train_mean[,i_],
                                                                                   sd_ = mean(sqrt(fitted_Multiskewbart$Sigma[i_,i_,])),
                                                                                   prob_ = 0.5),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_test",
                                                               value = cr_coverage(f_true = y_true_test[,i_],
                                                                                   f_post = (fitted_Multiskewbart$f_hat_test[,i_,]),
                                                                                   prob = 0.5),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
    comparison_metrics <- rbind(comparison_metrics, data.frame(metric = "cr_train",
                                                               value = cr_coverage(f_true = y_true_train[,i_],
                                                                                   f_post =  (fitted_Multiskewbart$f_hat_train[,i_,]),
                                                                                   prob = 0.5),
                                                               model  = "mvBART", fold = i,
                                                               mvn_dim = i_))
    
  }
  # Doing for the correlation parameters
  rho_ <- Sigma_[1,2]/(sqrt(Sigma_[1,1])*sqrt(Sigma_[2,2]))
  rho_post <- fitted_Multiskewbart$Sigma[1,2,]/(sqrt(fitted_Multiskewbart$Sigma[1,1,])*sqrt(fitted_Multiskewbart$Sigma[2,2,]))
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
    sigma_post <- sqrt(fitted_Multiskewbart$Sigma[jj_,jj_,])
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
  
  return(list(comparison_metrics = comparison_metrics,
              correlation_metrics = correlation_metrics,
              sigma = fitted_Multiskewbart$Sigma))
  
  
}


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
              correlation_metrics = correlation_metrics,
              sigma = stan_samples_class$Omega))
  
}


# Creating the function for the STAN code
# Generating the model results
cv_matrix_stan_mvn <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               stan_model_regression,
                               task_,
                               n_mcmc_,
                               n_burn_){
  
  # Stan SUR ####
  library(rstan)
  if(task_=="regression"){
    stop("Do not run a STAN model anymore. Use surbayes package instead for the regression task.")
  }
  
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
      iter = n_mcmc_,
      warmup = n_burn_
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
      iter = n_mcmc_,
      warmup = n_burn_
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
    
    
    # Doing for the main sigma parameters
    # ================================================================================================
    # NOT NECESSARY THE STAN CODE IS ONLY FOR CLASSIFICATION AND THE MAIN DIAGONAL IS FIXED AT 1!!!
    # ================================================================================================
    # for(jj_ in 1:mvn_dim_){
    #      sigma_ <- sqrt(Sigma_[jj_,jj_])
    #      sigma_post <- sqrt(stan_samples_class$Omega[,jj_,jj_])
    #      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = "cr_cov",
    #                                                                   value = cr_coverage(f_true = sigma_,
    #                                                                                       f_post = matrix(sigma_post,ncol = length(sigma_post)),
    #                                                                                       prob = 0.5),
    #                                                                   model = "bayesSUR",
    #                                                                   mvn_dim = mvn_dim_,
    #                                                                   param_index = paste0("sigma",jj_,jj_,collapse = ""),
    #                                                                   fold = i))
    #
    #      correlation_metrics <- rbind(correlation_metrics, data.frame(metric = 'rmse',
    #                                                                   value = rmse(x = mean(sigma_post),y = sigma_),
    #                                                                   model = 'bayesSUR',
    #                                                                   mvn_dim = mvn_dim_,
    #                                                                   param_index = paste0("sigma",jj_,jj_,collapse = ""),
    #                                                                   fold = i))
    # }
    
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
              correlation_metrics = correlation_metrics,
              sigma = stan_samples_class$Omega))
  
}

