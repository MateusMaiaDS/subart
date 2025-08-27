# Cross-validation wrapper for BART models
cv_matrix_bart <- function(cv_element_,
                           n_tree_,
                           mvn_dim_,
                           n_,
                           p_,
                           i,
                           task_,
                           n_mcmc_,
                           n_burn_) {
  library(subart)

  # Extract train/test data
  x_train <- cv_element_$train$x
  y_train <- cv_element_$train$y
  x_test <- cv_element_$test$x
  y_test <- cv_element_$test$y
  y_true_train <- cv_element_$train$y_true
  y_true_test <- cv_element_$test$y_true

  if (task_ == "classification") {
    z_true_train <- cv_element_$train$z_true
    z_true_test <- cv_element_$test$z_true
    z_train <- cv_element_$train$z
    z_test <- cv_element_$test$z
    p_true_train <- pnorm(z_true_train)
    p_true_test <- pnorm(z_true_test)
  }

  Sigma_ <- cv_element_$train$Sigma

  # Initialize storage
  bart_models <- vector("list", mvn_dim_)
  comparison_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    fold = integer()
  )
  correlation_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    param_index = character(),
    fold = integer()
  )

  n_ <- nrow(x_train)
  crps_train <- matrix(NA, n_, mvn_dim_)
  crps_test <- matrix(NA, n_, mvn_dim_)

  # Fit BART models
  for (i_ in seq_len(mvn_dim_)) {
    bart_models[[i_]] <- subart(
      x_train = x_train, y_mat = y_train[, i_, drop = FALSE],
      x_test = x_test, n_tree = n_tree_,
      n_mcmc = n_mcmc_, n_burn = n_burn_
    )

    if (task_ == "regression") {
      # RMSE metrics
      comparison_metrics <- rbind(comparison_metrics, data.frame(
        metric = c("rmse_train", "rmse_test"),
        value = c(
          rmse(bart_models[[i_]]$y_hat_mean, y_true_train[, i_]),
          rmse(bart_models[[i_]]$y_hat_test_mean, y_true_test[, i_])
        ),
        model = "BART",
        fold = i,
        mvn_dim = i_
      ))

      # CRPS per observation
      for (ii in seq_len(n_)) {
        crps_train[ii, i_] <- scoringRules::crps_sample(
          y_true_train[ii, i_],
          dat = bart_models[[i_]]$y_hat[ii, 1, ]
        )
        crps_test[ii, i_] <- scoringRules::crps_sample(
          y_true_test[ii, i_],
          dat = bart_models[[i_]]$y_hat_test[ii, 1, ]
        )
      }

      comparison_metrics <- rbind(comparison_metrics, data.frame(
        metric = c("crps_train", "crps_test"),
        value = c(mean(crps_train[, i_]), mean(crps_test[, i_])),
        model = "BART", fold = i, mvn_dim = i_
      ))

      # Interval & coverage metrics
      comparison_metrics <- rbind(comparison_metrics, data.frame(
        metric = c("pi_train", "pi_test", "ci_train", "ci_test", "cr_train", "cr_test"),
        value = c(
          pi_coverage(y_train[, i_], bart_models[[i_]]$y_hat[, 1, ], sqrt(bart_models[[i_]]$Sigma_post[1, 1, ]), 0.5),
          pi_coverage(y_test[, i_], bart_models[[i_]]$y_hat_test[, 1, ], sqrt(bart_models[[i_]]$Sigma_post[1, 1, ]), 0.5),
          ci_coverage(y_true_train[, i_], bart_models[[i_]]$y_hat_mean[, 1], mean(sqrt(bart_models[[i_]]$Sigma_post[1, 1, ])), 0.5),
          ci_coverage(y_true_test[, i_], bart_models[[i_]]$y_hat_test[, 1, ], mean(sqrt(bart_models[[i_]]$Sigma_post[1, 1, ])), 0.5),
          cr_coverage(y_true_train[, i_], bart_models[[i_]]$y_hat[, 1, ], 0.5),
          cr_coverage(y_true_test[, i_], bart_models[[i_]]$y_hat_test[, 1, ], 0.5)
        ),
        model = "BART", fold = i, mvn_dim = i_
      ))

      # Variance metrics
      sigma_ <- sqrt(Sigma_[i_, i_])
      sigma_post <- sqrt(bart_models[[i_]]$Sigma_post[1, 1, ])
      correlation_metrics <- rbind(correlation_metrics, data.frame(
        metric = c("cr_cov", "rmse"),
        value = c(
          cr_coverage(sigma_, matrix(sigma_post, ncol = length(sigma_post)), 0.5),
          rmse(mean(sigma_post), sigma_)
        ),
        model = "BART", mvn_dim = mvn_dim_, param_index = paste0("sigma", i_, i_), fold = i
      ))
    } else if (task_ == "classification") {
      # Predicted probabilities
      yhat_train_prob <- colMeans(pnorm(t(bart_models[[i_]]$y_hat[, 1, ])))
      yhat_test_prob <- colMeans(pnorm(t(bart_models[[i_]]$y_hat_test[, 1, ])))

      # Metrics
      comparison_metrics <- rbind(comparison_metrics, data.frame(
        metric = c(
          "logloss_train", "logloss_test", "brier_train", "brier_test",
          "acc_train", "acc_test", "mcc_train", "mcc_test",
          "z_cr_train", "z_cr_test", "p_cr_train", "p_cr_test"
        ),
        value = c(
          logloss(y_true_train[, i_], yhat_train_prob),
          logloss(y_true_test[, i_], yhat_test_prob),
          brierscore(y_true_train[, i_], yhat_train_prob),
          brierscore(y_true_test[, i_], yhat_test_prob),
          acc(y_true_train[, i_], ifelse(yhat_train_prob > 0.5, 1, 0)),
          acc(y_true_test[, i_], ifelse(yhat_test_prob > 0.5, 1, 0)),
          mcc(y_true_train[, i_], ifelse(yhat_train_prob > 0.5, 1, 0)),
          mcc(y_true_test[, i_], ifelse(yhat_test_prob > 0.5, 1, 0)),
          cr_coverage(z_true_train[, i_], bart_models[[i_]]$y_hat[, 1, ], 0.5),
          cr_coverage(z_true_test[, i_], bart_models[[i_]]$y_hat_test[, 1, ], 0.5),
          cr_coverage(p_true_train[, i_], pnorm(bart_models[[i_]]$y_hat[, 1, ]), 0.5),
          cr_coverage(p_true_test[, i_], pnorm(bart_models[[i_]]$y_hat_test[, 1, ]), 0.5)
        ),
        model = "BART", fold = i, mvn_dim = i_
      ))
    } else {
      stop("Invalid task.")
    }
  }

  sigma_list <- if (task_ == "regression") {
    lapply(bart_models, function(x) sqrt(x$Sigma_post[1, 1, ]))
  } else {
    NULL
  }

  list(
    comparison_metrics  = comparison_metrics,
    correlation_metrics = correlation_metrics,
    sigma               = sigma_list
  )
}



## This file stores all CV functions for BART, subart, mvBART
cv_matrix_subart <- function(cv_element_,
                             n_tree_,
                             mvn_dim_,
                             n_,
                             p_,
                             i,
                             task_,
                             n_mcmc_,
                             n_burn_) {
  library(subart)

  # Extract training and testing data
  x_train <- cv_element_$train$x
  y_train <- cv_element_$train$y
  x_test <- cv_element_$test$x
  y_test <- cv_element_$test$y
  y_true_train <- cv_element_$train$y_true
  y_true_test <- cv_element_$test$y_true
  Sigma_ <- cv_element_$train$Sigma

  if (task_ == "classification") {
    z_true_train <- cv_element_$train$z_true
    z_true_test <- cv_element_$test$z_true
    z_train <- cv_element_$train$z
    z_test <- cv_element_$test$z
    p_true_train <- pnorm(z_true_train)
    p_true_test <- pnorm(z_true_test)
  }

  # Initialize containers
  comparison_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = numeric(),
    fold = numeric()
  )

  correlation_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = numeric(),
    param_index = character(),
    fold = numeric()
  )

  n_ <- nrow(x_train)
  crps_pred_post_sample_train_bart <- matrix(NA, nrow = n_, ncol = mvn_dim_)
  crps_pred_post_sample_test_bart <- matrix(NA, nrow = n_, ncol = mvn_dim_)

  # Fit suBART model
  if (task_ == "regression") {
    subart_mod <- subart(
      x_train = x_train, y_mat = y_train, x_test = x_test,
      n_tree = n_tree_, n_mcmc = n_mcmc_, n_burn = n_burn_
    )
  } else if (task_ == "classification") {
    m_val <- ifelse(ncol(y_train) == 2, nrow(x_train) / 10, nrow(x_train) / 2)
    subart_mod <- subart(
      x_train = x_train, y_mat = y_train, x_test = x_test,
      m = m_val, n_tree = n_tree_, n_mcmc = n_mcmc_, n_burn = n_burn_
    )
  }

  # CRPS calculation for regression
  if (task_ == "regression") {
    crps_pred_post_sample_train_subart <- matrix(NA, nrow = n_, ncol = mvn_dim_)
    crps_pred_post_sample_test_subart <- matrix(NA, nrow = n_, ncol = mvn_dim_)

    for (ii in 1:n_) {
      for (i_ in 1:mvn_dim_) {
        crps_pred_post_sample_train_subart[ii, i_] <- scoringRules::crps_sample(
          y_true_train[ii, i_],
          dat = subart_mod$y_hat[ii, i_, ]
        )
        crps_pred_post_sample_test_subart[ii, i_] <- scoringRules::crps_sample(
          y_true_test[ii, i_],
          dat = subart_mod$y_hat_test[ii, i_, ]
        )
      }
    }
  }

  # Generate comparison metrics
  for (i_ in 1:mvn_dim_) {
    if (task_ == "regression") {
      comparison_metrics <- rbind(
        comparison_metrics,
        data.frame(
          metric = c(
            "rmse_train", "rmse_test", "crps_train", "crps_test",
            "pi_test", "pi_train", "ci_test", "ci_train",
            "cr_test", "cr_train"
          ),
          value = c(
            rmse(subart_mod$y_hat_mean[, i_], y_true_train[, i_]),
            rmse(subart_mod$y_hat_test_mean[, i_], y_true_test[, i_]),
            mean(crps_pred_post_sample_train_subart[, i_]),
            mean(crps_pred_post_sample_test_subart[, i_]),
            pi_coverage(y_test[, i_], subart_mod$y_hat_test[, i_, ], sqrt(subart_mod$Sigma_post[i_, i_, ]), prob = 0.5),
            pi_coverage(y_train[, i_], subart_mod$y_hat[, i_, ], sqrt(subart_mod$Sigma_post[i_, i_, ]), prob = 0.5, n_mcmc_replications = 100),
            ci_coverage(y_true_test[, i_], subart_mod$y_hat_test_mean[, i_], mean(sqrt(subart_mod$Sigma_post[i_, i_, ])), prob_ = 0.5),
            ci_coverage(y_true_train[, i_], subart_mod$y_hat_mean[, i_], mean(sqrt(subart_mod$Sigma_post[i_, i_, ])), prob_ = 0.5),
            cr_coverage(y_true_test[, i_], subart_mod$y_hat_test[, i_, ], prob = 0.5),
            cr_coverage(y_true_train[, i_], subart_mod$y_hat[, i_, ], prob = 0.5)
          ),
          model = "suBART",
          fold = i,
          mvn_dim = i_
        )
      )
    } else if (task_ == "classification") {
      y_hat_train <- rowMeans(pnorm(subart_mod$y_hat[, i_, ]))
      y_hat_test <- rowMeans(pnorm(subart_mod$y_hat_test[, i_, ]))
      y_hat_bin_train <- as.integer(y_hat_train > 0.5)
      y_hat_bin_test <- as.integer(y_hat_test > 0.5)

      comparison_metrics <- rbind(
        comparison_metrics,
        data.frame(
          metric = c(
            "logloss_train", "logloss_test", "brier_train", "brier_test",
            "acc_train", "acc_test", "mcc_train", "mcc_test",
            "z_cr_train", "z_cr_test", "p_cr_train", "p_cr_test"
          ),
          value = c(
            logloss(y_true_train[, i_], y_hat_train),
            logloss(y_true_test[, i_], y_hat_test),
            brierscore(y_true_train[, i_], y_hat_train),
            brierscore(y_true_test[, i_], y_hat_test),
            acc(y_true_train[, i_], y_hat_bin_train),
            acc(y_true_test[, i_], y_hat_bin_test),
            mcc(y_true_train[, i_], y_hat_bin_train),
            mcc(y_true_test[, i_], y_hat_bin_test),
            cr_coverage(z_true_train[, i_], subart_mod$y_hat[, i_, ], prob = 0.5),
            cr_coverage(z_true_test[, i_], subart_mod$y_hat_test[, i_, ], prob = 0.5),
            cr_coverage(p_true_train[, i_], pnorm(subart_mod$y_hat[, i_, ]), prob = 0.5),
            cr_coverage(p_true_test[, i_], pnorm(subart_mod$y_hat_test[, i_, ]), prob = 0.5)
          ),
          model = "suBART",
          fold = i,
          mvn_dim = i_
        )
      )
    }
  }

  # Correlation metrics (handling mvn_dim 2 and 3)
  if (mvn_dim_ %in% 2:3) {
    rho_indices <- combn(mvn_dim_, 2)
    for (k in 1:ncol(rho_indices)) {
      ii <- rho_indices[1, k]
      jj <- rho_indices[2, k]
      rho_true <- Sigma_[ii, jj] / (sqrt(Sigma_[ii, ii]) * sqrt(Sigma_[jj, jj]))
      rho_post <- subart_mod$Sigma_post[ii, jj, ] / (sqrt(subart_mod$Sigma_post[ii, ii, ]) * sqrt(subart_mod$Sigma_post[jj, jj, ]))
      correlation_metrics <- rbind(
        correlation_metrics,
        data.frame(
          metric = c("cr_cov", "rmse"),
          value = c(
            cr_coverage(rho_true, matrix(rho_post, ncol = length(rho_post)), prob = 0.5),
            rmse(mean(rho_post), rho_true)
          ),
          model = "suBART",
          mvn_dim = mvn_dim_,
          param_index = paste0("rho", ii, jj),
          fold = i
        )
      )
    }

    # Sigma metrics
    for (jj_ in 1:mvn_dim_) {
      sigma_true <- sqrt(Sigma_[jj_, jj_])
      sigma_post <- sqrt(subart_mod$Sigma_post[jj_, jj_, ])
      correlation_metrics <- rbind(
        correlation_metrics,
        data.frame(
          metric = c("cr_cov", "rmse"),
          value = c(
            cr_coverage(sigma_true, matrix(sigma_post, ncol = length(sigma_post)), prob = 0.5),
            rmse(mean(sigma_post), sigma_true)
          ),
          model = "suBART",
          mvn_dim = mvn_dim_,
          param_index = paste0("sigma", jj_, jj_),
          fold = i
        )
      )
    }
  }

  return(list(
    comparison_metrics = comparison_metrics,
    correlation_metrics = correlation_metrics,
    sigma = subart_mod$all_Sigma_post
  ))
}

# Cross-validation wrapper for Bayesian SUR models
cv_matrix_bayesSUR <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               task_,
                               n_mcmc_,
                               n_burn_) {
  library(surbayes)
  library(systemfit)

  # Extract train/test data
  x_train <- cv_element_$train$x
  y_train <- cv_element_$train$y
  x_test <- cv_element_$test$x
  y_test <- cv_element_$test$y
  y_true_train <- cv_element_$train$y_true
  y_true_test <- cv_element_$test$y_true

  if (task_ == "classification") {
    z_true_train <- cv_element_$train$z_true
    z_true_test <- cv_element_$test$z_true
    z_train <- cv_element_$train$z
    z_test <- cv_element_$test$z
    p_true_train <- pnorm(z_true_train)
    p_true_test <- pnorm(z_true_test)
  }

  Sigma_ <- cv_element_$train$Sigma

  # Initialize storage
  comparison_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    fold = integer()
  )
  correlation_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    param_index = character(),
    fold = integer()
  )

  n_ <- nrow(x_train)

  if (task_ == "regression") {
    # Create formulas for SUR
    colnames(y_train) <- colnames(y_test) <- paste0("y.", 1:mvn_dim_)
    train_data <- cbind(x_train, y_train)
    test_data <- cbind(x_test, y_test)

    eqSystem <- lapply(1:mvn_dim_, function(j) {
      as.formula(paste0("y.", j, " ~ ", paste0("X", 1:ncol(x_test), collapse = "+")))
    })

    # Fit Bayesian SUR
    sur_mod <- surbayes::sur_sample(
      formula.list = eqSystem,
      data = train_data,
      M = (n_mcmc_ - n_burn_)
    )

    # Predictions
    surmod_test_predict <- predict(sur_mod, newdata = test_data, nsims = (n_mcmc_ - n_burn_))[1:nrow(x_test), , ]
    surmod_train_predict <- predict(sur_mod, newdata = train_data, nsims = (n_mcmc_ - n_burn_))[1:nrow(x_train), , ]

    # Manual reconstruction using betadraw
    aux_counter <- 0:(mvn_dim_ - 1)
    for (i_aux in 1:mvn_dim_) {
      surmod_test_predict[, i_aux, ] <- tcrossprod(cbind(1, x_test), sur_mod$betadraw[, 11 * aux_counter[i_aux] + 1:11])
      surmod_train_predict[, i_aux, ] <- tcrossprod(cbind(1, x_train), sur_mod$betadraw[, 11 * aux_counter[i_aux] + 1:11])
    }

    # Compute CRPS
    crps_train <- matrix(NA, n_, mvn_dim_)
    crps_test <- matrix(NA, n_, mvn_dim_)
    for (ii in 1:n_) {
      for (i_ in 1:mvn_dim_) {
        crps_train[ii, i_] <- scoringRules::crps_sample(y_true_train[ii, i_], dat = surmod_train_predict[ii, i_, ])
        crps_test[ii, i_] <- scoringRules::crps_sample(y_true_test[ii, i_], dat = surmod_test_predict[ii, i_, ])
      }
    }

    sur_mod_sigma_array <- array(unlist(sur_mod$Sigmalist), dim = c(mvn_dim_, mvn_dim_, n_mcmc_ - n_burn_))

    # Metrics for each dimension
    for (i_ in 1:mvn_dim_) {
      comparison_metrics <- rbind(comparison_metrics, data.frame(
        metric = c(
          "rmse_train", "rmse_test", "crps_train", "crps_test",
          "pi_train", "pi_test", "ci_train", "ci_test"
        ),
        value = c(
          rmse(apply(surmod_train_predict[, i_, ], 1, mean), y_true_train[, i_]),
          rmse(apply(surmod_test_predict[, i_, ], 1, mean), y_true_test[, i_]),
          mean(crps_train[, i_]),
          mean(crps_test[, i_]),
          pi_coverage(y_train[, i_], surmod_train_predict[, i_, ], sqrt(sur_mod_sigma_array[i_, i_, ]), 0.5),
          pi_coverage(y_test[, i_], surmod_test_predict[, i_, ], sqrt(sur_mod_sigma_array[i_, i_, ]), 0.5),
          ci_coverage(
            y_true_train[, i_], colMeans(surmod_train_predict[, i_, ]),
            sqrt(sapply(sur_mod$Sigmalist, function(x) x[i_, i_])), 0.5
          ),
          ci_coverage(
            y_true_test[, i_], colMeans(surmod_test_predict[, i_, ]),
            sqrt(sapply(sur_mod$Sigmalist, function(x) x[i_, i_])), 0.5
          )
        ),
        model = "bayesSUR",
        fold = i,
        mvn_dim = i_
      ))
    }

    # Correlation metrics
    rho_indices <- combn(mvn_dim_, 2)
    for (k in 1:ncol(rho_indices)) {
      idx <- rho_indices[, k]
      rho_true <- Sigma_[idx[1], idx[2]] / (sqrt(Sigma_[idx[1], idx[1]]) * sqrt(Sigma_[idx[2], idx[2]]))
      rho_post <- sur_mod_sigma_array[idx[1], idx[2], ] / (sqrt(sur_mod_sigma_array[idx[1], idx[1], ]) * sqrt(sur_mod_sigma_array[idx[2], idx[2], ]))
      correlation_metrics <- rbind(correlation_metrics, data.frame(
        metric = c("cr_cov", "rmse"),
        value = c(
          cr_coverage(rho_true, matrix(rho_post, ncol = length(rho_post)), 0.5),
          rmse(mean(rho_post), rho_true)
        ),
        model = "bayesSUR",
        mvn_dim = mvn_dim_,
        param_index = paste0("rho", idx[1], idx[2]),
        fold = i
      ))
    }

    # Sigma metrics
    for (jj_ in 1:mvn_dim_) {
      sigma_true <- sqrt(Sigma_[jj_, jj_])
      sigma_post <- sqrt(sur_mod_sigma_array[jj_, jj_, ])
      correlation_metrics <- rbind(correlation_metrics, data.frame(
        metric = c("cr_cov", "rmse"),
        value = c(
          cr_coverage(sigma_true, matrix(sigma_post, ncol = length(sigma_post)), 0.5),
          rmse(mean(sigma_post), sigma_true)
        ),
        model = "bayesSUR",
        mvn_dim = mvn_dim_,
        param_index = paste0("sigma", jj_, jj_),
        fold = i
      ))
    }
  }

  return(list(
    comparison_metrics  = comparison_metrics,
    correlation_metrics = correlation_metrics,
    sigma               = sur_mod_sigma_array
  ))
}



# Cross-validation wrapper for skewBART models
cv_matrix_skewBART <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               task_,
                               n_mcmc_,
                               n_burn_) {
  library(skewBART)

  # Extract train/test data
  x_train <- cv_element_$train$x
  y_train <- cv_element_$train$y
  x_test <- cv_element_$test$x
  y_test <- cv_element_$test$y
  y_true_train <- cv_element_$train$y_true
  y_true_test <- cv_element_$test$y_true

  if (task_ == "classification") {
    z_true_train <- cv_element_$train$z_true
    z_true_test <- cv_element_$test$z_true
    z_train <- cv_element_$train$z
    z_test <- cv_element_$test$z
    p_true_train <- pnorm(z_true_train)
    p_true_test <- pnorm(z_true_test)
  }

  Sigma_ <- cv_element_$train$Sigma

  # Initialize storage
  comparison_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    fold = integer()
  )
  correlation_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    param_index = character(),
    fold = integer()
  )

  # Setup hypers and options for skewBART
  hypers <- skewBART::Hypers(as.matrix(x_train), as.matrix(y_train),
    num_tree = n_tree_ * ncol(y_train)
  )
  opts <- skewBART::Opts(
    num_burn = n_burn_, num_save = (n_mcmc_ - n_burn_),
    update_Sigma_mu = FALSE, update_s = FALSE, update_alpha = FALSE
  )

  # Fit MultiskewBART
  fitted <- skewBART::MultiskewBART(
    X = x_train, Y = y_train, test_X = x_test,
    do_skew = FALSE, hypers = hypers, opts = opts
  )

  n_samples <- dim(fitted$y_hat_train)[3]
  n_obs <- nrow(fitted$y_hat_train)

  # Compute CRPS
  crps_train <- matrix(NA, n_obs, mvn_dim_)
  crps_test <- matrix(NA, n_obs, mvn_dim_)
  crps_train_f <- matrix(NA, n_obs, mvn_dim_)
  crps_test_f <- matrix(NA, n_obs, mvn_dim_)

  for (ii in 1:n_obs) {
    for (j in 1:mvn_dim_) {
      crps_train[ii, j] <- scoringRules::crps_sample(y_true_train[ii, j], dat = fitted$y_hat_train[ii, j, ])
      crps_test[ii, j] <- scoringRules::crps_sample(y_true_test[ii, j], dat = fitted$y_hat_test[ii, j, ])
      crps_train_f[ii, j] <- scoringRules::crps_sample(y_true_train[ii, j], dat = fitted$f_hat_train[ii, j, ])
      crps_test_f[ii, j] <- scoringRules::crps_sample(y_true_test[ii, j], dat = fitted$f_hat_test[ii, j, ])
    }
  }

  # Comparison metrics
  for (j in 1:mvn_dim_) {
    metrics <- data.frame(
      metric = c(
        "rmse_train", "rmse_test", "crps_train", "crps_test",
        "pi_train", "pi_test", "ci_train", "ci_test", "cr_train", "cr_test"
      ),
      value = c(
        rmse(fitted$f_hat_train_mean[, j], y_true_train[, j]),
        rmse(fitted$f_hat_test_mean[, j], y_true_test[, j]),
        mean(crps_train_f[, j]),
        mean(crps_test_f[, j]),
        pi_coverage(y_train[, j], fitted$f_hat_train[, j, ], sqrt(fitted$Sigma[j, j, ]), 0.5, n_mcmc_replications = 100),
        pi_coverage(y_test[, j], fitted$f_hat_test[, j, ], sqrt(fitted$Sigma[j, j, ]), 0.5),
        ci_coverage(y_true_train[, j], fitted$f_hat_train_mean[, j], mean(sqrt(fitted$Sigma[j, j, ])), 0.5),
        ci_coverage(y_true_test[, j], fitted$f_hat_test_mean[, j], mean(sqrt(fitted$Sigma[j, j, ])), 0.5),
        cr_coverage(y_true_train[, j], fitted$f_hat_train[, j, ], 0.5),
        cr_coverage(y_true_test[, j], fitted$f_hat_test[, j, ], 0.5)
      ),
      model = "mvBART",
      fold = i,
      mvn_dim = j
    )
    comparison_metrics <- rbind(comparison_metrics, metrics)
  }

  # Correlation metrics
  rho_true <- Sigma_[1, 2] / (sqrt(Sigma_[1, 1]) * sqrt(Sigma_[2, 2]))
  rho_post <- fitted$Sigma[1, 2, ] / (sqrt(fitted$Sigma[1, 1, ]) * sqrt(fitted$Sigma[2, 2, ]))
  correlation_metrics <- rbind(
    correlation_metrics,
    data.frame(
      metric = c("cr_cov", "rmse"),
      value = c(
        cr_coverage(rho_true, matrix(rho_post, ncol = length(rho_post)), 0.5),
        rmse(mean(rho_post), rho_true)
      ),
      model = "mvBART",
      mvn_dim = mvn_dim_,
      param_index = "rho12",
      fold = i
    )
  )

  # Sigma metrics
  for (j in 1:mvn_dim_) {
    sigma_true <- sqrt(Sigma_[j, j])
    sigma_post <- sqrt(fitted$Sigma[j, j, ])
    correlation_metrics <- rbind(
      correlation_metrics,
      data.frame(
        metric = c("cr_cov", "rmse"),
        value = c(
          cr_coverage(sigma_true, matrix(sigma_post, ncol = length(sigma_post)), 0.5),
          rmse(mean(sigma_post), sigma_true)
        ),
        model = "mvBART",
        mvn_dim = mvn_dim_,
        param_index = paste0("sigma", j, j),
        fold = i
      )
    )
  }

  return(list(
    comparison_metrics = comparison_metrics,
    correlation_metrics = correlation_metrics,
    sigma = fitted$Sigma
  ))
}




# Cross-validation wrapper for STAN MVN models
cv_matrix_stan_mvn <- function(cv_element_,
                               n_tree_,
                               mvn_dim_,
                               n_,
                               p_,
                               i,
                               stan_model_regression,
                               task_,
                               n_mcmc_,
                               n_burn_) {
  library(rstan)

  # For replicate paper results this should be only used for the BayesSUR classification settings.
  
  # Extract train/test data
  x_train <- cv_element_$train$x
  y_train <- cv_element_$train$y
  x_test <- cv_element_$test$x
  y_test <- cv_element_$test$y
  y_true_train <- cv_element_$train$y_true
  y_true_test <- cv_element_$test$y_true

  if (task_ == "classification") {
    z_true_train <- cv_element_$train$z_true
    z_true_test <- cv_element_$test$z_true
    z_train <- cv_element_$train$z
    z_test <- cv_element_$test$z
    p_true_train <- pnorm(z_true_train)
    p_true_test <- pnorm(z_true_test)
  }

  Sigma_ <- cv_element_$train$Sigma

  # Initialize metric storage
  comparison_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    fold = integer()
  )
  correlation_metrics <- data.frame(
    metric = character(),
    value = numeric(),
    model = character(),
    mvn_dim = integer(),
    param_index = character(),
    fold = integer()
  )

  # Stan data & sampling
  stan_data <- if (task_ == "classification") {
    list(
      x_train = x_train, x_test = x_test, y = y_train,
      N = n_, D = mvn_dim_, K = p_
    )
  }

  stan_fit_class <- sampling(
    object = stan_model_regression,
    data = stan_data,
    pars = c("z_hat_train", "z_hat_test", "Omega"),
    include = TRUE,
    chains = 1,
    iter = n_mcmc_,
    warmup = n_burn_
  )

  stan_samples_class <- rstan::extract(stan_fit_class)
  stan_samples_class$z_hat_train_mean <- apply(stan_samples_class$z_hat_train, c(2:3), mean)
  stan_samples_class$z_hat_test_mean <- apply(stan_samples_class$z_hat_test, c(2:3), mean)
  stan_samples_class$p_hat_train <- pnorm(stan_samples_class$z_hat_train)
  stan_samples_class$p_hat_test <- pnorm(stan_samples_class$z_hat_test)
  stan_samples_class$p_hat_train_mean <- apply(stan_samples_class$p_hat_train, c(2:3), mean)
  stan_samples_class$p_hat_test_mean <- apply(stan_samples_class$p_hat_test, c(2:3), mean)

  # Comparison metrics for classification
  for (j in 1:mvn_dim_) {
    metrics <- data.frame(
      metric = c(
        "logloss_train", "logloss_test", "brier_train", "brier_test",
        "acc_train", "acc_test", "mcc_train", "mcc_test",
        "z_cr_train", "z_cr_test", "p_cr_train", "p_cr_test"
      ),
      value = c(
        logloss(y_true_train[, j], stan_samples_class$p_hat_train_mean[, j]),
        logloss(y_true_test[, j], stan_samples_class$p_hat_test_mean[, j]),
        brierscore(y_true_train[, j], stan_samples_class$p_hat_train_mean[, j]),
        brierscore(y_true_test[, j], stan_samples_class$p_hat_test_mean[, j]),
        acc(y_true_train[, j], ifelse(stan_samples_class$p_hat_train_mean[, j] > 0.5, 1, 0)),
        acc(y_true_test[, j], ifelse(stan_samples_class$p_hat_test_mean[, j] > 0.5, 1, 0)),
        mcc(y_true_train[, j], ifelse(stan_samples_class$p_hat_train_mean[, j] > 0.5, 1, 0)),
        mcc(y_true_test[, j], ifelse(stan_samples_class$p_hat_test_mean[, j] > 0.5, 1, 0)),
        cr_coverage(z_true_train[, j], t(stan_samples_class$z_hat_train[, , j]), 0.5),
        cr_coverage(z_true_test[, j], t(stan_samples_class$z_hat_test[, , j]), 0.5),
        cr_coverage(p_true_train[, j], t(stan_samples_class$p_hat_train[, , j]), 0.5),
        cr_coverage(p_true_test[, j], t(stan_samples_class$p_hat_test[, , j]), 0.5)
      ),
      model = "bayesSUR",
      fold = i,
      mvn_dim = j
    )
    comparison_metrics <- rbind(comparison_metrics, metrics)
  }

  # Correlation metrics
  if (mvn_dim_ == 2) {
    rho_ <- Sigma_[1, 2] / (sqrt(Sigma_[1, 1]) * sqrt(Sigma_[2, 2]))
    rho_post <- stan_samples_class$Omega[, 1, 2] / (sqrt(stan_samples_class$Omega[, 1, 1]) * sqrt(stan_samples_class$Omega[, 2, 2]))
    correlation_metrics <- rbind(
      correlation_metrics,
      data.frame(
        metric = c("cr_cov", "rmse"),
        value = c(
          cr_coverage(rho_, matrix(rho_post, ncol = length(rho_post)), 0.5),
          rmse(mean(rho_post), rho_)
        ),
        model = "bayesSUR",
        mvn_dim = mvn_dim_,
        param_index = "rho12",
        fold = i
      )
    )
  }

  if (mvn_dim_ == 3) {
    rho_list <- list(
      rho12 = c(Sigma_[1, 2], stan_samples_class$Omega[, 1, 2], 1, 2),
      rho13 = c(Sigma_[1, 3], stan_samples_class$Omega[, 1, 3], 1, 3),
      rho23 = c(Sigma_[2, 3], stan_samples_class$Omega[, 2, 3], 2, 3)
    )

    for (rho_name in names(rho_list)) {
      rho_true <- rho_list[[rho_name]][1]
      rho_post <- rho_list[[rho_name]][2] / (sqrt(stan_samples_class$Omega[, rho_list[[rho_name]][3], rho_list[[rho_name]][3]]) *
        sqrt(stan_samples_class$Omega[, rho_list[[rho_name]][4], rho_list[[rho_name]][4]]))
      correlation_metrics <- rbind(
        correlation_metrics,
        data.frame(
          metric = c("cr_cov", "rmse"),
          value = c(
            cr_coverage(rho_true, matrix(rho_post, ncol = length(rho_post)), 0.5),
            rmse(mean(rho_post), rho_true)
          ),
          model = "bayesSUR",
          mvn_dim = mvn_dim_,
          param_index = rho_name,
          fold = i
        )
      )
    }
  }

  return(list(
    comparison_metrics = comparison_metrics,
    correlation_metrics = correlation_metrics,
    sigma = stan_samples_class$Omega
  ))
}
