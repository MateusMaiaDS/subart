#' Multivariate Normal Bayesian Additive Regression trees.
#' @useDynLib subart
#' @importFrom Rcpp sourceCpp
#'
#' @description
#' \code{subart()} function models a Bayesian Additive Regression trees model considering the Multivariate Normal (MVN) distribution
#' from the target dependent variable \eqn{Y_{i} \in \mathbb{R}^{d}}.
#'
#' @details
#'
#' Add more details of the function description
#'
#' @returns
#' In case of a continuous response the model returns
#' \code{subart} object with the model predictions and parameters for the standard MVN. For the binary-classification problems the probit
#' Probit-MVN approach is used, and returns the \code{subart} object.
#'
#'
#' @param x_train A \code{data.frame} of the training data covariates. When \code{x_train} contains unordered categorical variables with more than two levels, the corresponding columns should be defined as factors.
#' @param y_mat A numeric matrix of the data responses.
#' @param x_test A \code{data.frame} of the test data covariates. If `NULL` only predictions for \code{x_train} will be used. The columns of \code{x_test} should match those of \code{x_train} in name, order and type.
#' @param n_tree The number of trees used in each set of trees for the respective \eqn{j} entry. The total number of trees is given by \eqn{j \times p}.
#' @param node_min_size The minimun number of observations within a terminal node.
#' @param n_mcmc The total number of MCMC iterations.
#' @param n_burn The number of MCMC iterations to be tretead as burn in samples
#' @param alpha The power parameter used in tree prior.
#' @param beta The base parameter used in tree prior.
#' @param nu The \eqn{\nu} parameter associated with the degree of freedom from the variance prior.
#' @param sigquant Quantile used to define the residuals variance. Remind that the prior definition is based on \eqn{P(\sigma^{(j)} < \hat{\sigma}^{(j)})} where \eqn{\hat{\sigma}^{(j)}} is a "rough data-based estimation" for example, the sample variance of the observed \eqn{y^{(j)}} values.
#' @param kappa Hyper-parameter from the \eqn{\mu_{t\ell}^{(j)}} (Chipman et. al 2010). Such that \eqn{\sigma} = 0.25
#' @param numcut The maximum number of possible values used in the tree decision rules. The uniform approximation for choose a decision rule over \eqn{X^{(j)}} is given a grid of size \code{numcut}.
#' @param usequants Boolean; if true the quantiles are going to be used to define the grid of cutpoints.
#' @param m Hyperparameter used in the definition of the prior setting of the correlation matrix for the Probit-Multivariate approach.
#' @param varimportance Boolean; if true returns a matrix with \code{n_mcmc} rows and \eqn{d} columns corresponding to the total sum of number of times that a variable \eqn{j} was used among all trees of a MCMC iteration.
#' @param specify_variables a list of numeric vectors where each respective element contains the indexes of the covariates allowed to be selected for the set of trees for the respective respose \eqn{Y_{j}}. The default is \code{NULL} and allow to all covariates from \eqn{X} to be selected for all trees.
#' @param diagnostic a boolean to compute or not the ESS for the posterior samples of the \eqn{\boldsymbol{\Sigma}}
#'
#' @export
subart <- function(x_train,
                  y_mat,
                  x_test = NULL,
                  n_tree = 100,
                  node_min_size = 5,
                  n_mcmc = 2000,
                  n_burn = 500,
                  alpha = 0.95,
                  beta = 2,
                  nu = 3,
                  sigquant = 0.9,
                  kappa = 2,
                  numcut = 100L, # Defining the grid of split rules
                  usequants = FALSE,
                  m = 20, # Degrees of freed for the classification setting.
                  varimportance = TRUE,
                  hier_prior_bool = TRUE, # Use a hierachical prior or not;
                  specify_variables = NULL, # Specify variables for each dimension (j) by name or index for.,
                  diagnostic = TRUE # Calculates the Effective Sample size for the covariance and correlation parameters
                  ) {


     # Transforming x_test in a simple 2line data.frame when x_test is set as NULL
     if(is.null(x_test)){
             x_test <- x_train[1:2,,drop =FALSE]
             null_x_test <- TRUE
     } else {
             null_x_test <- FALSE
     }

     # Handling error heading
     if(n_mcmc<=n_burn){
          stop("Number of MCMC iterations must be greater than the number of burn-in samples.")
     }

     if(is.vector(x_train)|| is.vector(x_test)){
          stop("x_train and x_test must be either a matrix or data.frame.")
     }

     if(nrow(x_train)<numcut){
          warning("numcut is smaller than the number of rows of x_train, numcut was re-defined as the nrow(x_train)")
          numcut <- nrow(x_train)
     }

     if(varimportance && (NCOL(x_train)==1)){
          warning("No varimportance is set as FALSE as there is only one predictor.")
          varimportance <- FALSE
     }

     ## End error handling


     # Defining initial paramters that are no longer up to the user to define
     scale_y <- TRUE # If the target variable Y gonna be scaled or not
     Sigma_init <- NULL # The initial value for the \Sigma matrix initialisation
     update_Sigma <- TRUE # It always TRUE was only useful to test exeperiments
     conditional_bool <- TRUE # Again is always true
     tn_sampler <- FALSE # Define if the truncated-normal sampler gonna be used or not

     if(!is.null(specify_variables) & (length(specify_variables)!=NCOL(y_mat))){
             stop("Specify a proper list for the variables to be used in the tree.")
     }

     if(is.null(specify_variables)){
             sv_bool <- FALSE # boolean to be used in the C++ code
             sv_matrix <- matrix(1,nrow = NCOL(y_mat),ncol = NCOL(x_train))
     } else {
             sv_bool <- TRUE
             sv_matrix <- matrix(0,nrow = NCOL(y_mat),ncol = NCOL(x_train))
             for(i in 1:NCOL(y_mat)){
                     sv_matrix[i,specify_variables[[i]]] <- 1
             }
     }



     # Changing to a classification model
     if(length(unique(c(y_mat[complete.cases(y_mat)])))==2){
           class_model <- TRUE
           scale_y <- FALSE
     } else {
           class_model <- FALSE
     }

     if(class_model){
          if(!identical(sort(round(unique(c(y_mat)))),c(0,1))){
               stop(" Use the y as c(0,1) vector for the classification model.")
          }
     }


     # Avoiding error of this kind
     if(class_model & scale_y){
             stop("Classificaton model should not scale y.")
     }

     # # Verifying if it's been using a y_mat matrix
     # if(NCOL(y_mat)<2 & class_model){
     #      stop("Insert a valid multivariate response for a classification task. ")
     # }

     # Verifying if x_train and x_test are matrices
     if(!is.data.frame(x_train) || !is.data.frame(x_test)){
          stop("Insert valid data.frame for both data and xnew.")
     }


     # Getting the valid
     dummy_x <- base_dummyVars(x_train)

     # Create a data.frame aux

     # Create a data.frame aux
     initial_rank <- FALSE

     # Create a list
     if(length(dummy_x$facVars)!=0 & initial_rank){

          # Selected rank_var categorical
          rank_var <- 1

          for(i in 1:length(dummy_x$facVars)){
               # See if the levels of the test and train matches
               if(!all(levels(x_train[[dummy_x$facVars[i]]])==levels(x_test[[dummy_x$facVars[i]]]))){
                    levels(x_test[[dummy_x$facVars[[i]]]]) <- levels(x_train[[dummy_x$facVars[[i]]]])
               }
               df_aux <- data.frame( x = x_train[,dummy_x$facVars[i]], y = y_mat[,rank_var])
               formula_aux <- stats::aggregate(y~x,df_aux,mean)
               formula_aux$y <- rank(formula_aux$y)
               x_train[[dummy_x$facVars[i]]] <- as.numeric(factor(x_train[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

               # Doing the same for the test set
               x_test[[dummy_x$facVars[i]]] <- as.numeric(factor(x_test[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

               categorical_indicators <- numeric(ncol(x_train))

          }

     } else if ((length(dummy_x$facVars)!=0) && isFALSE(initial_rank)) {

          for(i in 1:length(dummy_x$facVars)){

               x_train[[dummy_x$facVars[i]]] <- as.numeric(x_train[[dummy_x$facVars[i]]])
               x_test[[dummy_x$facVars[i]]] <- as.numeric(x_test[[dummy_x$facVars[i]]])

               categorical_indicators <- numeric(ncol(x_train))
               categorical_indicators[which(colnames(x_train) %in% dummy_x$facVars)] <- 1

          }

     } else {

          categorical_indicators <- numeric(ncol(x_train))

     }

     # Getting the train and test set
     x_train_scale <- as.matrix(x_train)
     x_test_scale <- as.matrix(x_test)

     # Scaling x
     x_min <- apply(as.matrix(x_train_scale),2,min)
     x_max <- apply(as.matrix(x_train_scale),2,max)

     # Storing the original
     x_train_original <- x_train
     x_test_original <- x_test


     # Normalising all the columns
     for(i in 1:ncol(x_train)){
             x_train_scale[,i] <- normalize_covariates_bart(y = x_train_scale[,i],a = x_min[i], b = x_max[i])
             x_test_scale[,i] <- normalize_covariates_bart(y = x_test_scale[,i],a = x_min[i], b = x_max[i])
     }

     # Creating the numcuts matrix of splitting rules
     xcut_m <- matrix(NA,nrow = numcut,ncol = ncol(x_train_scale))
     for(i in 1:ncol(x_train_scale)){

             if(nrow(x_train_scale)<numcut){
                        xcut_m[,i] <- sort(x_train_scale[,i])
             } else {
                        xcut_m[,i] <- seq(min(x_train_scale[,i]),
                                          max(x_train_scale[,i]),
                                          length.out = numcut+2)[-c(1,numcut+2)]
             }
     }


     # Scaling the y
     min_y <- apply(y_mat,2,min,na.rm = TRUE)
     max_y <- apply(y_mat,2,max,na.rm = TRUE)

     # Scaling the data
     if(scale_y){
             y_mat_scale <- y_mat
             for(n_col in 1:NCOL(y_mat)){
                y_mat_scale[,n_col] <- normalize_bart(y = y_mat[,n_col],a = min_y[n_col],b = max_y[n_col])
             }
     } else {
             y_mat_scale <- y_mat
     }
     # Getting the min and max for each column
     min_x <- apply(x_train_scale,2,min)
     max_x <- apply(x_train_scale, 2, max)


     # Defining tau_mu_j
     if(class_model){
             tau_mu_j <- rep((n_tree*(kappa^2))/9.0,NCOL(y_mat))
     } else {
             if(scale_y){
                     tau_mu_j <- rep((4*n_tree*(kappa^2)),NCOL(y_mat))
             } else {
                     tau_mu_j <- (4*n_tree*(kappa^2))/((max_y-min_y)^2)
             }

     }

     # Getting sigma
     sigma_mu_j <- tau_mu_j^(-1/2)

     # =========
     # Calculating prior for the \tau in case of regression and skipping it
     #in terms of classification
     # =========

     if(class_model){
             # Call the bart function
             if(is.null(Sigma_init) || NCOL(y_mat)==1){
                     Sigma_init <- diag(1,nrow = NCOL(y_mat))
             }
             mu_init <- apply(y_mat,2,mean,na.rm = TRUE)

             df <- nu + ncol(y_mat_scale) - 1
             # No extra parameters are need to calculate for the class model
     } else {
             # Getting the naive sigma value
             if(nrow(x_train_scale) > ncol(y_mat_scale)){
                nsigma <- apply(y_mat_scale, 2, function(Y){naive_sigma(x = x_train_scale,y = Y)})
             } else {
                nsigma <- apply(y_mat_scale, 2, function(Y){stats::sd(Y,na.rm = TRUE)})
             }
             # Define the ensity function
             phalft <- function(x, A, nu){
                     return(2 * stats::pt(x/A, nu) - 1)
             }

             # Define parameters
             df <- nu + ncol(y_mat_scale) - 1

             # Selecting hypera-parmeters for the t-distribution case
             if(hier_prior_bool){
                     A_j <- numeric()

                     for(i in 1:length(nsigma)){
                             # Calculating lambda
                             A_j[i] <- stats::optim(par = 0.01, f = function(A){(sigquant - phalft(nsigma[i], A, nu))^2},
                                                    method = "Brent",lower = 0.00001,upper = 100)$par
                     }

                     # Calculating lambda
                     qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
                     lambda <- (nsigma*nsigma*qchi)/df
                     rate_tau <- (lambda*df)/2

                     S_0_wish <- if(ncol(y_mat)!=1){
                          2*df*diag(c(rate_tau))
                     } else {
                          matrix(2*df*c(rate_tau),ncol = 1,nrow = 1)
                     }



             } else {
                     A_j <- numeric()
                     for(i in 1:length(nsigma)){
                             A_j[i] <- stats::optim(par = 0.01, f = function(A){(sigquant - stats::pgamma(q = 1/(nsigma[i]^2),
                                                                                                              shape = nu/2, rate = A/2,
                                                                                                              lower.tail = FALSE))^2},
                                                        method = "Brent",lower = 0.00001,upper = 100)$par
                     }

                     S_0_wish <- if(ncol(y_mat)!=1){
                          diag(A_j)
                     } else {
                          matrix(A_j,ncol = 1,nrow = 1)
                     }
             }


             # Call the bart function
             if(is.null(Sigma_init)){
                     Sigma_init <- if(ncol(y_mat)!=1){
                          diag(nsigma^2)
                     } else {
                          matrix(nsigma^2,ncol = 1,nrow = 1)
                     }
             }

             mu_init <- apply(y_mat_scale,2,mean,na.rm = TRUE)
     }



     # Generating the BART obj
     if(class_model){

          if(ncol(y_mat_scale)==1 ){ # For the univariate case
               na_boolean <- FALSE
               bart_obj <- cppbart_univariate_CLASS(x_train_scale,
                                                    y_mat_scale,
                                                    x_test_scale,
                                                    xcut_m,
                                                    n_tree,
                                                    node_min_size,
                                                    n_mcmc,
                                                    n_burn,
                                                    Sigma_init,
                                                    mu_init,
                                                    sigma_mu_j,
                                                    nu,
                                                    alpha,beta,
                                                    m,update_Sigma,
                                                    varimportance,
                                                    tn_sampler,
                                                    sv_bool,
                                                    sv_matrix,
                                                    categorical_indicators)
          } else {

               if(any(is.na(y_mat_scale))){
                    y_mat_scale[is.na(y_mat_scale)] <- -1
                    na_boolean <- TRUE
               } else {
                    na_boolean <- FALSE
               }

               bart_obj <- cppbart_CLASS(x_train_scale,
                                         y_mat_scale,
                                         x_test_scale,
                                         xcut_m,
                                         n_tree,
                                         node_min_size,
                                         n_mcmc,
                                         n_burn,
                                         Sigma_init,
                                         mu_init,
                                         sigma_mu_j,
                                         nu,
                                         alpha,beta,
                                         m,update_Sigma,
                                         varimportance,
                                         tn_sampler,
                                         sv_bool,
                                         sv_matrix,
                                         categorical_indicators)
          }

     } else {
          if(ncol(y_mat_scale)==1 ){ # For the univariate case
                    na_boolean <- FALSE
                    bart_obj <- cppbart_univariate(x_train_scale,
                                        y_mat_scale,
                                        x_test_scale,
                                        xcut_m,
                                        n_tree,
                                        node_min_size,
                                        n_mcmc,
                                        n_burn,
                                        Sigma_init,
                                        mu_init,
                                        sigma_mu_j,
                                        alpha,beta,nu,
                                        S_0_wish,
                                        A_j,
                                        update_Sigma,
                                        varimportance,
                                        sv_bool,
                                        hier_prior_bool,
                                        sv_matrix,
                                        categorical_indicators)
          } else {
                    if(any(is.na(y_mat_scale))){
                             number_na <- apply(y_mat_scale,2,function(x){sum(is.na(x),na.rm = TRUE)})
                             na_indicators <- ifelse(is.na(y_mat_scale),1,0)
                             y_mat_scale[is.na(y_mat_scale)] <- 0

                             na_boolean <- TRUE

                             bart_obj <- cppbart_missing(x_train_scale,
                                                        y_mat_scale,
                                                        number_na,
                                                        na_indicators,
                                                        x_test_scale,
                                                        xcut_m,
                                                        n_tree,
                                                        node_min_size,
                                                        n_mcmc,
                                                        n_burn,
                                                        Sigma_init,
                                                        mu_init,
                                                        sigma_mu_j,
                                                        alpha,beta,nu,
                                                        S_0_wish,
                                                        A_j,
                                                        update_Sigma,
                                                        varimportance,
                                                        sv_bool,
                                                        hier_prior_bool,
                                                        sv_matrix,
                                                        categorical_indicators)

                     } else {
                             na_boolean <- FALSE
                             bart_obj <- cppbart(x_train_scale,
                                                 y_mat_scale,
                                                 x_test_scale,
                                                 xcut_m,
                                                 n_tree,
                                                 node_min_size,
                                                 n_mcmc,
                                                 n_burn,
                                                 Sigma_init,
                                                 mu_init,
                                                 sigma_mu_j,
                                                 alpha,beta,nu,
                                                 S_0_wish,
                                                 A_j,
                                                 update_Sigma,
                                                 varimportance,
                                                 sv_bool,
                                                 hier_prior_bool,
                                                 sv_matrix,
                                                 categorical_indicators)
                     }
               }

     }


     # Returning the main components from the model
     y_train_post <- bart_obj[[1]]
     y_test_post <- bart_obj[[2]]
     y_mat_post <-if(na_boolean){
          bart_obj[[8]]
     } else {
             NULL
     }
     Sigma_post <- bart_obj[[3]]
     all_Sigma_post <- bart_obj[[4]]


     # Getting the mean values for the Sigma and \y_hat and \y_hat_test
     Sigma_for <- matrix(0,nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
     y_train_for <- matrix(0,nrow = nrow(y_mat),ncol = ncol(y_mat))
     y_test_for <- matrix(0,nrow = nrow(x_test),ncol = ncol(y_mat))

     Sigma_scale <- if(ncol(y_mat)!=1){
          diag((max_y-min_y))
     } else {
          matrix((max_y-min_y),ncol=1,nrow=1)
     }

     # Reg_model_bool
     if(scale_y){

             # Re-scaling Sigma_all, important to cover convergence issues.
             for(k in 1:(dim(all_Sigma_post)[3])){
                  all_Sigma_post[,,k] <- crossprod(Sigma_scale,tcrossprod(all_Sigma_post[,,k],Sigma_scale))

             }

             for(i in 1:(dim(Sigma_post)[3])){
                     Sigma_post[,,i] <- crossprod(Sigma_scale,tcrossprod(Sigma_post[,,i],Sigma_scale))
                     Sigma_for <- Sigma_for +  Sigma_post[,,i]
                     for( jj in 1:NCOL(y_mat)){
                             y_train_for[,jj] <- y_train_for[,jj] + unnormalize_bart(z = y_train_post[,jj,i],a = min_y[jj],b = max_y[jj])
                             y_test_for[,jj] <- y_test_for[,jj] +  unnormalize_bart(z = y_test_post[,jj,i],a = min_y[jj],b = max_y[jj])
                             y_train_post[,jj,i] <- unnormalize_bart(z = y_train_post[,jj,i],a = min_y[jj],b = max_y[jj])
                             y_test_post[,jj,i] <-  unnormalize_bart(z = y_test_post[,jj,i],a = min_y[jj],b = max_y[jj])
                             if(na_boolean){
                                y_mat_post[,jj,i] <- unnormalize_bart(z = y_mat_post[,jj,i],a = min_y[jj],b = max_y[jj])
                             }
                     }
             }
     } else {
             for(i in 1:(dim(Sigma_post)[3])){
                     Sigma_for <- Sigma_for + Sigma_post[,,i]
                     y_train_for <- y_train_for +  y_train_post[,,i]
                     y_test_for <- y_test_for +  y_test_post[,,i]
             }
     }


     Sigma_post_mean <- Sigma_for/dim(Sigma_post)[3]
     y_mat_mean <- y_train_for/dim(y_train_post)[3]
     y_mat_test_mean <- y_test_for/dim(y_test_post)[3]
     sigmas_mean <- sqrt(diag(Sigma_post_mean))

     # plot(y_mat_mean[,2],sim_data$y[,2])

     # Transforming to classification context

     # Getting the list of outcomes
     if(class_model){

             # Case if storing a variable selection or not
             if(varimportance){
                     var_importance <- array(NA,dim = c(n_mcmc,ncol(x_test_scale),ncol(y_mat)))
                     var_importance_raw <- bart_obj[[8]]
                     for(ii in 1:n_mcmc){
                             for(jj in 1:ncol(y_mat)){
                                     var_importance[ii,,jj] <- apply(bart_obj[[8]][ii][[1]][,,jj],2,sum)
                             }
                     }

                     class(var_importance) <- "varimportance"
             } else {
                    var_importance_raw <- var_importance <- NULL
             }

             # Calculate the ESS for all parameters throw a warning if any of them is smaller than half of the MCMC samples
             if(diagnostic){

                     diagnostic_bool = FALSE
                     ESS_val <- matrix(NA, nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
                     ESS_warn <- FALSE
                     for(i in 1:nrow(Sigma_post)){
                             # ESS_val[i,i] <- ESS(x = Sigma_post[i,i,]) # For classification there's no sample
                             j = i
                             while(j < nrow(Sigma_post)){
                                     j = j+1
                                     ESS_val[i,j] <- ESS_val[j,i] <- ESS(x = Sigma_post[i,j,]/(sqrt(Sigma_post[i,i,])*sqrt(Sigma_post[j,j,])))
                                     if(ESS_val[i,j]<round((n_mcmc-n_burn)/2,digits = 0)){
                                        ESS_warn <- TRUE
                                     }
                             }
                     }

             } else {
                     ESS_val <- NULL
             }

             if(ESS_warn){
                     warning(paste0("A ESS less than ",round((n_mcmc-n_burn)/2,digits = 0)," was obtanied. Verify the traceplots and adjust the priors to improve the sampling."))
             }



             # In case x_test is NULL
             if(null_x_test){
                     y_test_post <- NULL
                     y_mat_test_mean <- NULL
                     y_hat_test_mean_class <- NULL
                     x_test <- NULL
             } else {
                     y_hat_test_mean_class <- apply(y_mat_test_mean,2,function(x){ifelse(x>0,1,0)})
             }


             list_obj_ <- list(y_hat = y_train_post,
                  y_hat_test = y_test_post,
                  y_hat_mean = y_mat_mean,
                  y_hat_test_mean = y_mat_test_mean,
                  y_hat_mean_class = apply(y_mat_mean,2,function(x){ifelse(x>0,1,0)}),
                  y_hat_test_mean_class = y_hat_test_mean_class,
                  Sigma_post = Sigma_post,
                  Sigma_post_mean = Sigma_post_mean,
                  sigmas_post = bart_obj[[7]],
                  all_Sigma_post = all_Sigma_post,
                  var_importance = var_importance,
                  var_importance_raw = var_importance_raw,
                  prior = list(n_tree = n_tree,
                               alpha = alpha,
                               beta = beta,
                               tau_mu_j = tau_mu_j,
                               mu_init = mu_init,
                               tree_proposal = bart_obj[[5]],
                               tree_acceptance = bart_obj[[6]]),
                  mcmc = list(n_mcmc = n_mcmc,
                              n_burn = n_burn),
                  data = list(x_train = x_train,
                              y_mat = y_mat,
                              x_test = x_test),
                  ESS = ESS_val)

             class(list_obj_) <- "subart-probit"
     } else {

             # Case if storing a variable selection or not
             if(varimportance){
                     var_importance <- array(NA,dim = c(n_mcmc,ncol(x_test_scale),ncol(y_mat)))
                     var_importance_raw <- bart_obj[[7]]
                     for(ii in 1:n_mcmc){
                             for(jj in 1:ncol(y_mat)){
                                var_importance[ii,,jj] <- apply(bart_obj[[7]][ii][[1]][,,jj],2,sum)
                             }
                     }

                     class(var_importance) <- "varimportance"

             } else {
                     var_importance_raw <- var_importance <- NULL
             }

             # Calculate the ESS for all parameters throw a warning if any of them is smaller than half of the MCMC samples
             if(diagnostic){

                     diagnostic_bool = FALSE
                     ESS_val <- matrix(NA, nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
                     ESS_warn <- FALSE
                     for(i in 1:nrow(Sigma_post)){
                             ESS_val[i,i] <- ESS(x = sqrt(Sigma_post[i,i,]))
                             j = i
                             while(j < nrow(Sigma_post)){
                                     j = j+1
                                     ESS_val[i,j] <- ESS_val[j,i] <- ESS(x = Sigma_post[i,j,]/(sqrt(Sigma_post[i,i,])*sqrt(Sigma_post[j,j,])))
                                     if(ESS_val[i,j]<round((n_mcmc-n_burn)/2,digits = 0)){
                                             ESS_warn <- TRUE
                                     }
                             }
                     }

             } else {
                     ESS_val <- NULL
             }

             if(ESS_warn){
                     warning(paste0("A ESS less than ",round((n_mcmc-n_burn)/2,digits = 0)," was obtanied. Verify the traceplots and adjust the priors to improve the sampling."))
             }

             # Case x_test = NULL
             if(null_x_test){
                     y_test_post <- NULL
                     y_hat_test_mean <- NULL
                     x_test <- NULL
             }

             # Returning the data list
             data_list <- if(na_boolean){
                  list(x_train = x_train,
                       y_mat = y_mat,
                       x_test = x_test,
                       y_mat_post = y_mat_post)
             } else {
                  list(x_train = x_train,
                       y_mat = y_mat,
                       x_test = x_test)
             }

             list_obj_ <- list(y_hat = y_train_post,
                  y_hat_test = y_test_post,
                  y_hat_mean = y_mat_mean,
                  y_hat_test_mean = y_mat_test_mean,
                  Sigma_post = Sigma_post,
                  Sigma_post_mean = Sigma_post_mean,
                  sigmas_mean = sigmas_mean,
                  all_Sigma_post = all_Sigma_post,
                  var_importance = var_importance,
                  var_importance_raw = var_importance_raw,
                  prior = list(n_tree = n_tree,
                               alpha = alpha,
                               beta = beta,
                               tau_mu_j = tau_mu_j,
                               df = df,
                               A_j = A_j,
                               mu_init = mu_init,
                               tree_proposal = bart_obj[[5]],
                               tree_acceptance = bart_obj[[6]]),
                  mcmc = list(n_mcmc = n_mcmc,
                              n_burn = n_burn),
                  data = data_list,
                  ESS = ESS_val)

             class(list_obj_) <- "subart"
     }

     # Return the list with all objects and parameters
     return(list_obj_)
}

