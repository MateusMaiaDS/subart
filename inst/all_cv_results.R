## This file stores all runs for the results for each one of the models: BART, subart, mvBART

# Generating the model results
rm(list=ls())
library(doParallel)
devtools::load_all()
set.seed(43) # ORIGINal is 42
n_ <- 250
p_ <- 10
n_tree_ <- 50
n_mcmc_ <- 5000
n_burn_ <- 1000
mvn_dim_ <- 2
task_ <- "classification" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'
model <- "bayesSUR"

# Printing whcih model is being generated
cat("n_", n_, "p_" , p_, "tree", n_tree_, "mvn_dim", mvn_dim_, "task", task_, "sim " , sim_)

# It was run to test at first
n_rep <- 100
cv_ <- vector("list",length = n_rep)

if(task_ == "regression" & sim_ == "friedman1"){
     for(rep in 1:n_rep){
          cv_[[rep]]$train <- subart::sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
          cv_[[rep]]$test <- subart::sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
     }
} else if (task_ == "regression" & sim_ == "friedman2"){
     for(rep in 1:n_rep){
          cv_[[rep]]$train <- subart::sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
          cv_[[rep]]$test <- subart::sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
     }
} else if(task_ == "classification" & sim_ == "friedman1"){
     for(rep in 1:n_rep){
          cv_[[rep]]$train <- subart::sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
          cv_[[rep]]$test <- subart::sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
     }
} else {
     stop ("Insert a valid task and simulation")
}

# Running inside the function
i <- 1
# cv_element_ <- cv_[[i]]

# Setting up the parallel simulation
# number_cores <- 20
# cl <- parallel::makeCluster(number_cores)
# doParallel::registerDoParallel(cl)
if(model == "bayesSUR" & task_ == "classification"){
     source("inst/stan_classification_compile.R")
}

# result <- foreach(i = 1:n_rep,.packages = c("dbarts","systemfit","dplyr","subart","skewBART")) %dopar%{


     source("inst/all_cv_functions.R")

     devtools::load_all()

     aux_bart <- cv_matrix_bart(cv_element_ = cv_[[i]],
                           n_tree_ = n_tree_,
                           mvn_dim_ = mvn_dim_,
                           n_ = n_,
                           p_ = p_,
                           i =  i,
                           task_ = task_,
                           n_mcmc_ = n_mcmc_,
                           n_burn_ = n_burn_)

     aux_subart <- cv_matrix_subart(cv_element_ = cv_[[i]],
                                  n_tree_ = n_tree_,
                                  mvn_dim_ = mvn_dim_,
                                  n_ = n_,
                                  p_ = p_,
                                  i =  i,
                                  task_ = task_,
                                  n_mcmc_ = n_mcmc_,
                                  n_burn_ = n_burn_)

     aux_bayesSUR <- cv_matrix_bayesSUR(cv_element_ = cv_[[i]],
                                         n_tree_ = n_tree_,
                                         mvn_dim_ = mvn_dim_,
                                         n_ = n_,
                                         p_ = p_,
                                         i =  i,
                                         task_ = task_,
                                         n_mcmc_ = n_mcmc_,
                                         n_burn_ = n_burn_)

     aux_skewBART <- cv_matrix_skewBART(cv_element_ = cv_[[i]],
                                        n_tree_ = n_tree_,
                                        mvn_dim_ = mvn_dim_,
                                        n_ = n_,
                                        p_ = p_,
                                        i =  i,
                                        task_ = task_,
                                        n_mcmc_ = n_mcmc_,
                                        n_burn_ = n_burn_)

     aux_stan_class <- cv_matrix_stan_mvn(cv_element_ = cv_[[i]],
                                          n_tree_ = n_tree_,
                                          mvn_dim_ = mvn_dim_,
                                          n_ = n_,
                                          p_ = p_,
                                          i =  i,
                                          task_ = task_,
                                          stan_model_regression  = stan_model_regression,
                                          n_mcmc_ = n_mcmc_,
                                          n_burn_ = n_burn_)


# stopCluster(cl)

# Saving the results
# saveRDS(object = result, file = paste0("inst/cv_data/",task_,"/result/FIG_4_september_new_crps_",sim_,"_",task_,"_n_",n_,"_p_",p_,
#                                        "_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))

