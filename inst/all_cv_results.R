## This file stores all runs for the results for each one of the models: BART, subart, mvBART, bayesSUR

# Generating the model results
rm(list=ls())
library(doParallel)
devtools::load_all()
seed_ <- 43
set.seed(seed_) # ORIGINal is 42
n_ <- 250
p_ <- 10
n_tree_ <- 100
n_mcmc_ <- 5000
n_burn_ <- 1000
mvn_dim_ <- 2
task_ <- "classification" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'
model <- "bayesSUR"

if(!model %in% c("bayesSUR","subart","bart","mvBART"))
# Printing whcih model is being generated
cat("Model:", model,"n_", n_, "p_" , p_, "tree", n_tree_, "mvn_dim", mvn_dim_, "task", task_, "sim " , sim_)

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
# cv_element_ <- cv_[[i]]

# Setting up the parallel simulation
number_cores <- 20
cl <- parallel::makeCluster(number_cores)
doParallel::registerDoParallel(cl)


if(model == "bayesSUR" & task_ == "classification"){
     source("inst/stan_classification_compile.R")
}

# Selecting a path to save the model
path <- "/Users/mateusmaia/Documents/saving_models_test/"

#Small test to verify the specified path
if(!(file.exists(path) && file.info(path)$isdir)){
        stop("Insert a valid directory path to save the models")
}


result <- foreach(i = 1:n_rep,.packages = c("dbarts","surbayes","dplyr","subart","skewBART")) %dopar%{

     source("inst/all_cv_functions.R")

     devtools::load_all()

     if(model == 'bart'){
             aux <- cv_matrix_bart(cv_element_ = cv_[[i]],
                                   n_tree_ = n_tree_,
                                   mvn_dim_ = mvn_dim_,
                                   n_ = n_,
                                   p_ = p_,
                                   i =  i,
                                   task_ = task_,
                                   n_mcmc_ = n_mcmc_,
                                   n_burn_ = n_burn_)
     } else if( model == 'subart'){
             aux <- cv_matrix_subart(cv_element_ = cv_[[i]],
                                          n_tree_ = n_tree_,
                                          mvn_dim_ = mvn_dim_,
                                          n_ = n_,
                                          p_ = p_,
                                          i =  i,
                                          task_ = task_,
                                          n_mcmc_ = n_mcmc_,
                                          n_burn_ = n_burn_)
     } else if (model == 'bayesSUR' & task_ == 'regression'){
             aux <- cv_matrix_bayesSUR(cv_element_ = cv_[[i]],
                                                n_tree_ = n_tree_,
                                                mvn_dim_ = mvn_dim_,
                                                n_ = n_,
                                                p_ = p_,
                                                i =  i,
                                                task_ = task_,
                                                n_mcmc_ = n_mcmc_,
                                                n_burn_ = n_burn_)
     } else if (model == "bayesSUR" & task_ == "classification"){
             aux <- cv_matrix_stan_mvn(cv_element_ = cv_[[i]],
                                                  n_tree_ = n_tree_,
                                                  mvn_dim_ = mvn_dim_,
                                                  n_ = n_,
                                                  p_ = p_,
                                                  i =  i,
                                                  task_ = task_,
                                                  stan_model_regression  = stan_model_regression,
                                                  n_mcmc_ = n_mcmc_,
                                                  n_burn_ = n_burn_)
     } else if(model == 'mvBART') {
             aux <- cv_matrix_skewBART(cv_element_ = cv_[[i]],
                                                n_tree_ = n_tree_,
                                                mvn_dim_ = mvn_dim_,
                                                n_ = n_,
                                                p_ = p_,
                                                i =  i,
                                                task_ = task_,
                                                n_mcmc_ = n_mcmc_,
                                                n_burn_ = n_burn_)
     } else {
             stop( " no valid model and task selected")
     }

     aux

}


stopCluster(cl)

# Saving the results
saveRDS(object = result, file = paste0(path,"seed_",seed_,"_",model,"_",sim_,"_",task_,"_n_",n_,"_p_",p_,
                                       "_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))

