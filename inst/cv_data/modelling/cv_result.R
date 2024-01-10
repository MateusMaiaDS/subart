# Generating the model results
rm(list=ls())
library(doParallel)
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

# Running inside the function
# i <- 1
# cv_element_ <- cv_[[i]]

# Setting up the parallel simulation
number_cores <- 20
cl <- parallel::makeCluster(number_cores)
doParallel::registerDoParallel(cl)

result <- foreach(i = 1:n_rep,.packages = c("dbarts","systemfit","dplyr")) %dopar%{


     source("inst/cv_data/modelling/cv_functions.R")

     devtools::load_all()

     aux <- cv_matrix(cv_element_ = cv_[[i]],
                      n_tree_ = n_tree_,
                      mvn_dim_ = mvn_dim_,
                      n_ = n_,
                      p_ = p_,
                      i =  i,
                      task_ = task_)
     aux
}


stopCluster(cl)

# Saving the results
saveRDS(object = result, file = paste0("inst/cv_data/",task_,"/result/",sim_,"_",task_,"_n_",n_,"_p_",p_,
                                      "_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))

