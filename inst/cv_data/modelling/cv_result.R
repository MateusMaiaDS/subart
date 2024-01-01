# Generating the model results
rm(list=ls())
library(doParallel)
set.seed(42)
n_ <- 250
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 2
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'

# cv_ <- readRDS(file = paste0("inst/cv_data/",task,"/",sim_,"_n_",n_,"_p_",p_,"_d_",mvn_dim_,".Rds"))
# n_rep_ <- length(cv_)
i <- 1
# cv_element_ <- cv_[[i]]


# It was run to test at first
n_rep <- 100
cv_ <- vector("list",length = n_rep)

for(rep in 1:n_rep){
     cv_[[rep]]$train <- sim_class_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
     cv_[[rep]]$test <- sim_class_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
}

# Setting up the parallel simulation
number_cores <- n_rep_
cl <- parallel::makeCluster(number_cores)
doParallel::registerDoParallel(cl)

result <- foreach(i = 1:n_rep_,.packages("dbarts","systemfit","dplyr")){


     source("inst/cv_data/modelling/cv_functions.R")

     aux <- cv_matrix(cv_element_ = cv_[[i]],
                      n_tree_ = n_tree_,
                      mvn_dim_ = mvn_dim_,
                      n_ = n_,
                      p_ = p_,
                      i =  i )
     aux
}


stopCluster(cl)

# Saving the results
saveRDS(object = result, file = paste0("inst/cv_data/regression/result/",sim_,"_",task_,"_n_",n_,"_p_",p_,
                                      "_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))

