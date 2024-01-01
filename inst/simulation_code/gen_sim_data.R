# This chunk of code generate the list with 100 replications for each dataset for multiple parametrisations
rm(list=ls())
set.seed(42)
source("R/simulation_functions.R")
n_rep <- 100
p_ <- c(10, 20, 50) # For Friedman 1 there's the p_ = 100 and n_ = 1000, but I will disconsider those at principle for the other examples
n_ <- c(250,500,1000)
mvn_dim_ <- c(2,3)
cv_ <- vector("list",length = n_rep)
for(p in p_){
     for(n in n_){
         for(mvn_dim in mvn_dim_){

              for(rep in 1:n_rep){
                   cv_[[rep]]$train <- sim_class_mvn_friedman2(n = n,p = p,mvn_dim = mvn_dim)
                   cv_[[rep]]$test <- sim_class_mvn_friedman2(n = n,p = p,mvn_dim = mvn_dim)
              }
              # saveRDS(cv_,file = paste0("inst/cv_data/classification/friedman2_n_",n,"_p_",p,"_d_",mvn_dim,".Rds"))
         }
     }
}


