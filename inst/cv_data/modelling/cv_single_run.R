# Generating the model results
rm(list=ls())
library(doParallel)
devtools::load_all()
set.seed(42)
n_ <- 250
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 3
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

# Choose one repetition
rep_ <- 1
cv_element_ <- cv_[[rep_]]


# Getting the data elements
x_train <- cv_element_$train$x
y_train <- cv_element_$train$y
x_test <- cv_element_$test$x
y_test <- cv_element_$test$y
y_true_train <- cv_element_$train$y_true
y_true_test <- cv_element_$test$y_true

# True Sigma element
Sigma_ <- cv_element_$train$Sigma

# Doing the same for the MVN-BART
mvbart_mod <- mvnbart(x_train = x_train,y_mat = y_train,x_test = x_test,
                      n_tree = n_tree_,n_mcmc = 2500,n_burn = 500,df = 10,
                      m = nrow(x_train))



par(mfrow=c(2,3))
sqrt(mvbart_mod$Sigma_post[1,1,]) %>% plot(type = 'l', main = expression(sigma[1]))
sqrt(mvbart_mod$Sigma_post[2,2,]) %>% plot(type = 'l', main = expression(sigma[2]))
sqrt(mvbart_mod$Sigma_post[3,3,]) %>% plot(type = 'l', main = expression(sigma[3]))
(mvbart_mod$Sigma_post[1,3,]/(sqrt(mvbart_mod$Sigma_post[3,3,])*sqrt(mvbart_mod$Sigma_post[1,1,]))) %>% plot(type = 'l', main = expression(rho[13]))
(mvbart_mod$Sigma_post[1,2,]/(sqrt(mvbart_mod$Sigma_post[2,2,])*sqrt(mvbart_mod$Sigma_post[1,1,]))) %>% plot(type = 'l', main = expression(rho[12]))
(mvbart_mod$Sigma_post[2,3,]/(sqrt(mvbart_mod$Sigma_post[3,3,])*sqrt(mvbart_mod$Sigma_post[2,2,])))%>% plot(type = 'l', main = expression(rho[23]))
