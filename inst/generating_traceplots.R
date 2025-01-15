# Generating the model results
rm(list=ls())
library(doParallel)
library(subart)
seed_ <- 43
set.seed(seed_) # ORIGINal is 42
n_ <- 1000
p_ <- 10
n_tree_ <- 200
n_mcmc_ <- 10000
n_burn_ <- 2000
mvn_dim_ <- 2
task_ <- "classification" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'
model <- "subart"

set.seed(seed_)
if(!model %in% c("subart")){
     stop("Not a valid model!")
}
# Printing whcih model is being generated
print(cat("Model:", model,"n_", n_, "p_" , p_, "tree", n_tree_, "mvn_dim", mvn_dim_, "task", task_, "sim " , sim_))

if(task_ == "regression" & sim_ == "friedman1"){

          train <- subart::sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
          test <- subart::sim_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)

} else if (task_ == "regression" & sim_ == "friedman2"){

          train <- subart::sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)
          test <- subart::sim_mvn_friedman2(n = n_,p = p_,mvn_dim = mvn_dim_)

} else if(task_ == "classification" & sim_ == "friedman1"){

          train <- subart::sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)
          test <- subart::sim_class_mvn_friedman1(n = n_,p = p_,mvn_dim = mvn_dim_)

} else {
     stop ("Insert a valid task and simulation")
}

x_train <- train$x
x_test <- train$x
y_train <- train$y
nu_prop <- 100
# Doing the same for the MVN-BART
subart_mod <- if(task_ == "regression"){
     subart::subart(x_train = x_train,y_mat = y_train,x_test = x_test,
                                  n_tree = n_tree_,n_mcmc = n_mcmc_,n_burn = n_burn_,nu = 2) # Maybe need to change the df to 10
} else if(task_ == "classification"){
     subart::subart(x_train = x_train,y_mat = y_train,x_test = x_test,m = nu_prop,
                                  n_tree = n_tree_,n_mcmc = n_mcmc_,n_burn = n_burn_,nu = 2)
}


storing_posterior <- subart_mod$all_Sigma_post
storing_ess <- subart_mod$ESS

bart_mod <- dbarts::bart(x.train = x_train,y.train = y_train[,1],x.test = x_test,
                         ntree = 200,ndpost = 10000,nskip = 0)

# plot(bart_mod$yhat.train[2001:10000,], type = 'l',ylim = c(-2,2),ylab = expression(y[1,1]),
#      xlab = "mcmc iter", col = ggplot2::alpha("black", 0.5))
# lines(subart_mod$y_hat[1,1,], type = 'l',  col = ggplot2::alpha("blue", 0.5))
#
# ESS(bart_mod$yhat.train[2001:10000,1])
# ESS(subart_mod$y_hat[1,1,])

mcmc_obj <- list(all_Sigma_iter = storing_posterior,
                 ess = storing_ess,
                 y_hat = subart_mod$y_hat,
                 bart_mod = bart_mod$yhat.train)

path <- "inst/mcmc_result/"
saveRDS(mcmc_obj,file = paste0(path,"seed_",seed_,"_",model,"_",sim_,"_",task_,"_n_",n_,"_p_",p_,
                      "_ntree_",n_tree_,"_mvndim_",mvn_dim_,"_nmcmc_",n_mcmc_,"_nuprop_",nu_prop,".Rds"))

plot(mcmc_obj$all_Sigma_iter[1,2,], type = 'l',ylab = expression(rho),xlab = "MCMC ")
