## This file runs all models: BART, subart, mvBART, bayesSUR

rm(list = ls())
library(doParallel)
library(dplyr)
library(subart)
library(dbarts)
library(skewBART)
library(surbayes)

# -------------------------------
# Simulation settings
# -------------------------------
seed_ <- 43
set.seed(seed_)
n_ <- 250 # n = {250,500,1000}
p_ <- 10
n_tree_ <- 100
n_mcmc_ <- 5000 # See paper specifications to define
n_burn_ <- 2000
mvn_dim_ <- 2
task_ <- "regression" # 'classification' or 'regression'
sim_ <- "friedman1" # 'friedman1' or 'friedman2'
model <- "subart"

if (!model %in% c("bayesSUR", "subart", "bart", "mvBART")) {
  stop("Not a valid model!")
}

cat(
  "Model:", model,
  "n_", n_, "p_", p_,
  "tree", n_tree_,
  "mvn_dim", mvn_dim_,
  "task", task_,
  "sim", sim_, "\n"
)

n_rep <- 100 # Set 100 as the default in the paper
cv_ <- vector("list", length = n_rep)

# -------------------------------
# Generate simulation data
# -------------------------------
sim_fun <- switch(paste(task_, sim_, sep = "_"),
  "regression_friedman1" = subart::sim_mvn_friedman1,
  "regression_friedman2" = subart::sim_mvn_friedman2,
  "classification_friedman1" = subart::sim_class_mvn_friedman1,
  stop("Insert a valid task and simulation")
)

for (rep in 1:n_rep) {
  cv_[[rep]]$train <- sim_fun(n = n_, p = p_, mvn_dim = mvn_dim_)
  cv_[[rep]]$test <- sim_fun(n = n_, p = p_, mvn_dim = mvn_dim_)
}

# -------------------------------
# Parallel setup
# -------------------------------
number_cores <- 2
cl <- parallel::makeCluster(number_cores)
doParallel::registerDoParallel(cl)

if (model == "bayesSUR" & task_ == "classification") {
  source("AOAS_code/stan_classification_compile.R")
}

# -------------------------------
# Directory to save results
# -------------------------------
path <- "inst/"
if (!(file.exists(path) && file.info(path)$isdir)) {
  stop("Insert a valid directory path to save the models")
}

# -------------------------------
# Run models in parallel
# -------------------------------
result <- foreach(i = 1:n_rep, .packages = c("dbarts", "skewBART", "surbayes", "dplyr", "subart")) %dopar% {
  source("AOAS_code/subart_simulations/cv_functions.R")

  aux <- switch(model,
    "bart" = cv_matrix_bart(
      cv_element_ = cv_[[i]], n_tree_ = n_tree_, mvn_dim_ = mvn_dim_,
      n_ = n_, p_ = p_, i = i, task_ = task_, n_mcmc_ = n_mcmc_, n_burn_ = n_burn_
    ),
    "subart" = cv_matrix_subart(
      cv_element_ = cv_[[i]], n_tree_ = n_tree_, mvn_dim_ = mvn_dim_,
      n_ = n_, p_ = p_, i = i, task_ = task_, n_mcmc_ = n_mcmc_, n_burn_ = n_burn_
    ),
    "mvBART" = cv_matrix_skewBART(
      cv_element_ = cv_[[i]], n_tree_ = n_tree_, mvn_dim_ = mvn_dim_,
      n_ = n_, p_ = p_, i = i, task_ = task_, n_mcmc_ = n_mcmc_, n_burn_ = n_burn_
    ),
    "bayesSUR" = {
      if (task_ == "regression") {
        cv_matrix_bayesSUR(
          cv_element_ = cv_[[i]], n_tree_ = n_tree_, mvn_dim_ = mvn_dim_,
          n_ = n_, p_ = p_, i = i, task_ = task_, n_mcmc_ = n_mcmc_, n_burn_ = n_burn_
        )
      } else {
        cv_matrix_stan_mvn(
          cv_element_ = cv_[[i]], n_tree_ = n_tree_, mvn_dim_ = mvn_dim_,
          n_ = n_, p_ = p_, i = i, task_ = task_,
          stan_model_regression = stan_model_regression,
          n_mcmc_ = n_mcmc_, n_burn_ = n_burn_
        )
      }
    },
    stop("No valid model and task selected")
  )

  aux
}

stopCluster(cl)

# -------------------------------
# Save results
# -------------------------------
saveRDS(result, file = paste0(
  path, "seed_", seed_, "_", model, "_", sim_, "_", task_,
  "_n_", n_, "_p_", p_, "_ntree_", n_tree_, "_mvndim_", mvn_dim_, ".Rds"
))
