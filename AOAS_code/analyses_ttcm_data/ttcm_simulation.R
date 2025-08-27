rm(list = ls(all = TRUE))

# ==========================
# Packages
# ==========================
# devtools::install_github("MateusMaiaDS/subart")
# devtools::install_github("Seungha-Um/skewBART")

library(BART)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(mvtnorm)
library(progress)
library(skewBART)
library(subart)
library(surbayes)

# ==========================
# Define simulation
# ==========================

# load & preprocess
data("ttcm")
standardise <- function(x) (x - mean(x)) / sd(x)

X <- data.frame(
  gender = as.integer(ttcm$gender) - 1,
  age = standardise(ttcm$age),
  education = ttcm$education,
  disease_history = ttcm$disease_history,
  trauma_type = ttcm$trauma_type,
  fracture_region = ttcm$fracture_region,
  ISS = standardise(ttcm$ISS),
  hospital_duration = ttcm$hospital_duration,
  surgery = as.integer(ttcm$surgery) - 1,
  TTO = standardise(ttcm$TTO),
  admission = as.integer(as.factor(ttcm$admission)) - 1
)

# prognostic functions and treatment effects
mu_C <- function(gender, age, education, TTO, surgery) {
  return(2000 + 500 * age + (-200) * education + 500 * surgery)
}

tau_C <- function(gender, age, education, TTO, surgery) {
  return(500)
}

mu_Q <- function(gender, age, education, TTO, surgery) {
  return(0.5 + 0.2 * sin(age) * (gender + 1))
}

tau_Q <- function(gender, age, education, TTO, surgery) {
  return(-0.1 + 0.1 * exp(-TTO))
}

# true expected potential outcomes and propensity scores
data_true <- data.frame(
  mu_C  = mu_C(X$gender, X$age, X$education, X$TTO, X$surgery),
  tau_C = tau_C(X$gender, X$age, X$education, X$TTO, X$surgery),
  mu_Q  = mu_Q(X$gender, X$age, X$education, X$TTO, X$surgery),
  tau_Q = tau_Q(X$gender, X$age, X$education, X$TTO, X$surgery)
)
data_true$ps <- 0.9 * pnorm(-0.5 + 1 * X$surgery - 1.5 * standardise(data_true$mu_Q)) + 0.05

# true average treatment effects
Delta_C_true <- mean(data_true$tau_C)
Delta_Q_true <- mean(data_true$tau_Q)
INB_20_true <- 20000 * Delta_Q_true - Delta_C_true
INB_50_true <- 50000 * Delta_Q_true - Delta_C_true

# simulation function
sim <- function(data_true, rho) {
  n <- nrow(data_true)
  trt <- rbinom(n, 1, data_true$ps)
  Sigma <- matrix(c(500^2, 500 * 0.05 * rho, 500 * 0.05 * rho, 0.05^2), 2)
  eps <- mvtnorm::rmvnorm(n, sigma = Sigma)
  data.frame(
    trt = trt,
    C   = data_true$mu_C + trt * data_true$tau_C + eps[, 1],
    Q   = data_true$mu_Q + trt * data_true$tau_Q + eps[, 2]
  )
}

# ==========================
# Make simulated datasets
# - To replicate the results in the main paper, set rho = -0.25.
# - To replicate the results in the supplementary material, set rho = 0/-0.25/-0.5, respectively
# ==========================

n_sim <- 1000
rho <- -0.25 # choose rho = -0.25 for replication
data_sim_list <- replicate(n_sim, sim(data_true, rho), simplify = FALSE)

# ==========================
# Analyse simulated datasets
# - WARNING: the loop below takes a long time co complete. The results files with which to
#   replicate the tables in the paper are available on GitHub. See the end of this script.
# ==========================

# --- Helper functions ---
make_results_df <- function(n_sim) {
  cols <- c(
    "Delta_C_mean", "Delta_C_CI_lower", "Delta_C_CI_upper", "Delta_C_CI_length",
    "Delta_Q_mean", "Delta_Q_CI_lower", "Delta_Q_CI_upper", "Delta_Q_CI_length",
    "INB_20_mean", "INB_20_CI_lower", "INB_20_CI_upper", "INB_20_CI_length",
    "INB_50_mean", "INB_50_CI_lower", "INB_50_CI_upper", "INB_50_CI_length"
  )
  setNames(as.data.frame(matrix(NA, nrow = n_sim, ncol = length(cols))), cols)
}

compute_inb <- function(df) {
  df$INB_20 <- 20000 * df$Delta_Q - df$Delta_C
  df$INB_50 <- 50000 * df$Delta_Q - df$Delta_C
  df
}

save_results <- function(model, i, samples, summary) {
  results_samples[[model]][[i]] <<- samples
  results_summary[[model]][i, ] <<- posterior_summary(samples)
}

extract_deltas <- function(fit, n) {
  data.frame(
    Delta_C = apply(fit$y_hat_test[(n + 1):(2 * n), 1, ] - fit$y_hat_test[1:n, 1, ], 2, mean),
    Delta_Q = apply(fit$y_hat_test[(n + 1):(2 * n), 2, ] - fit$y_hat_test[1:n, 2, ], 2, mean)
  )
}

posterior_summary <- function(posterior) {
  Delta_C <- posterior$Delta_C
  Delta_Q <- posterior$Delta_Q
  INB_20 <- posterior$INB_20
  INB_50 <- posterior$INB_50
  return(
    c(
      mean(Delta_C),
      quantile(Delta_C, 0.25),
      quantile(Delta_C, 0.75),
      quantile(Delta_C, 0.75) - quantile(Delta_C, 0.25),
      mean(Delta_Q),
      quantile(Delta_Q, 0.25),
      quantile(Delta_Q, 0.75),
      quantile(Delta_Q, 0.75) - quantile(Delta_Q, 0.25),
      mean(INB_20),
      quantile(INB_20, 0.25),
      quantile(INB_20, 0.75),
      quantile(INB_20, 0.75) - quantile(INB_20, 0.25),
      mean(INB_50),
      quantile(INB_50, 0.25),
      quantile(INB_50, 0.75),
      quantile(INB_50, 0.75) - quantile(INB_50, 0.25)
    )
  )
}

# Create empty results objects
models <- c(
  "suBART", "mvBART", "BayesSUR",
  "suBART_ps", "mvBART_ps", "BayesSUR_ps", "indBART_ps"
)
results_summary <- setNames(lapply(models, function(x) make_results_df(n_sim)), models)
results_samples <- list()

# --- Inference loop ---
n_post <- 2000
n_burn <- 1000
n_obs <- nrow(data_true)
pb <- progress_bar$new(total = n_sim)
for (i in seq_len(n_sim)) {
  pb$tick()

  # --- Data prep ---
  data_sim <- data_sim_list[[i]]
  X_train <- cbind(X, data_sim$trt)
  colnames(X_train) <- c(colnames(X), "trt")
  X_test <- rbind(X_train, X_train)
  X_test$trt <- c(rep(0, nrow(X_train)), rep(1, nrow(X_train)))
  Y_train <- as.matrix(data_sim[, c("C", "Q")])

  # --- suBART ---
  suBART_fit <- subart(
    x_train = X_train, y_train = Y_train, x_test = X_test,
    n_mcmc = n_burn + n_post, n_burn = n_burn, n_tree = 100
  )
  df <- extract_deltas(suBART_fit, n_obs) |> compute_inb()
  save_results("suBART", i, df, results_summary)

  # --- mvBART ---
  X_train_mvBART <- model.matrix(~ . - 1, data = X_train)
  X_test_mvBART <- model.matrix(~ . - 1, data = X_test)
  hypers <- Hypers(X = X_train_mvBART, Y = Y_train, num_tree = 200)
  opts <- Opts(num_burn = n_burn, num_save = n_post)
  mvBART_fit <- MultiskewBART(
    X = X_train_mvBART, Y = Y_train, test_X = X_test_mvBART,
    do_skew = FALSE, hypers = hypers, opts = opts
  )

  df <- extract_deltas(mvBART_fit, n_obs) |> compute_inb()
  save_results("mvBART", i, df, results_summary)

  # --- BayesSUR ---
  BayesSUR_fit <- sur_sample(
    formula.list = list(
      C ~ . - C - Q,
      Q ~ . - C - Q
    ),
    data = cbind(X_train, Y_train),
    M = n_post
  )
  df <- data.frame(
    Delta_C = BayesSUR_fit$betadraw[, "1.trt"],
    Delta_Q = BayesSUR_fit$betadraw[, "2.trt"]
  ) |> compute_inb()
  save_results("BayesSUR", i, df, results_summary)

  # --- Propensity score ---
  ps_fit <- subart(
    x_train = select(X_train, !c("trt")),
    y_train = as.matrix(X_train$trt),
    n_mcmc  = n_burn + n_post,
    n_burn  = n_burn,
    n_tree  = 100
  )
  X_train_ps <- X_train
  X_train_ps$ps <- apply(pnorm(ps_fit$y_hat), 1, mean)
  X_test_ps <- rbind(X_train_ps, X_train_ps)
  X_test_ps$trt <- c(rep(0, nrow(X_train_ps)), rep(1, nrow(X_train_ps)))

  # suBART_ps
  suBART_ps_fit <- subart(
    x_train = X_train, y_train = Y_train, x_test = X_test,
    n_mcmc = n_burn + n_post, n_burn = n_burn, n_tree = 100
  )
  df <- extract_deltas(suBART_ps_fit, n_obs) |> compute_inb()
  save_results("suBART_ps", i, df, results_summary)

  # mvBART_ps
  X_train_mvBART_ps <- model.matrix(~ . - 1, data = X_train_ps)
  X_test_mvBART_ps <- model.matrix(~ . - 1, data = X_test_ps)
  hypers <- Hypers(X = X_train_mvBART_ps, Y = Y_train, num_tree = 200)
  opts <- Opts(num_burn = n_burn, num_save = n_post)
  mvBART_ps_fit <- MultiskewBART(
    X = X_train_mvBART_ps, Y = Y_train, test_X = X_test_mvBART_ps,
    do_skew = FALSE, hypers = hypers, opts = opts
  )

  df <- extract_deltas(mvBART_ps_fit, n_obs) |> compute_inb()
  save_results("mvBART_ps", i, df, results_summary)

  # BayesSUR_ps
  BayesSUR_ps_fit <- sur_sample(
    formula.list = list(
      C ~ . - C - Q,
      Q ~ . - C - Q
    ),
    data = cbind(X_train_ps, Y_train),
    M = n_post
  )
  df <- data.frame(
    Delta_C = BayesSUR_ps_fit$betadraw[, "1.trt"],
    Delta_Q = BayesSUR_ps_fit$betadraw[, "2.trt"]
  ) |> compute_inb()
  save_results("BayesSUR_ps", i, df, results_summary)

  # indBART_ps
  indBART_ps_c_fit <- subart(
    x_train = X_train, y_train = matrix(Y_train[, 1], ncol = 1), x_test = X_test,
    n_mcmc = n_burn + n_post, n_burn = n_burn, n_tree = 100
  )
  indBART_ps_q_fit <- subart(
    x_train = X_train, y_train = matrix(Y_train[, 2], ncol = 1), x_test = X_test,
    n_mcmc = n_burn + n_post, n_burn = n_burn, n_tree = 100
  )
  df <- data.frame(
    Delta_C = apply(indBART_ps_c_fit$y_hat_test[(n_obs + 1):(2 * n_obs), 1, ] - indBART_ps_c_fit$y_hat_test[1:n_obs, 1, ], 2, mean),
    Delta_Q = apply(indBART_ps_q_fit$y_hat_test[(n_obs + 1):(2 * n_obs), 1, ] - indBART_ps_q_fit$y_hat_test[1:n_obs, 1, ], 2, mean)
  ) |> compute_inb()
  save_results("indBART_ps", i, df, results_summary)
}


# ==========================
# Make tables in main paper and supplementary material
# ==========================

# --- helper to compute metrics for one estimand ---
compute_metrics <- function(res, est, true) {
  data.frame(
    bias = mean(res[[paste0(est, "_mean")]] - true),
    sd = sd(res[[paste0(est, "_mean")]]),
    RMSE = sqrt(mean((res[[paste0(est, "_mean")]] - true)^2)),
    coverage = mean(res[[paste0(est, "_CI_lower")]] < true &
      true < res[[paste0(est, "_CI_upper")]]),
    length = mean(res[[paste0(est, "_CI_upper")]] - res[[paste0(est, "_CI_lower")]])
  )
}

# --- define estimands and their true values ---
estimands <- list(
  Delta_C = Delta_C_true,
  Delta_Q = Delta_Q_true,
  INB_20  = INB_20_true,
  INB_50  = INB_50_true
)

# If you want to avoid running the whole loop above, you can instead load the needed file
# from GitHub by uncommenting the appropriate line.

# For Table A.1
# results_summary <- readRDS("C:/Users/jes238/OneDrive - Vrije Universiteit Amsterdam/Documents/suBART/summary_causal_experiments_rho_eq_0.rds")

# For Table A.2, Table 3, and Table 4
# results_summary <- readRDS("C:/Users/jes238/OneDrive - Vrije Universiteit Amsterdam/Documents/suBART/summary_causal_experiments_rho_eq_-0.25.rds")

# For Table A.3
# results_summary <- readRDS("C:/Users/jes238/OneDrive - Vrije Universiteit Amsterdam/Documents/suBART/summary_causal_experiments_rho_eq_-0.5.rds")

# --- build results ---
sim_results <- do.call(rbind, lapply(names(results_summary), function(model) {
  res <- results_summary[[model]]
  metrics <- do.call(cbind, lapply(names(estimands), function(est) {
    compute_metrics(res, est, estimands[[est]])
  }))
  data.frame(model, metrics, check.names = FALSE)
}))
colnames(sim_results) <- c(
  "model",
  "bias_Delta_C", "sd_Delta_C", "RMSE_Delta_C", "coverage_Delta_C", "length_Delta_C",
  "bias_Delta_Q", "sd_Delta_Q", "RMSE_Delta_Q", "coverage_Delta_Q", "length_Delta_Q",
  "bias_INB_20", "sd_INB_20", "RMSE_INB_20", "coverage_INB_20", "length_INB_20",
  "bias_INB_50", "sd_INB_50", "RMSE_INB_50", "coverage_INB_50", "length_INB_50"
)

# Table 3
select(
  filter(sim_results, model != "indBART_ps"),
  "model",
  "bias_Delta_C", "sd_Delta_C", "RMSE_Delta_C", "coverage_Delta_C", "length_Delta_C",
  "bias_Delta_Q", "sd_Delta_Q", "RMSE_Delta_Q", "coverage_Delta_Q", "length_Delta_Q"
)

# Table 4
select(
  filter(sim_results, model != "indBART_ps"),
  "model",
  "bias_INB_20", "sd_INB_20", "RMSE_INB_20", "coverage_INB_20", "length_INB_20",
  "bias_INB_50", "sd_INB_50", "RMSE_INB_50", "coverage_INB_50", "length_INB_50"
)

# Table A.1/A.2/A.3, depending on the choice of rho
filter(sim_results, model == "suBART_ps" | model == "indBART_ps")

# ========================== #
# ========================== #
# ========================== #
