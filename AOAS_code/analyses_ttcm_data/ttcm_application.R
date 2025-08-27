rm(list = ls(all = TRUE))

# ==========================
# Packages
# ==========================
# devtools::install_github("MateusMaiaDS/subart")
# devtools::install_github("Seungha-Um/skewBART")

library(BART)
library(bayesplot)
library(dplyr)
library(ggdensity)
library(ggplot2)
library(gridExtra)
library(skewBART)
library(subart)
library(surbayes)
library(systemfit)

# ==========================
# Helper functions
# ==========================
compute_delta <- function(y_hat_test, trt) {
  n_post <- dim(y_hat_test)[3]
  delta <- matrix(NA, nrow = n_post, ncol = 2)
  for (i in 1:n_post) {
    delta[i, ] <- c(
      mean(y_hat_test[trt == 1, 1, i]) - mean(y_hat_test[trt == 0, 1, i]),
      mean(y_hat_test[trt == 1, 2, i]) - mean(y_hat_test[trt == 0, 2, i])
    )
  }
  return(delta)
}

compute_post <- function(delta_matrix) {
  post <- data.frame(
    Delta_c = delta_matrix[, 1],
    Delta_q = delta_matrix[, 2]
  )
  post$INB20 <- 20000 * post$Delta_q - post$Delta_c
  post$INB50 <- 50000 * post$Delta_q - post$Delta_c
  return(post)
}

plot_CE_plane <- function(post, title) {
  ggplot(post) +
    geom_point(aes(Delta_q, Delta_c), size = 1, alpha = 0.05, show.legend = FALSE) +
    geom_hdr_lines(aes(Delta_q, Delta_c), probs = c(0.5, 0.75, 0.9, 0.95), alpha = 1, linewidth = 0.5) +
    annotate("point", x = mean(post$Delta_q), y = mean(post$Delta_c), color = "red", shape = 15, size = 2) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(x = expression(Delta[q]), y = expression(Delta[c]), title = title) +
    theme_classic() +
    theme(text = element_text(size = 14)) +
    scale_x_continuous(limits = c(-0.1, 0.3)) +
    scale_y_continuous(limits = c(-4000, 2000))
}

# ==========================
# Data preparation
# ==========================
data("ttcm")
X_train <- select(ttcm, -one_of(c("Q", "C")))
Y_train <- select(ttcm, "C", "Q")

# Duplicate training data for testing
X_test <- rbind(X_train, X_train)
X_test$trt <- c(rep(0, nrow(X_train)), rep(1, nrow(X_train)))

# ==========================
# Set number of MCMC draws (used for all model fits below)
# ==========================
n_post <- 4000
n_burn <- 1000

# ==========================
# Propensity score estimation
# ==========================
ps_fit <- subart(
  x_train = select(X_train, !c("trt")),
  y_train = as.matrix(X_train$trt),
  n_mcmc = n_burn + n_post,
  n_burn = n_burn,
  n_tree = 100
)

# Create training and test data with PS
X_train_ps <- X_train
X_train_ps$ps <- apply(pnorm(ps_fit$y_hat), 1, mean)

X_test_ps <- rbind(X_train_ps, X_train_ps)
X_test_ps$trt <- c(rep(0, nrow(X_train_ps)), rep(1, nrow(X_train_ps)))

# Plot PS distributions (Figure 4)
ggplot() +
  geom_histogram(data = filter(X_train_ps, trt == 0), aes(x = ps), bins = 50, fill = "blue", alpha = 0.5) +
  geom_histogram(data = filter(X_train_ps, trt == 1), aes(x = ps, y = -after_stat(count)), bins = 50, fill = "red", alpha = 0.5) +
  geom_hline(yintercept = 0) +
  annotate("label", x = 0.05, y = 3, label = "t = 0", fill = "blue", alpha = 0.5, color = "white", hjust = 1, size = 5) +
  annotate("label", x = 0.05, y = -3, label = "t = 1", fill = "red", alpha = 0.5, color = "white", hjust = 1, size = 5) +
  scale_y_continuous(labels = abs) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-8, 5)) +
  labs(x = "Propensity score", y = "Count") +
  theme_classic() +
  theme(text = element_text(size = 12))

# ==========================
# Model fitting & post-processing
# ==========================
# suBART without PS
suBART_fit <- subart(
  x_train = X_train,
  y_train = Y_train,
  x_test = X_test,
  n_tree = 100,
  n_mcmc = n_burn + n_post,
  n_burn = n_burn,
  varimportance = FALSE
)
suBART_fit$Delta <- compute_delta(suBART_fit$y_hat_test, X_test$trt)
post_suBART <- compute_post(suBART_fit$Delta)
CE_plane_suBART <- plot_CE_plane(post_suBART, "suBART")

# suBART with PS
suBART_ps_fit <- subart(
  x_train = X_train_ps,
  y_train = Y_train,
  x_test = X_test_ps,
  n_tree = 100,
  n_mcmc = n_burn + n_post,
  n_burn = n_burn,
  nu = 2,
  varimportance = TRUE
)
suBART_ps_fit$Delta <- compute_delta(suBART_ps_fit$y_hat_test, X_test_ps$trt)
post_suBART_ps <- compute_post(suBART_ps_fit$Delta)
CE_plane_suBART_ps <- plot_CE_plane(post_suBART_ps, "ps-suBART")

# mvBART without PS
X_train_mvBART <- model.matrix(~ . - 1, data = X_train)
X_test_mvBART <- model.matrix(~ . - 1, data = X_test)
hypers <- Hypers(X = X_train_mvBART, Y = Y_train, num_tree = 200)
opts <- Opts(num_burn = n_burn, num_save = n_post)
mvBART_fit <- MultiskewBART(
  X = X_train_mvBART, Y = Y_train, test_X = X_test_mvBART,
  do_skew = FALSE, hypers = hypers, opts = opts
)

mvBART_fit$Delta <- compute_delta(mvBART_fit$y_hat_test, X_test$trt)
post_mvBART <- compute_post(mvBART_fit$Delta)
CE_plane_mvBART <- plot_CE_plane(post_mvBART, "mvBART")

# mvBART with PS
X_train_mvBART_ps <- model.matrix(~ . - 1, data = X_train_ps)
X_test_mvBART_ps <- model.matrix(~ . - 1, data = X_test_ps)
hypers <- Hypers(X = X_train_mvBART_ps, Y = Y_train, num_tree = 200)
opts <- Opts(num_burn = n_burn, num_save = n_post)
mvBART_ps_fit <- MultiskewBART(
  X = X_train_mvBART_ps, Y = Y_train, test_X = X_test_mvBART_ps,
  do_skew = FALSE, hypers = hypers, opts = opts
)

mvBART_ps_fit$Delta <- compute_delta(mvBART_ps_fit$y_hat_test, X_test_ps$trt)
post_mvBART_ps <- compute_post(mvBART_ps_fit$Delta)
CE_plane_mvBART_ps <- plot_CE_plane(post_mvBART_ps, "ps-mvBART")

# BayesSUR without PS
BayesSUR_fit <- sur_sample(
  formula.list = list(
    C ~ . - C - Q,
    Q ~ . - C - Q
  ),
  data = cbind(X_train, Y_train),
  M = n_post
)

BayesSUR_fit$Delta <- data.frame(
  Delta_c = BayesSUR_fit$betadraw[, "1.trt"],
  Delta_q = BayesSUR_fit$betadraw[, "2.trt"]
)
post_BayesSUR <- compute_post(BayesSUR_fit$Delta)
CE_plane_BayesSUR <- plot_CE_plane(post_BayesSUR, "BayesSUR")

# BayesSUR with PS
BayesSUR_ps_fit <- sur_sample(
  formula.list = list(
    C ~ . - C - Q,
    Q ~ . - C - Q
  ),
  data = cbind(X_train_ps, Y_train),
  M = n_post
)

BayesSUR_ps_fit$Delta <- data.frame(
  Delta_c = BayesSUR_ps_fit$betadraw[, "1.trt"],
  Delta_q = BayesSUR_ps_fit$betadraw[, "2.trt"]
)
post_BayesSUR_ps <- compute_post(BayesSUR_ps_fit$Delta)
CE_plane_BayesSUR_ps <- plot_CE_plane(post_BayesSUR_ps, "ps-BayesSUR")

# ==========================
# Cost-effectiveness planes (CEP; Figure 5)
# ==========================
grid.arrange(CE_plane_suBART_ps,
  CE_plane_mvBART_ps,
  CE_plane_BayesSUR_ps,
  CE_plane_suBART,
  CE_plane_mvBART,
  CE_plane_BayesSUR,
  nrow = 2
)

# ==========================
# Cost-effectiveness acceptability curve (CEAC; Figure 6)
# ==========================
compute_CEAC <- function(post_list, lambda) {
  CEAC_df <- data.frame(
    lambda = rep(lambda, length(post_list)),
    p = unlist(lapply(post_list, function(post) sapply(lambda, function(l) mean(l * post$Delta_q - post$Delta_c > 0)))),
    Model = factor(rep(names(post_list), each = length(lambda)))
  )
  return(CEAC_df)
}

lambda <- seq(0, 50000, by = 10)
post_list <- list(
  "suBART" = post_suBART,
  "ps-suBART" = post_suBART_ps,
  "mvBART" = post_mvBART,
  "ps-mvBART" = post_mvBART_ps,
  "BayesSUR" = post_BayesSUR,
  "ps-BayesSUR" = post_BayesSUR_ps
)
CEAC <- compute_CEAC(post_list, lambda)
CEAC_points <- CEAC[CEAC$lambda %% 10000 == 0 & CEAC$lambda > 0, ]

ggplot() +
  geom_line(data = CEAC, aes(x = lambda, y = p, color = Model, linetype = Model), linewidth = 0.8) +
  geom_point(data = CEAC_points, aes(x = lambda, y = p, color = Model, shape = Model), size = 2.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_hline(yintercept = 0.7, linetype = "solid") +
  geom_hline(yintercept = 0.95, linetype = "dotted") +
  labs(x = expression(lambda), y = "Probability of cost-effectiveness") +
  scale_x_continuous(labels = paste0((0:5) * 10, "k"), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.7, 1), expand = c(0, 0)) +
  scale_shape_manual(values = c(16, 16, 17, 17, 15, 15)) +
  scale_linetype_manual(values = c("solid", "longdash", "solid", "longdash", "solid", "longdash")) +
  scale_colour_manual(values = c("#F8766D", "#F8766D", "#00BA38", "#00BA38", "#619CFF", "#619CFF")) +
  theme_classic() +
  theme(text = element_text(size = 12), axis.line = element_blank(), legend.position = "right", legend.key.width = unit(4, "line"))

# ==========================
# Summary table (Table 5)
# ==========================
compute_summary <- function(post, method) {
  data.frame(
    method = method,
    Delta_c = paste0(round(mean(post$Delta_c), 0), " [", round(quantile(post$Delta_c, 0.025), 0), ", ", round(quantile(post$Delta_c, 0.975), 0), "]"),
    Delta_q = paste0(round(mean(post$Delta_q), 3), " [", round(quantile(post$Delta_q, 0.025), 3), ", ", round(quantile(post$Delta_q, 0.975), 3), "]"),
    INB20 = paste0(round(mean(post$INB20), 0), " [", round(quantile(post$INB20, 0.025), 0), ", ", round(quantile(post$INB20, 0.975), 0), "]"),
    INB50 = paste0(round(mean(post$INB50), 0), " [", round(quantile(post$INB50, 0.025), 0), ", ", round(quantile(post$INB50, 0.975), 0), "]"),
    ICER = mean(post$Delta_c) / mean(post$Delta_q)
  )
}

results <- rbind(
  compute_summary(post_suBART, "suBART"),
  compute_summary(post_suBART_ps, "suBART_PS"),
  compute_summary(post_mvBART, "mvBART"),
  compute_summary(post_mvBART_ps, "mvBART_PS"),
  compute_summary(post_BayesSUR, "BayesSUR"),
  compute_summary(post_BayesSUR_ps, "ps-BayesSUR")
)
results

# ==========================
# CATE calculation
# ==========================
data_CATE <- X_train_ps
data_CATE$tau_c <- suBART_ps_fit$y_hat_test_mean[X_test_ps$trt == 1, 1] - suBART_ps_fit$y_hat_test_mean[X_test_ps$trt == 0, 1]
data_CATE$tau_q <- suBART_ps_fit$y_hat_test_mean[X_test_ps$trt == 1, 2] - suBART_ps_fit$y_hat_test_mean[X_test_ps$trt == 0, 2]
data_CATE$CINB20 <- 20000 * data_CATE$tau_q - data_CATE$tau_c

# ==========================
# Generic CATE plot function
# ==========================
plot_CATE <- function(data, x_var, y_vars, x_label = NULL) {
  plots <- lapply(y_vars, function(y_var) {
    ggplot(data) +
      geom_point(aes(!!sym(x_var), !!sym(y_var))) +
      labs(x = x_label %||% x_var, y = y_var) +
      theme_classic() +
      theme(text = element_text(size = 12))
  })
  grid.arrange(grobs = plots, nrow = 1)
}

# ==========================
# Plot CATE vs propensity score (Figure D.1)
# ==========================
plot_CATE(data_CATE, x_var = "ps", y_vars = c("tau_c", "tau_q", "CINB20"), x_label = "Propensity score")

# ==========================
# Plot CATE vs TTO (Figure D.2)
# ==========================
plot_CATE(data_CATE, x_var = "TTO", y_vars = c("tau_c", "tau_q", "CINB20"), x_label = "TTO")

# ==========================
# Plot CATE vs surgery status (Figure D.3)
# ==========================
plot_CATE(data_CATE, x_var = "surgery", y_vars = c("tau_c", "tau_q", "CINB20"), x_label = "Surgery status")

# ==========================
# Variable importance (Figure D.4)
# ==========================
varimp_c <- data.frame(
  var = colnames(X_train_ps),
  prop = apply(suBART_ps_fit$var_importance[, , 1], 2, mean) / 100
)

varimp_c_plot <- ggplot(varimp_c) +
  geom_point(aes(prop, reorder(var, prop))) +
  geom_segment(aes(x = 0, xend = prop, y = reorder(var, prop), yend = reorder(var, prop))) +
  theme_classic() +
  labs(x = "Percent usage", y = "") +
  theme(text = element_text(size = 12))

varimp_q <- data.frame(
  var = colnames(X_train_ps),
  prop = apply(suBART_ps_fit$var_importance[, , 2], 2, mean) / 100
)

varimp_q_plot <- ggplot(varimp_q) +
  geom_point(aes(prop, reorder(var, prop))) +
  geom_segment(aes(x = 0, xend = prop, y = reorder(var, prop), yend = reorder(var, prop))) +
  theme_classic() +
  labs(x = "Percent usage", y = "") +
  theme(text = element_text(size = 12))

grid.arrange(varimp_c_plot, varimp_q_plot, nrow = 1)

# ========================== #
# ========================== #
# ========================== #
