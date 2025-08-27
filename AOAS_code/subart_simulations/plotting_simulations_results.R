rm(list = ls(all = TRUE))

# -------------------------------
# WARNING: Users must run the code in 'cv_results.R' to obtain results for plotting using the present script
# -------------------------------

# -------------------------------
# Packages
# -------------------------------
library(cowplot)
library(tidyverse)

# -------------------------------
# Plotting results
# -------------------------------

# -------------------------------
# Settings
# -------------------------------
n_ <- 1000 # set to n_ <- 250 or n_ <- 500 to consider other settings in the paper
p_ <- 10
n_tree_ <- 100
mvn_dim_ <- 3 # set to mvn_dim_ <- 2 to consider other setting in the paper
seed_ <- 43
task_ <- "classification" # 'classification' or 'regression'
sim_ <- "friedman1" # 'friedman1' or 'friedman2'

set.seed(43)

# -------------------------------
# Models to process
# -------------------------------
models <- switch(paste(task_, mvn_dim_, sep = "_"),
  "classification_2" = c("BayesSUR", "subart", "bart"),
  "classification_3" = c("BayesSUR", "subart", "bart"),
  "regression_2" = c("BayesSUR", "subart", "bart", "mvBART"),
  "regression_3" = c("BayesSUR", "subart", "bart"),
  stop("No valid setting")
)

results_path <- "/subart_simulations/"
save_plots_path <- "/subart_simulations/"

if (!(file.exists(results_path) && file.info(results_path)$isdir)) {
  stop("Insert a valid directory path from which to load saved results")
}

if (!(file.exists(save_plots_path) && file.info(save_plots_path)$isdir)) {
  stop("Insert a valid directory path fto save plots to")
}

# -------------------------------
# Load results
# -------------------------------
result_df <- result_df_corr <- data.frame()

for (model in models) {
  file_ntree <- ifelse(model == "BayesSUR", 100, n_tree_)
  file_path <- paste0(
    results_path, "seed_", seed_, "_", model, "_", sim_, "_", task_,
    "_n_", n_, "_p_", p_, "_ntree_", file_ntree, "_mvndim_", mvn_dim_, ".Rds"
  )

  result <- readRDS(file_path)
  result_df <- rbind(result_df, lapply(result, function(x) x$comparison_metrics) %>% bind_rows())
  result_df_corr <- rbind(result_df_corr, lapply(result, function(x) x$correlation_metrics) %>% bind_rows())
}

# -------------------------------
# Factor adjustments
# -------------------------------
result_df <- result_df %>%
  mutate(model = ifelse(model == "BayesSUR", "BayesSUR", model)) %>%
  mutate(model = factor(model, levels = c("suBART", "BART", "mvBART", "BayesSUR"))) %>%
  mutate(mvn_dim = paste0("j=", mvn_dim)) %>%
  mutate(mvn_dim = factor(mvn_dim, levels = c("j=1", "j=2", "j=3")))
text_size <- 16

# -------------------------------
# Metric plots
# -------------------------------
if (task_ == "regression") {
  metrics <- list(
    rmse = list(name = "rmse_test", ylab = "RMSE"),
    crps = list(name = "crps_test", ylab = "CRPS"),
    pi   = list(name = "pi_test", ylab = "PI coverage", hline = 0.5)
  )
} else if (task_ == "classification") {
  metrics <- list(
    logloss = list(name = "logloss_test", ylab = "Log.Loss"),
    acc     = list(name = "acc_test", ylab = "ACC"),
    ci      = list(name = "p_cr_test", ylab = "CI coverage", hline = 0.5)
  )
} else {
  stop("Invalid task_ to generate a plot")
}

plot_list <- list()
for (m in names(metrics)) {
  dat <- result_df %>% filter(metric == metrics[[m]]$name)
  ylab_ <- metrics[[m]]$ylab
  hline <- metrics[[m]]$hline

  p <- ggplot(dat, aes(x = model, y = value, col = model)) +
    geom_boxplot(show.legend = FALSE) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
    scale_color_manual(values = c(
      "suBART" = "#F8766D", "BART" = "#00BA38",
      "mvBART" = "#C77CFF", "BayesSUR" = "#619CFF"
    )) +
    ylab(ylab_) +
    xlab("") +
    facet_wrap(~mvn_dim, scales = "free_y") +
    theme_classic() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      text = element_text(size = text_size)
    )

  if (!is.null(hline)) {
    p <- p + geom_hline(yintercept = hline, lty = "dashed", col = "black")
  }

  plot_list[[m]] <- p
}

# -------------------------------
# Legend plot
# -------------------------------
legend_plot <- ggplot(
  result_df %>% filter(metric == names(metrics)[1]),
  aes(x = model, y = value, col = model)
) +
  geom_boxplot() +
  scale_color_manual(values = c(
    "suBART" = "#F8766D", "BART" = "#00BA38",
    "mvBART" = "#C77CFF", "BayesSUR" = "#619CFF"
  )) +
  theme_classic() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    text = element_text(size = text_size),
    legend.position = "bottom"
  ) +
  labs(color = "Model:")

legend <- cowplot::get_plot_component(legend_plot, pattern = "guide-box-bottom", return_all = TRUE)

main_plot <- cowplot::plot_grid(cowplot::plot_grid(plot_list[[1]], plot_list[[2]], plot_list[[3]], ncol = 3),
  legend,
  nrow = 2, rel_heights = c(1, 0.1)
)

# -------------------------------
# Export plot
# -------------------------------
width_inches <- 15
height_inches <- 5

ggsave(paste0(save_plots_path, sim_, "_", task_, "_", n_, "_", mvn_dim_, "_n_tree_", n_tree_, ".pdf"),
  plot = main_plot, width = width_inches, height = height_inches
)

# -------------------------------
# Summary of correlations
# -------------------------------
summary_df_corr_cov <- result_df_corr %>%
  filter(metric == "cr_cov") %>%
  group_by(param_index, model) %>%
  summarise(mean_cv = mean(value)) %>%
  pivot_wider(names_from = model, values_from = mean_cv) %>%
  mutate(metric = "PI coverage")

summary_df_corr_rmse <- result_df_corr %>%
  filter(metric == "rmse") %>%
  group_by(param_index, model) %>%
  summarise(mean_cv = mean(value)) %>%
  pivot_wider(names_from = model, values_from = mean_cv) %>%
  mutate(metric = "RMSE")

summary_sd_df_corr_rmse <- result_df_corr %>%
  filter(metric == "rmse") %>%
  group_by(param_index, model) %>%
  summarise(mean_cv = sd(value)) %>%
  pivot_wider(names_from = model, values_from = mean_cv) %>%
  mutate(metric = "sd_RMSE")

summary_corr_comparison <- bind_rows(
  summary_df_corr_cov,
  summary_df_corr_rmse,
  summary_sd_df_corr_rmse
)

summary_corr_comparison

# ------------------------------- #
# ------------------------------- #
# ------------------------------- #
