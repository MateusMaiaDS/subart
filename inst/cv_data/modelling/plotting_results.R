# Plotting results
library(tidyverse)
rm(list=ls())
devtools::load_all()
set.seed(42)
n_ <- 250
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 2
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman1" # For this can be either 'friedman1' or 'friedman2'

result <- readRDS(paste0("~/spline_bart_lab/mvnbart6/inst/cv_data/regression/result/",
                         sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))

result_STAN <- readRDS(paste0("~/spline_bart_lab/mvnbart6/inst/cv_data/regression/result/STAN_",
                         sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))


library(cowplot)
# Plotting first the RMSE for the test sample
result_df <- lapply(result,function(x){x$comparison_metrics}) %>% do.call(rbind,.)
result_df_corr <- lapply(result,function(x){x$correlation_metrics}) %>% do.call(rbind,.)
result_df_STAN <- lapply(result_STAN,function(x){x$comparison_metrics}) %>% do.call(rbind,.)
result_df_corr_STAN <- lapply(result_STAN,function(x){x$correlation_metrics}) %>% do.call(rbind,.)

# Merging both models
result_df <- rbind(result_df,result_df_STAN)
result_df_corr <- rbind(result_df_corr, result_df_corr_STAN)

rmse_plot <- result_df %>% filter(metric == "rmse_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     ylab("RMSE test")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()

crps_plot <- result_df %>% filter(metric == "crps_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     ylab("CRPS test")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()


pi_plot <- result_df %>% filter(metric == "pi_test") %>%
     mutate(mvn_dim = as.factor(mvn_dim)) %>%
     ggplot()+
     geom_boxplot(mapping = aes(x = model, y = value))+
     geom_hline(yintercept = 0.5, lty = 'dashed', col = 'blue')+
     ylab("PI coverage")+
     xlab("Model")+
     facet_wrap(~mvn_dim,scales = "free_y")+
     theme_classic()

cowplot::plot_grid(rmse_plot,crps_plot,pi_plot,ncol = 3)

# Plotting results for the
result_df_corr %>% group_by(metric,param_index,model) %>% summarise(mean_cv = mean(value))
result_df_corr %>% filter(metric == "rmse") %>% ggplot()+
        geom_boxplot(mapping = aes( x = param_index, y = value, col = model))+
        theme_classic()
