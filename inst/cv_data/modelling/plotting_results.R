# Plotting results
library(tidyverse)
rm(list=ls())
devtools::load_all()
set.seed(42)
n_ <- 1000
p_ <- 10
n_tree_ <- 50
mvn_dim_ <- 3
df_ <- 2
task_ <- "regression" # For this it can be either 'classification' or 'regression'
sim_ <- "friedman2" # For this can be either 'friedman1' or 'friedman2'

result <- readRDS(paste0("inst/cv_data/",task_,"/result_new/mvbart6_",
                         sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,"_df_",df_,".Rds"))

result_STAN <- readRDS(paste0("inst/cv_data/",task_,"/result/STAN_",
                              sim_,"_",task_,"_n_",n_,"_p_",p_,"_ntree_",n_tree_,"_mvndim_",mvn_dim_,".Rds"))


library(cowplot)
# Plotting first the RMSE for the test sample
result_df <- lapply(result,function(x){x$comparison_metrics}) %>% do.call(rbind,.) %>% filter(model!="SUR")
result_df_corr <- lapply(result,function(x){x$correlation_metrics}) %>% do.call(rbind,.) %>% filter(model!="SUR")
result_df_STAN <- lapply(result_STAN,function(x){x$comparison_metrics}) %>% do.call(rbind,.)
result_df_corr_STAN <- lapply(result_STAN,function(x){x$correlation_metrics}) %>% do.call(rbind,.)

# Merging both models
result_df <- rbind(result_df,result_df_STAN)
result_df_corr <- rbind(result_df_corr, result_df_corr_STAN)

# Changing the factor
if(task_=="regression"){
     result_df <- result_df %>% mutate(model = ifelse(model=="mvBART","suBART",model)) %>% mutate(model = factor(model,levels = c("suBART","BART", "bayesSUR")))


     if(mvn_dim_==2) {
          text_size <- 15
     } else if(mvn_dim_ == 3){
          text_size <- 20
     }
     rmse_plot <- result_df %>% filter(metric == "rmse_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model),show.legend = FALSE)+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+ # Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          ylab("RMSE")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))

     # Just a decoy to get a legend
     rmse_plot_legend <- result_df %>% filter(metric == "rmse_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model))+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+# Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          ylab("RMSE")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))

     rmse_plot_legend <- rmse_plot_legend + labs(color = "Model:")

     crps_plot <- result_df %>% filter(metric == "crps_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = (value),col = model),show.legend = FALSE)+
          # scale_y_log10(labels = scales::number_format(accuracy = 0.01))+# Maybe use the log-scale
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
          ylab("CRPS")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))


     pi_plot <- result_df %>% filter(metric == "pi_test") %>%
          mutate(mvn_dim = as.factor(mvn_dim)) %>%
          ggplot()+
          geom_boxplot(mapping = aes(x = model, y = value,col = model),show.legend = FALSE)+
          scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+ #
          geom_hline(yintercept = 0.5, lty = 'dashed', col = 'blue')+
          ylab("PI coverage")+
          xlab("")+
          facet_wrap(~mvn_dim,scales = "free_y")+
          theme_classic()+
          theme(axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                text = element_text(size = text_size))

     legend <- get_legend(rmse_plot_legend + theme(legend.position = "bottom"))

     main_plot <-  cowplot::plot_grid(cowplot::plot_grid(rmse_plot,crps_plot,pi_plot,ncol = 3),
                                      legend,nrow = 2,
                                      rel_heights = c(1, 0.1))  # Adjust legend width as needed

     main_plot

     # Set fixed width and height for the plot
     if(mvn_dim_==2){
          width_inches <- 12  # Adjust as needed
          height_inches <- 5  # Adjust as needed
     } else if(mvn_dim_==3){
          width_inches <- 18  # Adjust as needed
          height_inches <- 5  # Adjust as needed
     } else {
          stop("Enter valid mvn_dim")
     }

     # # Export as TIFF
     # # (attention change your path here)
     ggsave(paste0("/localusers/researchers/mmarques/spline_bart_lab/mvbart6_plots/",sim_,"_",task_,"_",n_,"_",mvn_dim_,".tiff"), plot = main_plot,
            width = width_inches, height = height_inches, dpi = 300)

     # Export as PDF
     ggsave(paste0("/localusers/researchers/mmarques/spline_bart_lab/mvbart6_plots/",sim_,"_",task_,"_",n_,"_",mvn_dim_,".pdf"), plot = main_plot,
            width = width_inches, height = height_inches)
} # End of iteration

# Plotting results for the CR coverage
summary_df_corr_cov <-result_df_corr %>% filter(metric == "cr_cov") %>%  group_by(param_index,model) %>%
     summarise(mean_cv = mean(value)) %>% pivot_wider(names_from = model, values_from = mean_cv) %>%
     mutate(metric = "PI coverage")

summary_df_corr_rmse <- result_df_corr %>% filter(metric == "rmse") %>%  group_by(param_index,model) %>%
     summarise(mean_cv = sqrt(mean(value^2))) %>% print(n = 24 )%>% pivot_wider(names_from = model, values_from = mean_cv) %>%
     mutate(metric = "RMSE")

summary_corr_comparison <- rbind(summary_df_corr_cov,
                                 summary_df_corr_rmse) %>%
     dplyr::filter(stringr::str_detect(param_index,"rho"))
summary_corr_comparison

