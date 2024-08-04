#!/usr/bin/env Rscript
library(boot)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(lme4)
library(MASS)
library(brms)
library(ggplot2)
library(lme4)
library(lmerTest)
library(purrr)
library(lme4)
library(lmerTest)
library(JuliaCall)
library(plotrix)
library(viridis)
library(Rmisc)

set.seed(123)
theme_set(theme_light())
options(digits = 8)
options(dplyr.summarise.inform = TRUE)

library(jglmm)
options(JULIA_HOME = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin")
julia_setup(JULIA_HOME = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin")
jglmm_setup()

rm(list = ls())

reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun)
}

scale_x_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_x_discrete(labels = function(x) gsub(reg, "", x), guide = guide_axis(angle = 75), ...)
}


call_model <- function(df){
  model_lm <- jglmm(agree_renorm ~ (1|id) + 
    average_lex_freq_alternative + average_word_length_alternative + sentence_length_alternative +
      sentiment_pos_alternative + sentiment_neg_alternative, data = df)
  return(model_lm)
}

get_effect_sizes <- function(form, df, psychometric, model_name, score_name){
  model <- jglmm(as.formula(form), data = df)
  mod_summary <- broom:tidy(model)
  # extract effect sizes
  # fixed_score <- testsum[testsum$term == test_score, c("estimate", "std.error", "p.value")]
  interaction_term <- mod_summary[mod_summary$term == paste0(model_name, " & ", score_name), c("estimate", "std.error", "p.value")]
  return(interaction_term)
}

sig <- function(x) {
  return(ifelse(x > 0.05, "^{\\dag}", ""))
}

vircol <- viridis(9)[c(2, 7)]

## Prepare data
comparison_glm <- read_csv("data/final_df/all_models_gpt_paraphrases_glm.csv")
comparison_glm$lmtype <- "glm"
comparison_mlm <- read_csv("data/final_df/all_models_gpt_paraphrases_mlm.csv")
comparison_mlm$lmtype <- "mlm"
comparison_all <- bind_rows(comparison_glm, comparison_mlm)

alternatives_all <- comparison_all %>%
  filter(original_alternative == "alternative")

alternatives_all_preproc <- alternatives_all %>%
    mutate_at(vars(-original_statement, -alternative_statement, -original_alternative, -id, -model, -lmtype, -prompt), as.numeric)

# id as factor
alternatives_all_preproc$id <- as.factor(alternatives_all_preproc$id)
# lmtype as factor
alternatives_all_preproc$lmtype <- as.factor(alternatives_all_preproc$lmtype)

# first, reorder models
alternatives_all_preproc$model <- factor(
  alternatives_all_preproc$model, levels = c(
    "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o",
    "falcon-7b", "falcon-7b-instruct", "llama-7b", "llama-7b-chat", "llama-13b", "llama-13b-chat", "phi2", "tiny-llama-1b",
    "bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased", "distilroberta-base", "google", "roberta-base", "roberta-large")
)

# now rename model labels
alternatives_all_preproc$model <- recode(alternatives_all_preproc$model, "gpt-3.5-turbo" = "GPT-3.5", "gpt-4-turbo-preview" = "GPT-4", "gpt-4o" = "GPT-4o",
  "falcon-7b" = "Falcon-7B", "falcon-7b-instruct" = "Falcon-7B-Instruct", "llama-7b" = "Llama-2-7B", "llama-7b-chat" = "Llama-2-7B-Chat", "llama-13b" = "Llama-2-13B", "llama-13b-chat" = "Llama-2-13B-Chat", "phi2" = "Phi2", "tiny-llama-1b" = "Tiny-Llama",
  "bert-base-uncased" = "BERT-Base", "bert-large-uncased" = "BERT-Large", "distilbert-base-uncased" = "DistilBERT-Base", "distilroberta-base" = "DistilRoBERTa-Base", "google" = "Electra", "roberta-base" = "RoBERTa-Base", "roberta-large" = "RoBERTa-Large"
)

### Experiment 1a: Assess comprehensibility on group level
alternatives_all_preproc <- alternatives_all_preproc %>% 
  mutate(instruction_validity = agree_alternative + disagree_alternative) %>% 
  mutate(delta_agree = agree_alternative - disagree_alternative)  %>%
  mutate(agree_renorm = agree_alternative / (agree_alternative + disagree_alternative)) %>%
  mutate(agree_renorm_binary = ifelse(agree_renorm > 0.5, 1, 0)) %>%
  mutate(disagree_renorm_binary = ifelse(agree_renorm < 0.5, 1, 0)) 

# compute mean and sd for each model over all paraphrases
summary_df <- alternatives_all_preproc %>%
  group_by(model) %>%
  summarise_at(vars(instruction_validity), list(mean = mean, sd = std.error))

# reassign lmtype
summary_df <- summary_df %>%
    mutate(lmtype = ifelse(model %in% comparison_glm$model, "glm", "mlm"))

# Assess comprehensibility on statement level
summary_comp_micro <- alternatives_all_preproc %>%
  group_by(model, id) %>%
  summarise_at(vars(instruction_validity), list(mean.val = mean, sd.val = sd, se.val = std.error))

# reassign lmtype
summary_comp_micro <- summary_comp_micro %>%
  mutate(lmtype = ifelse(model %in% comparison_glm$model, "glm", "mlm"))

# sort df by lmtype
summary_comp_micro <- summary_comp_micro %>%
  arrange(lmtype, desc(mean.val))


# subset only glms 
glm_subset <- alternatives_all_preproc %>%
  filter(lmtype == "glm")
# remove gpt-3.5 turbo and gpt-4-preview
glm_subset_validity <- glm_subset %>%
  filter(model != "GPT-4" & model != "GPT-4o" & model != "GPT-3.5")

## Figure 4 (Appendix): 
ggplot(data = glm_subset_validity, aes(x = reorder_within(id, instruction_validity, model, median), y = instruction_validity)) +
    # get boxplot
    geom_boxplot(outlier.size=0.5, color=vircol[1]) +
    facet_wrap(~model, ncol=2, scales="free_x") +
    # dashed line at 0
    # geom_hline(yintercept = 0.5, linetype = "dashed") +
    ylab("Probability of agreement to statement") +
    xlab("Statement") +
    scale_x_reordered() + 
    scale_colour_manual(values = vircol) + 
    # scale_x_discrete(guide = guide_axis(angle = 45)) +
    theme(legend.position = "bottom", text = element_text(family = "sans"),
          axis.text.x = element_text(size = 7)) +
    #scale_colour_viridis(discrete = T) +
    theme(text = element_text(family = "sans")) +
    # increase font size for x-axis, y-axis lgend and facet-wrap
    theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), strip.text = element_text(size = 12))
 ggsave("plots/validity_micro.pdf", width = 17, height = 17, dpi=200, units="in")
 

### Experiment 2: Assess p(agree)
subset <- c("Llama-2-7B-Chat", "Falcon-7B-Instruct", "GPT-3.5", "GPT-4o", "RoBERTa-Large", "BERT-Large")


plot_data_subset <- alternatives_all_preproc %>%
  filter(model %in% subset)

## Figure 2:
ggplot(data = plot_data_subset, aes(x = reorder_within(id, agree_renorm, model, median), y = agree_renorm, color=lmtype)) +
    geom_boxplot(outlier.size=0.5) +
    facet_wrap(~model, ncol=2, scales="free_x") +
    # dashed line at 0
    geom_hline(yintercept = 0.5, linetype = "dashed") +
    ylab("Probability of agreement to statement") +
    xlab("Statement") +
    scale_x_reordered() + 
    scale_colour_manual(values = vircol) + 
    # scale_x_discrete(guide = guide_axis(angle = 45)) +
    theme(legend.position = "bottom", text = element_text(family = "sans")) +
    #scale_colour_viridis(discrete = T) +
    theme(text = element_text(family = "sans")) +
    # increase font size for x-axis, y-axis lgend and facet-wrap
    theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), strip.text = element_text(size = 12))
ggsave("plots/boxplots_agree_micro_subset.pdf", width = 18, height = 15, dpi=150, units="in")

# get all models that are NOT in subset
plot_data_rest <- alternatives_all_preproc %>%
  filter(!(model %in% subset))

## Figure 5:
ggplot(data = plot_data_rest, aes(x = reorder_within(id, agree_renorm, model, median), y = agree_renorm, color=lmtype)) +
    # get boxplot
    geom_boxplot(outlier.size=0.5) +
    facet_wrap(~model, ncol=2, scales="free_x") +
    # dashed line at 0
    geom_hline(yintercept = 0.5, linetype = "dashed") +
    ylab("Probability of agreement to statement") +
    xlab("Statement") +
    scale_x_reordered() + 
    scale_colour_manual(values = vircol) + 
    # scale_x_discrete(guide = guide_axis(angle = 45)) +
    theme(legend.position = "bottom", text = element_text(family = "sans")) +
    #scale_colour_viridis(discrete = T) +
    theme(text = element_text(family = "sans")) +
    theme(axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), strip.text = element_text(size = 12))
ggsave("plots/boxplots_agree_micro_all.pdf", width = 17, height = 20, dpi=200, units="in")


summary_comp_p_agree <- alternatives_all_preproc %>%
  group_by(model, id) %>%
  dplyr::summarise(avg_pag = mean(agree_renorm),
            min_max = max(agree_renorm) - min(agree_renorm),
            # sum of argree_renorm_binary divided by n
            n_agree_binary = sum(agree_renorm_binary) / n(),
            n_disagree_binary = 1-(sum(agree_renorm_binary) / n()),
            inconsistent_with_majority = min(n_agree_binary, n_disagree_binary),
            # ci_width = CI(agree_renorm)[1] - CI(agree_renorm)[3],
            sd = sd(agree_renorm) )


summary_comp_p_agree_avg <- summary_comp_p_agree %>%
  group_by(model) %>%
  dplyr::summarise(
    f_005 = sum(inconsistent_with_majority > 0.05)/n(),
    f_01 = sum(inconsistent_with_majority > 0.1)/n(),
    f_025 = sum(inconsistent_with_majority > 0.25)/n(),
    min_max_mean = mean(min_max),
    min_max_sd = sd(min_max),
    sd_mean = mean(sd),
    sd_sd = sd(sd),
)

summary_comp_p_agree_avg <- summary_comp_p_agree_avg %>%
  left_join(summary_df, by = "model")


summary_comp_p_agree_avg$validity <-  paste(round(summary_df$mean, 2), "\\db{", round(summary_df$sd, 2), "}", sep = "")
summary_comp_p_agree_avg$min_max <- paste(round(summary_comp_p_agree_avg$min_max_mean, 2), "\\db{", round(summary_comp_p_agree_avg$min_max_sd, 2), "}", sep = "")
summary_comp_p_agree_avg$f_005 <- paste(round(summary_comp_p_agree_avg$f_005, 2))
summary_comp_p_agree_avg$f_01 <- paste(round(summary_comp_p_agree_avg$f_01, 2))
summary_comp_p_agree_avg$f_025 <- paste(round(summary_comp_p_agree_avg$f_025, 2))
summary_comp_p_agree_avg$sd <- paste(round(summary_comp_p_agree_avg$sd_mean, 2), "\\db{", round(summary_comp_p_agree_avg$sd_sd, 2), "}", sep = "")
summary_comp_p_agree_avg$model <- as.character(summary_comp_p_agree_avg$model)

## Table 1: Response validity and consistency
sink("tables/validity_and_consistency_new.txt")
cat("Model & Validity & Min-Max & SD & $F_{5}$ & $F_{10}$ & $F_{25}$ \\\\\n")
for (i in 1:nrow(summary_comp_p_agree_avg)) {
  cat(summary_comp_p_agree_avg$model[i], " & ", "$",summary_comp_p_agree_avg$validity[i], "$ & ", "$",summary_comp_p_agree_avg$min_max[i], "$ & ", "$",summary_comp_p_agree_avg$sd[i], "$ & ", "$", summary_comp_p_agree_avg$f_005[i], "$ & ", "$", summary_comp_p_agree_avg$f_01[i], "$ & ", "$" ,summary_comp_p_agree_avg$f_025[i], "$ \\\\\n")
  # cat(summary_df$model[i], " & ", "$",summary_df$validity[i], "$ \\\\\n")
}
sink()

summary_dag_micro <- alternatives_all_preproc %>%
  group_by(model, id) %>%
  summarise_at(vars(delta_agree, agree_renorm, agree_alternative), list(mean = mean, sd = sd, se = std.error))

# attach lmtype
summary_dag_micro <- summary_dag_micro %>%
  mutate(lmtype = ifelse(model %in% comparison_glm$model, "glm", "mlm"))


# for each model and ID, find statement with highest and lowest agree_alternative, and write this row to a new df
max_alternatives_all_preproc <- alternatives_all_preproc %>%
  dplyr::select(model, id, agree_alternative, alternative_statement) %>%
  dplyr::group_by(model, id) %>%
  dplyr::filter(agree_alternative == max(agree_alternative, na.rm=TRUE))

min_alternatives_all_preproc <- alternatives_all_preproc %>%
  dplyr::select(model, id, agree_alternative, alternative_statement) %>%
  dplyr::group_by(model, id) %>%
  dplyr::filter(agree_alternative == min(agree_alternative, na.rm=TRUE))

diff <- max_alternatives_all_preproc %>%
  inner_join(min_alternatives_all_preproc, by = c("model", "id"), suffix = c("_max", "_min"))

diff$difference <- diff$agree_alternative_max - diff$agree_alternative_min
# sort by difference
diff <- diff[order(diff$difference, decreasing = TRUE),]
# kick out gpt-3.5 and gpt-4
diff <- diff %>%
  filter(model != "GPT-3.5" & model != "GPT-4")

## Table 2: Examples with highest min-max difference
write_csv(diff, "tables/min_max_statements.csv")


### Experiment 3: Assess predictors of p(agree)

par <- c("average_lex_freq_alternative", "average_word_length_alternative", "sentence_length_alternative", "sentiment_pos_alternative", "sentiment_neg_alternative")

model_results <- list()
for (model_name in unique(levels(alternatives_all_preproc$model))) {
  print(model_name)
  model_df <- alternatives_all_preproc %>%
    filter(model == model_name)
  model_lm <- call_model(model_df)
  model_results[[model_name]] <- rsample::tidy(model_lm)
}

## Table 3: Effect sizes of predictors
sink("tables/effect_sizes_all.txt")
for (i in 1:length(model_results)) {
  model_name <- names(model_results)[i]
  result <- model_results[[i]]
  cat(paste(
    model_name, " & $", 
    round(result[result$term == par[1],"estimate"], 3), " \\db{", round(result[result$term == par[1],"std.error"], 3), "}", sig(result[result$term == par[1],"p.value"]), "$ & $",
    round(result[result$term == par[2],"estimate"], 3), " \\db{", round(result[result$term == par[2],"std.error"], 3), "}", sig(result[result$term == par[2],"p.value"]), "$ & $",
    round(result[result$term == par[3],"estimate"], 3), " \\db{", round(result[result$term == par[3],"std.error"], 3), "}", sig(result[result$term == par[3],"p.value"]), "$ & $",
    round(result[result$term == par[4],"estimate"], 3), " \\db{", round(result[result$term == par[4],"std.error"], 3), "}", sig(result[result$term == par[4],"p.value"]), "$ & $",
    round(result[result$term == par[5],"estimate"], 3), " \\db{", round(result[result$term == par[5],"std.error"], 3), "}", sig(result[result$term == par[5],"p.value"]), "$ \\\\"
    ), sep="\n")
}
sink()

### Experiment 4: Assess average variance of p(agree) of each statement across models
summary_var_micro <- alternatives_all_preproc %>%
  group_by(id, model) %>%
  summarise_at(vars(agree_renorm), list(sd = sd))

glm_names <- c("Llama-2-13B-Chat", "GPT-3.5", "GPT-4", "Llama-2-7B-Chat", "Phi2", "Falcon-7B", "Llama-2-7B", "Tiny-Llama", "GPT-4o", "Falcon-7B-Instruct", "Llama-2-13B")

# rassign lmtype
summary_var_micro <- summary_var_micro %>%
  mutate(lmtype = ifelse(model %in% glm_names, "glm", "mlm"))

# normalize variance into [0, 1] across models
summary_var_micro_norm <- summary_var_micro %>%
  dplyr::group_by(model) %>%
  dplyr::mutate(var_norm = sd/max(sd))

# compute mean and sd for each statement over all models
summary_var_micro_norm_mean_avg <- summary_var_micro_norm %>%
  dplyr::group_by(id) %>%
  dplyr::summarise_at(vars(var_norm), list(mean = mean, se = std.error))

summary_var_micro_norm_mean <- summary_var_micro_norm %>%
  dplyr::group_by(id, lmtype) %>%
  dplyr::summarise_at(vars(var_norm), list(mean = mean, se = std.error))

# split df by lmtype
summary_var_micro_norm_mean_glm <- summary_var_micro_norm_mean %>%
  filter(lmtype == "glm")
summary_var_micro_norm_mean_mlm <- summary_var_micro_norm_mean %>%
  filter(lmtype == "mlm")

# first, rearrange ids as factor, by mean of lmtype glm
new_order <- summary_var_micro_norm_mean_avg$id[order(summary_var_micro_norm_mean_avg$mean)]

# assign order to each df
summary_var_micro_norm_mean_glm$id_sort <- factor(summary_var_micro_norm_mean_glm$id, levels = new_order)
summary_var_micro_norm_mean_mlm$id_sort <- factor(summary_var_micro_norm_mean_mlm$id, levels = new_order)
summary_var_micro_norm_mean_avg$id_sort <- factor(summary_var_micro_norm_mean_avg$id, levels = new_order)
summary_var_micro_norm_mean_avg$lmtype <- "average"

# put back together
summary_var_micro_norm_mean <- rbind(summary_var_micro_norm_mean_glm, summary_var_micro_norm_mean_mlm)

## Figure 3: SDs of p(agree) across models
ggplot(data = summary_var_micro_norm_mean, aes(x = id_sort, y = mean, color=lmtype, shape=lmtype)) +
    geom_point(position = position_dodge(width = .5), size = 2) +
    geom_errorbar(aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se),
            width = .1, position = position_dodge(width = .5), linewidth = 0.4) +
    ylab("Normalized standard deviations of agreement to statement (averaged over models)") +
    xlab("Statement") +
    scale_x_discrete(guide = guide_axis(angle = 75)) +
    scale_colour_manual(values = vircol) +
    theme(text = element_text(family = "sans"))

summary_var_micro_norm_mean_all <- rbind(summary_var_micro_norm_mean_glm, summary_var_micro_norm_mean_mlm, summary_var_micro_norm_mean_avg)

ggplot(data = summary_var_micro_norm_mean_all, aes(x = id_sort, y = mean, color=lmtype, shape=lmtype)) +
    geom_point(position = position_dodge(width = .5), size = 2) +
    geom_errorbar(aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se),
            width = .1, position = position_dodge(width = .5), linewidth = 0.4) +
    ylab("Normalized standard deviations of agreement to statement (averaged over models)") +
    xlab("Statement") +
    scale_x_discrete(guide = guide_axis(angle = 75)) +
    scale_colour_manual(values = c("darkgray", vircol)) +
    theme(text = element_text(family = "sans")) + 
    # increase font size for x-axis, y-axis and legend
    theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), legend.text = element_text(size = 12), legend.title = element_text(size = 12))

ggsave("plots/sd_across_statements.pdf", width = 16, height = 8, dpi=200, units="in")


## Figure 6 (Appendix): Correlation Matrix
cor.mtest <- function(mat, ...) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat<- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
        for (j in (i + 1):n) {
            tmp <- cor.test(mat[, i], mat[, j], ...)
            p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
        }
    }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

M <- cor(alternatives_all_preproc[, par])
p.mat <- cor.mtest(alternatives_all_preproc[, par])

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))


pdf(file = "plots/correlation_matrix.pdf", width = 10, height = 10)
corrplot(M, method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=20, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)
dev.off()