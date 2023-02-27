### XGboost

library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(vtreat)
library(FeatureSelection)
library(SHAPforxgboost)
library(here)
library(data.table)



KS_1415 <- read.csv("C:/Users/Yang Hsiu/Desktop/Dengue/dengue project 10-7/dengue project 10-7/KS_1415.csv",sep = ",")
KS_0613 <- read.csv("C:/Users/Yang Hsiu/Desktop/Dengue/dengue project 10-7/dengue project 10-7/KS_0613.csv",sep = ",")
KS_sum <- read.csv("C:/Users/Yang Hsiu/Desktop/Dengue/dengue project 10-7/dengue project 10-7/KS_sum.csv",sep = ",")


b <- which(KS_sum$Number.of.households <= 10)

KS <- KS_sum[-b,]

KS <- KS[,-c(1:4,104)]


KS$IR <- log(KS$IR*100000+1)

smp_size <- floor(0.8 * nrow(KS))

set.seed(1)
train_ind <- sample(seq_len(nrow(KS)), size = smp_size)

KS_train <- KS[train_ind, ]
KS_test <- KS[-train_ind, ]

p <- KS_train[,"IR"]

params_glmnet = list(alpha = 1, family = 'gaussian', nfolds = 5, parallel = TRUE)

params_xgboost <- list( params = list("objective" = "reg:linear", "bst:eta" = 0.001, "subsample" = 0.75, "max_depth" = 5,
                                     "colsample_bytree" = 0.75, "nthread" = 6),
                       nrounds = 1000, print.every.n = 250, maximize = FALSE)

params_ranger = list(dependent.variable.name = 'y', probability = FALSE, num.trees = 1000, verbose = TRUE, mtry = 5, 
 min.node.size = 10, num.threads = 6, classification = FALSE, importance = 'permutation')


params_features <- list(keep_number_feat = NULL, union = TRUE)


feat <- wrapper_feat_select(X = KS_train[,-1], y = p, params_glmnet = params_glmnet, params_xgboost = params_xgboost, 
                           params_ranger = params_ranger, xgb_sort = 'Gain', CV_folds = 5, stratified_regr = FALSE, 
                           scale_coefs_glmnet = FALSE, cores_glmnet = 5, params_features = params_features, verbose = TRUE)


params_barplot = list(keep_features = 20, horiz = TRUE, cex.names = 1.0)

barplot_feat_select(feat, params_barplot, xgb_sort = 'Cover')

dataXY <- KS_train[,c("IR","SI","F050203","F050201","F030303","F090801","F050204","F050101","HD","F030304","F050102","F050302","F050401","F050403",
                     "F070201","F050301","F050404","F050202","F010102","F060201","F060400")]

y_var <-  "IR"
dataX <- dataXY[,-1]
dataX <- as.matrix(dataX)


param_list <- list(objective = "reg:squarederror",  # For regression
                   eta = 0.02,
                   max_depth = 10,
                   gamma = 0.01,
                   subsample = 0.95
)
mod <- xgboost::xgboost(data = dataX, 
                        label = as.matrix(dataXY[[y_var]]), 
                        params = param_list, nrounds = 10,
                        verbose = FALSE, nthread = parallel::detectCores() - 2,
                        early_stopping_rounds = 8)


# To return the SHAP values and ranked features by mean|SHAP|
shap_values <- shap.values(xgb_model = mod, X_train = dataX)
# The ranked features by mean |SHAP|
shap_values$mean_shap_score



# to show that `rowSum` is the output:
shap_data <- copy(shap_values$shap_score)
shap_data[, BIAS := shap_values$BIAS0]
pred_mod <- predict(mod, dataX, ntreelimit = 10)
shap_data[, `:=`(rowSum = round(rowSums(shap_data),6), pred_mod = round(pred_mod,6))]
rmarkdown::paged_table(shap_data[1:20,])

shap_long <- shap.prep(xgb_model = mod, X_train = dataX)
# is the same as: using given shap_contrib
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = dataX)
# **SHAP summary plot**
shap.plot.summary(shap_long)


g1 <- shap.plot.dependence(data_long = shap_long, x = 'F090801', y = 'F090801',color_feature = "F090801") + ggtitle("(A) SHAP values of Time trend vs. Time trend")
g2 <- shap.plot.dependence(data_long = shap_long, x = 'SI', y = 'SI', color_feature = 'SI') +  ggtitle("(B) SHAP values of CWV vs. Time trend")

gridExtra::grid.arrange(g1, g2, ncol = 2)


plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 4, n_groups = 6)
# you may choose to zoom in at a location, and set y-axis limit using `y_parent_limit`  
shap.plot.force_plot(plot_data, zoom_in_location = 400, y_parent_limit = c(-0.1,0.1))