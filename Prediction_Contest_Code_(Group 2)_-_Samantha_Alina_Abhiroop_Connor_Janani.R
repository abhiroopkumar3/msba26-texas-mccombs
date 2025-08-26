# ASSIGNMENT: Prediction Contest

# GROUP: Group 2
# AUTHORS: Samantha Feinberg, Alina Hota, Abhiroop Kumar, Connor Therrien, Janani Vakkanti
# COURSE: Intro to Machine Learning (STA 380)
# SEMESTER: Summer 2025
# SUBMISSIN DUE: Aug 8 at 11:59pm
# PROFESSOR: Jared Murray

# LAST UPDATED BY: Abhiroop Kumar
# LAST UPDATED DATE: 2025-08-08

# SOLUTION:




# ==========================================
# LOAD REQUIRED LIBRARIES
# ==========================================

library(tidyverse)
library(tidymodels)
library(glmnet)
library(pls)
library(corrplot)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)




# ==========================================
# DATA IMPORT
# ==========================================

# Load 'austinhouses.csv' file
# austin_houses <- read.csv("austinhouses.csv", stringsAsFactors = TRUE) # ensure .R file is in same folder/directory as 'austinhouses.csv'
austin_houses <- read.csv("C:/ABHIROOP_CHUWI_DATA/STUDIES/ML_workDir/Intro to Machine Learning - I/Assignments/Prediction Contest/austinhouses.csv", stringsAsFactors = TRUE)
# colnames(austin_houses)




# ==========================================
# X/Y GRAPHS
# ==========================================

numeric_data <- austin_houses[, -c(6,7,8,9)]
numeric_data <- numeric_data %>%
  select_if(is.numeric)
corrplot(cor(numeric_data), type = "lower", diag = FALSE, method = "color", main = "Correlation Matrix of Numeric Variables")

# Exclude the target variable from the list of variables to plot against
vars_to_plot <- setdiff(names(austin_houses), "latestPrice")

# Loop through each variable and plot
for (var in vars_to_plot) {
  plot(austin_houses[[var]], austin_houses$latestPrice,
       main = paste("latestPrice vs", var),
       xlab = var,
       ylab = "latestPrice")
}




# ==========================================
# DATA CLEANING
# ==========================================
# Remove unrelated columns
cols_to_remove <- c("streetAddress", "description", "hasGarage", "hasSpa", "hasView", "homeType", "latest_saledate", "latest_salemonth", "numOfPhotos", "numOfAccessibilityFeatures", "numOfAppliances", "numOfWindowFeatures", "avgSchoolSize")
austin_houses_less_cols <- austin_houses %>%
  select(-all_of(cols_to_remove))
# colnames(austin_houses_less_cols)
# dim(austin_houses_less_cols)
# nrow(austin_houses_less_cols)

# Drop all rows with empty/NA values
austin_houses_clean <- austin_houses_less_cols %>%
  mutate(across(where(is.character), \(x) na_if(x, ""))) %>% # declare 'x'="", then replace value of x with 'NA' in all columns with value of x (empty value)
  drop_na() # remove rows with NA values
# nrow(austin_houses_clean)




# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Add new features
austin_houses_final <- austin_houses_clean %>%
  mutate(
    # Age of house when it was last sold
    age_when_sold = latest_saleyear - yearBuilt,
    # Ratio of living area to lot size
    area_ratio = livingAreaSqFt / lotSizeSqFt,
    # Total number of rooms
    total_rooms = numOfBedrooms + numOfBathrooms,
    # Quality of school by score (higher is better)
    school_quality_score = avgSchoolRating * avgSchoolDistance,
    # New log-transformed target variable
    log_latestPrice = log(latestPrice)
  ) %>%
  select(-latestPrice) # remove original target variable




# ==========================================
# UNIFIED TRAIN/TEST SPLIT
# ==========================================

set.seed(1)
austinhouses_split <- initial_split(austin_houses_final, prop = 0.8, strata = log_latestPrice) # split dataset
austinhouses_train <- training(austinhouses_split) # 80% training data
austinhouses_test <- testing(austinhouses_split) # 20% testing data

# Predictor matrix (x) & Response vector (y) for glmnet & pls
x_training <- model.matrix(log_latestPrice ~ ., austinhouses_train)[, -1]
y_training <- austinhouses_train$log_latestPrice
x_testing <- model.matrix(log_latestPrice ~ ., austinhouses_test)[, -1]
y_testing <- austinhouses_test$log_latestPrice




# ==========================================
# LINEAR REGRESSION
# ==========================================

# Use the unified train_data and test_data from the UNIFIED TRAIN/TEST SPLIT section
set.seed(1)
train_data <- austinhouses_train
test_data <- austinhouses_test

# Specify categorical, logical, and numeric columns
cat_cols <- c("zipcode")
log_cols <- c("hasAssociation")
num_cols <- setdiff(names(train_data %>% select(-log_latestPrice)), c(cat_cols, log_cols))

# Build the recipe
recipe_spec <- recipe(log_latestPrice ~ ., data = train_data) %>%
  step_mutate(
    zipcode = factor(zipcode)
  ) %>%
  step_impute_median(all_numeric(), -all_outcomes()) %>%
  step_impute_mode(all_nominal(), -all_outcomes()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_normalize(all_of(num_cols)) %>%
  step_dummy(all_of(cat_cols), one_hot = TRUE)


# Define the model
linreg_model <- linear_reg() %>% 
  set_engine("lm")

# Combine into a workflow
linreg_workflow <- workflow() %>%
  add_model(linreg_model) %>%
  add_recipe(recipe_spec)

# Train the model
linreg_fit <- fit(linreg_workflow, data = train_data)

# Predict on test data (log scale)
preds_log <- predict(linreg_fit, test_data)

# Calculate RMSE on log scale
rmse_val_log <- sqrt(mean((y_testing - preds_log$.pred)^2))

# Output results
print(paste("Linear Regression RMSE (on log scale):", round(rmse_val_log, 4)))
print(paste("Max price:", exp(max(train_data$log_latestPrice))))
print(paste("Min price:", exp(min(train_data$log_latestPrice))))

cat("\n\n")




# ==========================================
# RIDGE REGRESSION
# ==========================================

# Cross-validation to find optimal lambda
set.seed(1)
cv_ridge <- cv.glmnet(x_training, y_training, alpha = 0, nfolds = 10) # 'alpha = 0' is Ridge Regression
# plot(cv_ridge, main = "Ridge Regression: Cross-Validation Error vs. Lambda")
optimal_lambda_ridge <- cv_ridge$lambda.min # optimal lambda value (reduces cross-validation error)
cat(paste("Optimal lambda for Ridge Regression:", round(optimal_lambda_ridge, 4), "\n"))

# Fit Ridge model
ridge_model <- glmnet(x_training, y_training, alpha = 0, lambda = optimal_lambda_ridge)
ridge_predictions_log <- predict(ridge_model, newx = x_testing) # test set prediction
ridge_rmse_log <- sqrt(mean((ridge_predictions_log - y_testing)^2)) # mean squared error
cat(paste("Test RMSE for Ridge Regression (on log scale):", round(ridge_rmse_log, 4), "\n"))
# cat(paste("Ridge Regression Coefficients (for optimal lambda):", coef(ridge_model), "\n")) # ridge coefficients

cat(paste("<< ABHIROOP PARAGRAPH >>"))
cat("\n\n")




# ==========================================
# LASSO REGRESSION
# ==========================================

set.seed(1)
grid <- 10^seq(10, -2, length = 100)

lasso.mod <- glmnet(x_training, y_training, alpha = 1, lambda = grid)
# plot(lasso.mod, main = "Lasso Regression: Coefficients vs. Log Lambda")

cv.lasso.out <- cv.glmnet(x_training, y_training, alpha = 1)
# plot(cv.lasso.out, main = "Lasso Regression: Cross-Validation Error vs. Lambda")
bestlassolam <- cv.lasso.out$lambda.min
cat(paste("Optimal lambda for Lasso Regression:", round(bestlassolam, 4), "\n"))

lasso.pred_log <- predict(lasso.mod, s = bestlassolam, newx = x_testing)
lasso_rmse_log <- sqrt(mean((lasso.pred_log - y_testing)^2))
cat(paste("Test RMSE for Lasso Regression (on log scale):", round(lasso_rmse_log, 4), "\n"))

# Fit lasso on the full dataset with optimal lambda to get coefficients
lasso.out <- glmnet(rbind(x_training, x_testing), c(y_training, y_testing), alpha = 1)
lasso.coef <- predict(lasso.out, type = "coefficients", s = bestlassolam)
# cat("Lasso Regression Coefficients (for optimal lambda):\n")
# print(lasso.coef)

cat("\n\n")




# ==========================================
# STEPWISE
# ==========================================

# Use the pre-defined austinhouses_train and austinhouses_test
set.seed(1)
houses_train <- austinhouses_train
houses_test <- austinhouses_test

# Define test target (log scale)
y_test_log <- houses_test$log_latestPrice

# Model 1: Start from NULL (intercept only), add/delete single main effects
step_null <- stats::step(lm(log_latestPrice ~ 1, data = houses_train),
                         scope = ~ .,
                         direction = "both"
)

# Model 2: Start from full main effects model, add/delete single main effects
step_main <- stats::step(lm(log_latestPrice ~ ., data = houses_train),
                         scope = ~ .,
                         direction = "both",
)

# Model 3: Start from main effects only, consider interactions in scope
# Select interaction based on location, size, and amenties
main_model <- lm(log_latestPrice ~ ., data = houses_train) # Define main_model for step_custom

step_custom <- stats::step(
  main_model,
  scope = list(
    lower = ~1,
    upper = log_latestPrice ~ . + 
      lotSizeSqFt:livingAreaSqFt
  ),
  direction = "both",
  trace = 1 # or trace = TRUE for more output
)

# Predict on test data (log scale)
y_pred_null_log <- predict(step_null, newdata = houses_test)
y_pred_main_log <- predict(step_main, newdata = houses_test)
y_pred_custom_log <- predict(step_custom, newdata = houses_test)

# Calculate RMSE on log scale
rmse_null_log <- sqrt(mean((y_test_log - y_pred_null_log)^2))
rmse_main_log <- sqrt(mean((y_test_log - y_pred_main_log)^2))
rmse_custom_log <- sqrt(mean((y_test_log - y_pred_custom_log)^2))

# Print RMSEs
cat(paste("RMSE Stepwise from NULL (on log scale):", round(rmse_null_log, 4), "\n"))
cat(paste("RMSE Stepwise from Full (main effects) (on log scale):", round(rmse_main_log, 4), "\n"))
cat(paste("RMSE Stepwise from Full (main effects) + interactions in scope (on log scale):", round(rmse_custom_log, 4), "\n"))

cat("\n\n")




# ==========================================
# BINARY TREES
# ==========================================

# Define test target
set.seed(1)
y_test_log <- austinhouses_test$log_latestPrice

# Fit a large tree to predict log_latestPrice using all other variables
tree_full <- rpart(log_latestPrice ~ ., data = austin_houses_final,
                   control = rpart.control(cp = 0.0001, minsplit = 10))

# Print number of internal (non-leaf) nodes
cat("Initial tree size (non-leaf nodes):", 
    nrow(tree_full$frame[tree_full$frame$var != "<leaf>", ]), "\n")

# Plot the full tree
# rpart.plot(tree_full, type = 3, extra = 1, fallen.leaves = TRUE,
#    main = "Regression Tree for latestPrice (Unpruned)")

# Predict on test set (log scale)
tree_preds_log <- predict(tree_full, newdata = austinhouses_test)

# Calculate RMSE on log scale
rmse_tree_log <- sqrt(mean((y_test_log - tree_preds_log)^2))

cat("RMSE (Full Tree, unpruned, on log scale):", rmse_tree_log, "\n")

# Find optimal CP using minimum cross-validation error
min_xerror_row <- which.min(tree_full$cptable[, "xerror"])
optimal_cp_min <- tree_full$cptable[min_xerror_row, "CP"]

#Prune the full tree
tree_pruned <- rpart::prune(tree_full, cp = optimal_cp_min)

#Plot the pruned tree
# rpart.plot(tree_pruned, main = "Min CV Error", type = 3, extra = 1, fallen.leaves = TRUE)

# Print number of internal (non-leaf) nodes
cat("Initial tree size (non-leaf nodes):", 
    nrow(tree_pruned$frame[tree_pruned$frame$var != "<leaf>", ]), "\n")

#Predict on test set (log scale)
tree_pruned_preds_log <- predict(tree_pruned, newdata = austinhouses_test)

# Calculate RMSE on log scale
rmse_pruned_tree_log <- sqrt(mean((y_test_log - tree_pruned_preds_log)^2))

cat("RMSE (Pruned Tree, on log scale):", rmse_pruned_tree_log, "\n")

cat("\n\n")



# ==========================================
# BAGGING APPROACH
# ==========================================

# Response variable = log(latestPrice)
set.seed(1)
bagging_train <- austinhouses_train
bagging_test <- austinhouses_test

bagging_predictors <- setdiff(names(bagging_train), "log_latestPrice")
formula_str <- paste("log_latestPrice ~", paste(bagging_predictors, collapse = " + "))
formula_obj <- as.formula(formula_str)

# ntree = 500 >>> enough to stabilize error
set.seed(1)
bagging_fit <- randomForest(formula_obj, data = bagging_train, mtry = length(bagging_predictors), ntree = 500, importance = TRUE)
summary(bagging_fit)

# Test set prediction (log scale)
bagging_pred_log <- predict(bagging_fit, newdata = bagging_test)

# Test RMSE on log scale
actual_log_bagging_test <- bagging_test$log_latestPrice
bagging_test_rmse_log = sqrt(mean((actual_log_bagging_test - bagging_pred_log)^2))
cat(paste("RMSE (Bagging Approach, on log scale):", round(bagging_test_rmse_log, 4), "\n"))

cat("\n\n")




# ==========================================
# RANDOM FOREST
# ==========================================

set.seed(1)
rf_train_data <- austinhouses_train
rf_test_data <- austinhouses_test

rand_forest(mode = "regression") %>%
  set_engine("randomForest") # <- uses the classic package


# 1. Define the recipe
rf_recipe <- recipe(log_latestPrice ~ ., data = rf_train_data) %>%
  step_mutate(
    zipcode = as.factor(zipcode)
  ) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())

# 2. Specify the Random Forest model
rf_model <- rand_forest(mtry = 3, trees = 100, min_n = 5) %>%
  set_engine("ranger", importance = "impurity", seed = 1) %>%
  set_mode("regression")

# 3. Create the workflow
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

# 4. Fit the model
rf_fit <- fit(rf_workflow, data = rf_train_data)

# 5. Predict on test data (log scale)
rf_preds_log <- predict(rf_fit, new_data = rf_test_data)

# 6. Evaluate RMSE on log scale
rmse_rf_log <- sqrt(mean((y_testing - rf_preds_log$.pred)^2))
cat(paste("Random Forest RMSE (on log scale):", round(rmse_rf_log, 4)))

cat("\n\n")




# ==========================================
# BOOSTING MODEL
# ==========================================

# Convert logical columns to factors
austinhouses_train_gbm <- austinhouses_train %>%
  mutate(across(where(is.logical), as.factor))

set.seed(1)
austin_boosted <- gbm(log_latestPrice ~. , data = austinhouses_train_gbm,
                      distribution = "gaussian", n.trees = 5000,
                      shrinkage = 0.01, interaction.depth = 4)
summary(austin_boosted)

y_test_log <- austinhouses_test$log_latestPrice
yhat.boost_log <- predict(austin_boosted, newdata = austinhouses_test, n.trees = 5000)
rmse_boosting_log <- sqrt(mean((y_test_log - yhat.boost_log)^2))

cat("RMSE Boosting (on log scale):", rmse_boosting_log, "\n")

cat("\n\n")




# ==========================================
# TEST RMSE FOR ALL MODELS
# ==========================================

cat(paste("TEST RMSE FOR ALL MODELS (on log scale):\n1. Test RMSE for Linear Regression:", round(rmse_val_log, 4), "\n2. Test RMSE for Ridge Regression:", round(ridge_rmse_log, 4), "\n3. Test RMSE for Lasso Regression:", round(lasso_rmse_log, 4), "\n4. Test RMSE for Stepwise:\n\ta. Test RMSE for Stepwise from NULL:", round(rmse_null_log, 4), "\n\tb. Test RMSE for Stepwise from Full (main effects):", round(rmse_main_log, 4), "\n\tc. Test RMSE for Stepwise from Full (main effects) + interactions in scope:", round(rmse_custom_log, 4), "\n5. Test RMSE for Full Tree (Unpruned):", round(rmse_tree_log, 4), "\n6. Test RMSE for Pruned Tree:", round(rmse_pruned_tree_log, 4), "\n7. Test RMSE for Bagging Approach:", round(bagging_test_rmse_log, 4), "\n8. Test RMSE for Random Forest:", round(rmse_rf_log, 4), "\n9. Test RMSE for Boosting Model:", round(rmse_boosting_log, 4), "\n<< Model with Lowest RMSE: 'Binary Trees (Unpruned)'>>\n\n"))



# ==========================================
# PREDICTING NEW DATA
# ==========================================

# Load 'austinhouses_holdout.csv' file
# austin_houses_holdout <- read.csv("austinhouses_holdout.csv", stringsAsFactors = TRUE)
austin_houses_holdout <- read.csv("C:/ABHIROOP_CHUWI_DATA/STUDIES/ML_workDir/Intro to Machine Learning - I/Assignments/Prediction Contest/austinhouses_holdout.csv", stringsAsFactors = TRUE)

austin_houses_holdout_less_cols <- austin_houses_holdout %>%
  select(-all_of(cols_to_remove))

# Apply same data cleaning and feature engineering as original dataset
holdout_final <- austin_houses_holdout_less_cols %>%
  mutate(
    age_when_sold = latest_saleyear - yearBuilt,
    area_ratio = livingAreaSqFt / lotSizeSqFt,
    total_rooms = numOfBedrooms + numOfBathrooms,
    school_quality_score = avgSchoolRating * avgSchoolDistance
  ) %>%
  mutate(area_ratio = replace(area_ratio, is.infinite(area_ratio), 0))

# Apply same Regression Tree (Unpruned) Model as original data set
set.seed(1)
holdout_predicted_prices_log <- predict(tree_full, newdata = holdout_final)
holdout_predicted_prices <- exp(holdout_predicted_prices_log)

# Predicted prices
cat("Predicted 'latestPrice' summary & values for austinhouses_hold.csv:\n")
summary(holdout_predicted_prices)
print(head(holdout_predicted_prices, 100))

cat("\n\n")