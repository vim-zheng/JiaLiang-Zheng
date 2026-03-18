
```R


# ================================================================================
# [Machine Learning Predictive Model: NOAF after CABG]
# ================================================================================


# ==================== Part 0: Environment Setup ====================
setwd("D:/Rproject/ML_and_Data_Analysis/NOAF_after_CABG/machine_learning_code")

packages <- c(
  "readxl", "dplyr", "caret", "pROC",
  "openxlsx", "glmnet", "RColorBrewer",
  "Boruta", "mice", "writexl"
)

new.packages <- setdiff(packages, rownames(installed.packages()))
if (length(new.packages)) {
  install.packages(new.packages,
                   repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
}

library(readxl);  library(dplyr);    library(caret)
library(pROC);    library(openxlsx); library(glmnet)
library(Boruta);  library(mice);     library(writexl)

cat("All packages loaded.\n\n")


# ==================== Part 1: Data Loading and Preprocessing ====================
cat("========== Step 1: Data Loading and Preprocessing ==========\n")

dat_all        <- read_excel("D:/Rproject/ML_and_Data_Analysis/raw_data.xlsx")
patient_id_col <- "Patient_ID"
response_var   <- "NOAF_outcome"

cat("Raw data dimensions:", nrow(dat_all), "x", ncol(dat_all), "\n")
cat("Outcome distribution:\n"); print(table(dat_all[[response_var]]))

# Outcome → factor(no, yes)
dat_all[[response_var]] <- factor(dat_all[[response_var]], levels = c("no", "yes"))

# Character columns → factor (exclude ID and outcome)
char_cols <- setdiff(names(dat_all)[sapply(dat_all, is.character)],
                     c(patient_id_col, response_var))
if (length(char_cols) > 0) {
  dat_all[char_cols] <- lapply(dat_all[char_cols], factor)
  cat("Converted", length(char_cols), "character variable(s) to factor.\n")
}

missing_before <- sapply(dat_all, function(x) sum(is.na(x)))
missing_before <- missing_before[missing_before > 0]
if (length(missing_before) > 0) {
  cat("Variables with missing values:\n"); print(missing_before)
} else {
  cat("No missing values detected.\n")
}
cat("Step 1 complete.\n\n")


# ==================== Part 2: 70/30 Train-Validation Split ====================
# [L1] Split on RAW data before any imputation.
#      Ensures validation-set values never influence MICE imputation models.
# =============================================================================
cat("========== Step 2: 70/30 Stratified Split (raw data) ==========\n")

set.seed() # set the random seeds
split_idx  <- createDataPartition(dat_all[[response_var]], p = 0.7, list = FALSE)
train_pool <- dat_all[ split_idx, ]
valid_pool <- dat_all[-split_idx, ]

cat(sprintf("Training set   : %d rows (%.1f%%)\n",
            nrow(train_pool), nrow(train_pool) / nrow(dat_all) * 100))
cat(sprintf("Validation set : %d rows (%.1f%%)\n",
            nrow(valid_pool), nrow(valid_pool) / nrow(dat_all) * 100))
cat("Training outcome:\n");   print(table(train_pool[[response_var]]))
cat("Validation outcome:\n"); print(table(valid_pool[[response_var]]))

write_xlsx(train_pool, "train_pool_raw.xlsx")
write_xlsx(valid_pool, "valid_pool_raw.xlsx")
cat("Raw split sets exported.\nStep 2 complete.\n\n")


# ==================== Part 3: MICE Multiple Imputation ====================
# [L2] Imputation performed INDEPENDENTLY on each subset.
#      Training and validation sets each generate their own MICE model.
#      No cross-contamination of distributional information.
# =========================================================================
cat("========== Step 3: MICE Multiple Imputation ==========\n")

run_mice_imputation <- function(data, id_col, outcome_col,
                                m = 5, maxit = 10, seed = 20251106) {
  cols_to_impute <- setdiff(colnames(data), c(id_col, outcome_col))
  data_for_mice  <- data[, cols_to_impute]

  cat("  Running MICE on", ncol(data_for_mice),
      "feature columns | m =", m, "| maxit =", maxit, "\n")

  mice_obj <- mice(data_for_mice, m = m, maxit = maxit,
                   seed = seed, printFlag = FALSE)

  completed_features <- complete(mice_obj, action = 1)

  imputed_data <- bind_cols(
    data[, c(id_col, outcome_col)],
    completed_features
  )
  return(list(imputed_data = imputed_data, mice_obj = mice_obj))
}

# 3.1 Training set
cat("--- 3.1 Imputing Training Set ---\n")
train_mice_result <- run_mice_imputation(train_pool, patient_id_col, response_var)
train_imputed     <- train_mice_result$imputed_data
cat("  Post-imputation missing (train):",
    sum(is.na(train_imputed[, setdiff(colnames(train_imputed),
                                       c(patient_id_col, response_var))])), "\n\n")

# 3.2 Validation set (independent — no leakage)
cat("--- 3.2 Imputing Validation Set (independent) ---\n")
valid_mice_result <- run_mice_imputation(valid_pool, patient_id_col, response_var)
valid_imputed     <- valid_mice_result$imputed_data
cat("  Post-imputation missing (valid):",
    sum(is.na(valid_imputed[, setdiff(colnames(valid_imputed),
                                       c(patient_id_col, response_var))])), "\n\n")

write_xlsx(train_imputed, "train_imputed.xlsx")
write_xlsx(valid_imputed, "valid_imputed.xlsx")
cat("Imputed datasets exported.\nStep 3 complete.\n\n")


# ==================== Part 4: Feature Selection (Boruta + LASSO) ====================
# [L3] ALL feature selection uses ONLY train_imputed.
#      valid_imputed is never seen at this stage.
# ====================================================================================
cat("========== Step 4: Feature Selection (Boruta -> LASSO) ==========\n")

candidate_vars <- setdiff(colnames(train_imputed), c(patient_id_col, response_var))
cat("Candidate features:", length(candidate_vars), "\n\n")

# --- 4.1 Boruta ---
cat("--- 4.1 Boruta (training set only) ---\n")
set.seed()#set the random seeds
boruta_res  <- Boruta(x = train_imputed[, candidate_vars],
                      y = as.factor(train_imputed[[response_var]]),
                      doTrace = 2, maxRuns = 100)
boruta_vars <- getSelectedAttributes(boruta_res, withTentative = TRUE)
cat("Boruta retained:", length(boruta_vars), "variables\n")
if (length(boruta_vars) == 0) stop("Boruta retained no variables.")

# --- 4.1.5 Sanitize variable names ---
cat("--- 4.1.5 Sanitizing variable names ---\n")
clean_boruta_vars <- make.names(boruta_vars)
changed <- boruta_vars[boruta_vars != clean_boruta_vars]
if (length(changed) > 0) {
  cat("Renamed:\n")
  for (v in changed) cat(sprintf("  '%s' -> '%s'\n", v, make.names(v)))
  old_names <- names(train_imputed)
  new_names <- ifelse(old_names %in% boruta_vars, make.names(old_names), old_names)
  names(train_imputed) <- new_names
  names(valid_imputed)  <- new_names
  boruta_vars <- clean_boruta_vars
  cat("Sanitization complete.\n\n")
} else {
  cat("No renaming needed.\n\n")
}

# --- 4.2 LASSO (10-fold CV, lambda.min, training set only) ---
cat("--- 4.2 LASSO (training set only) ---\n")
X_lasso <- model.matrix(reformulate(boruta_vars), train_imputed)[, -1]
y_lasso <- as.numeric(train_imputed[[response_var]] == "yes")

set.seed()#set the random seeds
cv_lasso   <- cv.glmnet(X_lasso, y_lasso,
                        family = "binomial", alpha = 1, nfolds = 10)
coef_lasso <- coef(cv_lasso, s = "lambda.min")
lasso_vars_raw <- rownames(coef_lasso)[
  coef_lasso != 0 & rownames(coef_lasso) != "(Intercept)"
]
cat("LASSO retained (raw dummy names):", length(lasso_vars_raw), "\n")

# Map dummy names back to original column names
lasso_vars <- unique(unlist(lapply(lasso_vars_raw, function(vn) {
  matched <- boruta_vars[sapply(boruta_vars,
                                function(ov) grepl(paste0("^", ov), vn))]
  if (length(matched) > 0) matched else vn
})))
cat("After mapping:", length(lasso_vars), "variables\n\n")

# --- 4.3 Forced clinical variables ---
force_vars <- c("CRP", "NLR")
final_vars <- unique(c(lasso_vars, force_vars))
cat("Final variable set (n =", length(final_vars), "):\n")
print(final_vars)

# --- 4.4 Construct final modeling datasets ---
use_cols   <- intersect(unique(c(patient_id_col, response_var, final_vars)),
                        colnames(train_imputed))
train_data <- train_imputed[, use_cols]
test_data  <- valid_imputed[, use_cols]

cat("\nFinal train:", nrow(train_data), "x", ncol(train_data), "\n")
cat("Final valid:", nrow(test_data),  "x", ncol(test_data),  "\n\n")

write.xlsx(
  data.frame(Stage      = c("Candidate", "Boruta", "LASSO", "Final"),
             N_features = c(length(candidate_vars), length(boruta_vars),
                            length(lasso_vars),     length(final_vars))),
  "0_Feature_Selection_Summary.xlsx", overwrite = TRUE
)
cat("Feature selection summary exported.\nStep 4 complete.\n\n")


# ==================== Part 5: Cross-Validation Controller ====================
cat("========== Step 5: Cross-Validation Setup ==========\n")

set.seed()#set the random seeds
# The `trainControl` parameter remains unchanged; the class weights are passed in through the `weights` parameter of `train()`.
cv_control <- trainControl(
  method          = "repeatedcv",
  number          = 5,     # 5-fold
  repeats         = 3,     # repeated 3 times — more stable with ~176 events
  savePredictions = "final",
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)
cat("CV configured: 5-fold x 3 repeats.\n\n")


# ==================== Part 6: Model Training ====================
# =========================================================================
cat("========== Step 6: Model Training (8 models) ==========\n")

train_data[[response_var]] <- factor(train_data[[response_var]], levels = c("no","yes"))
test_data[[response_var]]  <- factor(test_data[[response_var]],  levels = c("no","yes"))

model_settings <- data.frame(
  AlgorithmName = c(
    "Lasso", "DiscriminantModel", "LogisticModel",
    "SVM_RBF", "GradientBoosting", "NaiveBayes",
    "AdaptiveBoosting", "NeuralNet"
  ),
  Implementation = c(
    "glmnet", "lda", "glm",
    "svmRadial", "xgbTree", "nb",
    "AdaBoost.M1", "nnet"
  ),
  NeedsScaling = c(
    TRUE,   # Lasso
    TRUE,   # DiscriminantModel
    FALSE,  # LogisticModel
    TRUE,   # SVM_RBF
    FALSE,  # GradientBoosting
    FALSE,  # NaiveBayes
    FALSE,  # AdaptiveBoosting
    TRUE    # NeuralNet
  ),
  stringsAsFactors = FALSE
)

# ── Class Weight Calculation ──────────────────────────────────────────────────
# [L5] Inverse-frequency weighting, computed on TRAINING SET ONLY.
#      Ensures minority class (yes/NOAF) receives proportionally higher weight.
#      Weights are row-level vectors passed into caret::train().
#      Validation set (test_data) is never involved — no leakage.
# ─────────────────────────────────────────────────────────────────────────────

# Remove Patient_ID — identifier, not a predictor
train_data_model <- train_data %>% select(-all_of(patient_id_col))
test_data_model  <- test_data  %>% select(-all_of(patient_id_col))

outcome_table <- table(train_data_model[[response_var]])
class_weights <- (1 / outcome_table) / sum(1 / outcome_table) * length(train_data_model[[response_var]])

sample_weights <- ifelse(
  train_data_model[[response_var]] == "yes",
  class_weights["yes"],
  class_weights["no"]
)

cat(sprintf(
  "Class weights → no: %.4f | yes: %.4f  (ratio 1:%.2f)\n\n",
  class_weights["no"], class_weights["yes"],
  class_weights["yes"] / class_weights["no"]
))

# Methods that do NOT support external weights:
#   - nb        : silently ignores weights
#   - AdaBoost.M1 : has its own internal sample re-weighting mechanism;
#                   passing external weights may conflict with that logic
no_weight_methods <- c("nb", "AdaBoost.M1")

training_times <- numeric()
modelContainer <- list()
model_formula  <- as.formula(paste(response_var, "~ ."))

total_start <- Sys.time()
cat("Training started:", format(total_start, "%Y-%m-%d %H:%M:%S"), "\n")

for (idx in seq_len(nrow(model_settings))) {
  algoName   <- model_settings$AlgorithmName[idx]
  algoImpl   <- model_settings$Implementation[idx]
  needsScale <- model_settings$NeedsScaling[idx]
  useWeights <- !(algoImpl %in% no_weight_methods)
  start_time <- Sys.time()

  cat(sprintf(
    "\n[%d/%d] %-20s (%s) | Scale: %s | Weights: %s | %s\n",
    idx, nrow(model_settings), algoName, algoImpl,
    ifelse(needsScale, "YES — center+scale inside each fold [L4]", "NO"),
    ifelse(useWeights, "YES [L5]", "NO — method incompatible"),
    format(start_time, "%H:%M:%S")
  ))

  # NULL for tree-based models — no preprocessing needed
  preproc_steps <- if (needsScale) c("center", "scale") else NULL

  tryCatch({
    model <- train(
      model_formula,
      data       = train_data_model,
      method     = algoImpl,
      metric     = "ROC",
      trControl  = cv_control,
      preProcess = preproc_steps,                                  # [L4] fold-internal standardization
      weights    = if (useWeights) sample_weights else NULL        # [L5] class imbalance correction
    )
    modelContainer[[algoName]] <- model
    training_times[algoName]   <- as.numeric(
      difftime(Sys.time(), start_time, units = "secs"))
    cat(sprintf("[%d/%d] Done: %-20s | %.1f sec\n",
                idx, nrow(model_settings), algoName, training_times[algoName]))
  }, error = function(e) {
    cat(sprintf("[ERROR] %s training failed: %s\n", algoName, e$message))
  })
}

cat("\nAll models trained. Total elapsed:",
    round(as.numeric(difftime(Sys.time(), total_start, units = "mins")), 1),
    "min\n\n")

# Export training time record
write.xlsx(
  data.frame(
    Model        = names(training_times),
    Standardized = model_settings$NeedsScaling[
      match(names(training_times), model_settings$AlgorithmName)],
    Seconds      = unlist(training_times)
  ),
  "1_ML_Training_Time_Record.xlsx", overwrite = TRUE
)
cat("Training time record exported to 1_ML_Training_Time_Record.xlsx\n")

# Save model objects for downstream analysis
saveRDS(modelContainer,   "modelContainer.rds")
saveRDS(test_data_model,  "test_data_model.rds")
saveRDS(model_settings,   "model_settings.rds")
cat("Model objects saved: modelContainer.rds / test_data_model.rds / model_settings.rds\n\n")

cat("\nReady for downstream evaluation.\n")


```
