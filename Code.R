# Load required libraries
# These libraries are essential for data manipulation, visualization, modeling, and evaluation.
library(dplyr)        # Data manipulation
library(ggplot2)      # Data visualization
library(caret)        # Machine learning and data splitting
library(randomForest) # Random Forest model
library(cluster)      # Clustering analysis (if needed)
library(factoextra)   # Factor analysis visualization
library(lubridate)    # Date and time manipulation
library(gbm)          # Gradient Boosting Machine
library(e1071)        # Support Vector Machine
library(MLmetrics)    # For calculating F1 Score
library(smotefamily)  # SMOTE for class balancing
library(reshape2)


# Load the dataset
data <- read.csv("Occupancy_Estimation.csv")

# 1. Initial Data Cleaning and Pre-processing
# Check the structure of the data to understand the data types and dimensions.
str(data)

# Display summary statistics for each column to identify any anomalies or outliers.
summary(data)

# Check for missing values and handle them
cat("Number of missing values in each column:\n")
print(colSums(is.na(data)))

# Remove rows with missing values (or apply a different strategy as needed).
data <- na.omit(data)


# Correlation Analysis for Numeric Features
# Select only numeric columns for correlation analysis.
numeric_features <- data %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_features)
print(correlation_matrix)

# Visualize the correlation matrix to see relationships between numeric features.
ggcorrplot::ggcorrplot(correlation_matrix, title = "Correlation Matrix of Numeric Features")

# Distribution Plots for Key Features
# 1. Distribution of S1_Temp - To check the spread of temperature values from Sensor 1.
ggplot(data, aes(x = S1_Temp)) +
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  ggtitle("Distribution of S1_Temp") +
  xlab("S1_Temp") +
  ylab("Frequency") +
  theme_minimal()

# 2. Distribution of S1_Light - To analyze light intensity values from Sensor 1.
ggplot(data, aes(x = S1_Light)) +
  geom_histogram(binwidth = 10, fill = "lightgreen", color = "black") +
  ggtitle("Distribution of S1_Light Intensity") +
  xlab("S1_Light") +
  ylab("Frequency") +
  theme_minimal()

# 3. Distribution of CO2 Levels - To examine CO2 level variations in the environment.
ggplot(data, aes(x = S5_CO2)) +
  geom_histogram(binwidth = 50, fill = "lightpink", color = "black") +
  ggtitle("Distribution of CO2 Levels (S5_CO2)") +
  xlab("CO2 Level") +
  ylab("Frequency") +
  theme_minimal()

# Room Occupancy Analysis
# 4. Plot to see the frequency distribution of room occupancy counts.
ggplot(data, aes(x = Room_Occupancy_Count)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Count of Room Occupancy") +
  xlab("Room Occupancy Count") +
  ylab("Frequency") +
  theme_minimal()

# 5. Scatter plot for S1_Temp vs Room_Occupancy_Count
ggplot(data, aes(x = S1_Temp, y = Room_Occupancy_Count)) +
  geom_point(color = "blue", alpha = 0.5) +
  ggtitle("Scatter Plot of S1_Temp vs Room Occupancy Count") +
  xlab("S1_Temp") +
  ylab("Room Occupancy Count") +
  theme_minimal()

# 6. Scatter plot for S5_CO2 vs Room_Occupancy_Count
ggplot(data, aes(x = S5_CO2, y = Room_Occupancy_Count)) +
  geom_point(color = "red", alpha = 0.5) +
  ggtitle("Scatter Plot of S5_CO2 vs Room Occupancy Count") +
  xlab("S5_CO2") +
  ylab("Room Occupancy Count") +
  theme_minimal()


# 2. Verify Data Types and Convert if Necessary
# Convert Date and Time columns to character type to combine them.
data$Date <- as.character(data$Date)
data$Time <- as.character(data$Time)

# Combine Date and Time columns into a single time-stamp column for temporal analysis.
data$timestamp <- ymd_hms(paste(data$Date, data$Time))

# Extract day of the week and hour from the time-stamp to analyze temporal trends.
data$day_of_week <- weekdays(data$timestamp)
data$hour <- hour(data$timestamp)

# 7. Average Occupancy by Day of the Week - Visualizing trends in room occupancy by weekday.
ggplot(data, aes(x = day_of_week, y = Room_Occupancy_Count, fill = day_of_week)) +
  stat_summary(fun = mean, geom = "bar") +
  ggtitle("Average Room Occupancy by Day of the Week") +
  xlab("Day of the Week") +
  ylab("Average Occupancy Count") +
  theme_minimal()

# 8. Hourly Occupancy Patterns - Analyzing average occupancy patterns over hours.
ggplot(data, aes(x = hour, y = Room_Occupancy_Count)) +
  stat_summary(fun = mean, geom = "line", color = "blue") +
  ggtitle("Hourly Room Occupancy Patterns") +
  xlab("Hour of the Day") +
  ylab("Average Occupancy Count") +
  theme_minimal()

# Sensor Data Analysis by Room Occupancy
# 9. Box-plot of CO2 Levels by Room Occupancy - To compare CO2 levels across different occupancy counts.
ggplot(data, aes(x = as.factor(Room_Occupancy_Count), y = S5_CO2, fill = as.factor(Room_Occupancy_Count))) +
  geom_boxplot() +
  ggtitle("CO2 Levels by Room Occupancy") +
  xlab("Room Occupancy Count") +
  ylab("CO2 Levels (S5_CO2)") +
  theme_minimal()

# 10. Sound Levels by Room Occupancy - Examining sound levels across occupancy counts.
ggplot(data, aes(x = as.factor(Room_Occupancy_Count), y = S1_Sound)) +
  geom_boxplot(fill = "lightgreen") +
  ggtitle("Sound Levels by Room Occupancy") +
  xlab("Room Occupancy Count") +
  ylab("Sound Level (S1_Sound)") +
  theme_minimal()

# 11. PIR Sensor Activity by Hour - Analyzing PIR (motion) sensor activity over time.
ggplot(data, aes(x = hour, y = S6_PIR)) +
  stat_summary(fun = mean, geom = "bar", fill = "steelblue") +
  ggtitle("PIR Sensor Activity by Hour") +
  xlab("Hour of Day") +
  ylab("Average PIR Activity (S6_PIR)") +
  theme_minimal()

# 12. Heatmap of Occupancy Count by Hour and Day - To identify peak occupancy times by weekday and hour.
occupancy_heatmap <- data %>%
  group_by(day_of_week, hour) %>%
  summarize(avg_occupancy = mean(as.numeric(as.character(Room_Occupancy_Count))))

ggplot(occupancy_heatmap, aes(x = hour, y = day_of_week, fill = avg_occupancy)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ggtitle("Heatmap of Average Occupancy by Hour and Day of the Week") +
  xlab("Hour of Day") +
  ylab("Day of the Week") +
  theme_minimal()


# 13. Room Occupancy by Light Intensity Range (Using S1_Light)
data$light_level <- cut(data$S1_Light, breaks = c(-Inf, 100, 500, Inf), labels = c("low", "medium", "high"))
ggplot(data, aes(x = light_level, y = Room_Occupancy_Count, fill = light_level)) +
  stat_summary(fun = mean, geom = "bar") +
  ggtitle("Room Occupancy by Light Intensity Range") +
  xlab("Light Intensity Level") +
  ylab("Average Occupancy Count") +
  theme_minimal()

# 14. Average Temperature by Room Occupancy Level (Using S1_Temp)
ggplot(data, aes(x = as.factor(Room_Occupancy_Count), y = S1_Temp, fill = as.factor(Room_Occupancy_Count))) +
  geom_boxplot() +
  ggtitle("Average Temperature by Room Occupancy Level") +
  xlab("Room Occupancy Count") +
  ylab("Temperature (S1_Temp)") +
  theme_minimal()


# 15. Average Occupancy by Month
data$month <- month(data$timestamp, label = TRUE)
ggplot(data, aes(x = month, y = as.numeric(as.character(Room_Occupancy_Count)), fill = month)) +
  stat_summary(fun = mean, geom = "bar") +
  ggtitle("Average Occupancy by Month") +
  xlab("Month") +
  ylab("Average Occupancy Count") +
  theme_minimal()



# 3. Feature Engineering
# Converting Room Occupancy Count to factor as it’s a categorical target variable for classification tasks.
data$Room_Occupancy_Count <- as.factor(data$Room_Occupancy_Count)

# Creating new features based on hour and time stamp for deeper analysis.
data$hour <- hour(data$timestamp)
data$part_of_day <- case_when(
  data$hour >= 6 & data$hour < 12 ~ "morning",
  data$hour >= 12 & data$hour < 18 ~ "afternoon",
  data$hour >= 18 & data$hour < 24 ~ "evening",
  TRUE ~ "night"
)

# Adding additional derived features (e.g., temperature range, weekend indicator).
data <- data %>%
  mutate(
    temp_range = cut(S1_Temp, breaks = c(-Inf, 15, 25, Inf), labels = c("cool", "moderate", "warm")),
    is_weekend = ifelse(weekdays(timestamp) %in% c("Saturday", "Sunday"), 1, 0)
  )


# 16. Average Occupancy Count by Temperature and Light Levels
ggplot(data, aes(x = temp_range, y = as.numeric(as.character(Room_Occupancy_Count)), fill = light_level)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  ggtitle("Average Occupancy Count by Temperature and Light Levels") +
  xlab("Temperature Range") +
  ylab("Average Occupancy Count") +
  labs(fill = "Light Level") +
  theme_minimal()


# Splitting data into training and testing sets (80%-20% split).
set.seed(123)
trainIndex <- createDataPartition(data$Room_Occupancy_Count, p = .8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Drop non-numeric columns like Date, Time, day_of_week, part_of_day, and time-stamp
train_data <- train_data %>% select(-Date, -Time, -day_of_week, -part_of_day, -timestamp)

# Convert all non-numeric columns to numeric, except the target variable
train_data_numeric <- train_data %>% 
  mutate_if(is.factor, as.numeric) %>%  # Convert factors to numeric
  mutate_if(is.character, as.numeric)    # Convert characters to numeric

# Print the structure of train_data_numeric to inspect non-numeric columns, if any
print("Structure of train_data_numeric after conversion:")
str(train_data_numeric)

# Verify that all columns are now numeric
if (!all(sapply(train_data_numeric, is.numeric))) {
  stop("Error: Non-numeric data detected after conversion.")
}

# Applying SMOTE to balance the training data
smote_result <- SMOTE(X = train_data_numeric[, -which(names(train_data_numeric) == "Room_Occupancy_Count")], 
                      target = train_data_numeric$Room_Occupancy_Count, 
                      K = 5, dup_size = 2)

# Combine SMOTE-generated data with the original columns
train_data_smote <- data.frame(smote_result$data)
train_data_smote$Room_Occupancy_Count <- factor(train_data_smote$class, labels = levels(train_data$Room_Occupancy_Count))
train_data_smote$class <- NULL  # Remove the redundant 'class' column

# Ensure the distribution of classes is balanced after SMOTE
print("Class distribution after SMOTE:")
table(train_data_smote$Room_Occupancy_Count)

# Ensure train and test data have at least one row.
if (nrow(train_data) == 0 || nrow(test_data) == 0) {
  stop("Error: Training or testing data split failed. Check 'createDataPartition' function parameters.")
}



# Model Training

# Define a helper function to calculate F1 Score with level alignment
calculate_f1 <- function(pred, actual) {
  pred <- factor(pred, levels = levels(factor(actual))) # Align levels
  F1_Score(pred, factor(actual), positive = levels(factor(actual))[1])
}

# Define a helper function to calculate Recall with level alignment
calculate_recall <- function(pred, actual) {
  pred <- factor(pred, levels = levels(factor(actual))) # Align levels
  confusion <- confusionMatrix(pred, factor(actual))
  mean(confusion$byClass[, "Recall"], na.rm = TRUE) # Average recall for all classes
}

# Define a helper function to calculate Precision with level alignment
calculate_precision <- function(pred, actual) {
  pred <- factor(pred, levels = levels(factor(actual))) # Align levels
  confusion <- confusionMatrix(pred, factor(actual))
  mean(confusion$byClass[, "Precision"], na.rm = TRUE) # Average precision for all classes
}

plot_confusion_matrix <- function(confusion_matrix, model_name) {
  confusion_data <- as.data.frame(confusion_matrix$table)
  ggplot(confusion_data, aes(Prediction, Reference, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "blue") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    ggtitle(paste(model_name, "Confusion Matrix")) +
    theme_minimal()
}


# Initialize a data frame to store model metrics with Precision
model_metrics <- data.frame(Model = character(), Accuracy = numeric(), F1_Score = numeric(), Recall = numeric(), Precision = numeric(), stringsAsFactors = FALSE)



# 1. Poisson Regression
train_data_numeric <- train_data_smote
test_data_numeric <- test_data
train_data_numeric$Room_Occupancy_Count <- as.numeric(as.character(train_data_numeric$Room_Occupancy_Count))
test_data_numeric$Room_Occupancy_Count <- as.numeric(as.character(test_data_numeric$Room_Occupancy_Count))

poisson_model <- glm(Room_Occupancy_Count ~ S1_Temp + hour + is_weekend, data = train_data_numeric, family = poisson)
poisson_pred <- predict(poisson_model, newdata = test_data_numeric, type = "response")
poisson_pred_class <- round(poisson_pred)
poisson_accuracy <- mean(poisson_pred_class == test_data_numeric$Room_Occupancy_Count)
print(paste("Poisson Regression Accuracy:", poisson_accuracy))

# Poisson Regression F1 Score, Precision and Recall
poisson_f1 <- calculate_f1(poisson_pred_class, test_data_numeric$Room_Occupancy_Count)
poisson_recall <- calculate_recall(poisson_pred_class, test_data_numeric$Room_Occupancy_Count)
poisson_precision <- calculate_precision(poisson_pred_class, test_data_numeric$Room_Occupancy_Count)

# Add results to model_metrics
model_metrics <- rbind(model_metrics, data.frame(Model = "Poisson Regression", Accuracy = poisson_accuracy, F1_Score = poisson_f1, Recall = poisson_recall, Precision = poisson_precision))

# Print and plot confusion matrix for Poisson Regression
poisson_confusion <- confusionMatrix(factor(poisson_pred_class, levels = levels(factor(test_data_numeric$Room_Occupancy_Count))), factor(test_data_numeric$Room_Occupancy_Count))
print("Poisson Regression Confusion Matrix:")
print(poisson_confusion)
plot_confusion_matrix(poisson_confusion, "Poisson Regression")


# 2. Decision Tree
tree_model <- train(Room_Occupancy_Count ~ S1_Temp + hour + is_weekend, data = train_data_smote, method = "rpart")
tree_pred <- predict(tree_model, newdata = test_data)
tree_accuracy <- mean(tree_pred == test_data$Room_Occupancy_Count)
print(paste("Decision Tree Accuracy:", tree_accuracy))

# Decision Tree F1 Score, Precision and Recall
tree_f1 <- calculate_f1(tree_pred, test_data$Room_Occupancy_Count)
tree_recall <- calculate_recall(tree_pred, test_data$Room_Occupancy_Count)
tree_precision <- calculate_precision(tree_pred, test_data$Room_Occupancy_Count)
model_metrics <- rbind(model_metrics, data.frame(Model = "Decision Tree", Accuracy = tree_accuracy, F1_Score = tree_f1, Recall = tree_recall, Precision = tree_precision))

# Print and plot confusion matrix for Decision Tree
tree_confusion <- confusionMatrix(tree_pred, test_data$Room_Occupancy_Count)
print("Decision Tree Confusion Matrix:")
print(tree_confusion)
plot_confusion_matrix(tree_confusion, "Decision Tree")



# 3. Random Forest
rf_model <- randomForest(Room_Occupancy_Count ~ S1_Temp + hour + is_weekend, data = train_data_smote)
rf_pred <- predict(rf_model, newdata = test_data)
rf_pred <- factor(rf_pred, levels = levels(test_data$Room_Occupancy_Count))
rf_accuracy <- mean(rf_pred == test_data$Room_Occupancy_Count)
print(paste("Random Forest Accuracy:", rf_accuracy))

# Random Forest F1 Score, Precision and Recall
rf_f1 <- calculate_f1(rf_pred, test_data$Room_Occupancy_Count)
rf_recall <- calculate_recall(rf_pred, test_data$Room_Occupancy_Count)
rf_precision <- calculate_precision(rf_pred, test_data$Room_Occupancy_Count)
model_metrics <- rbind(model_metrics, data.frame(Model = "Random Forest", Accuracy = rf_accuracy, F1_Score = rf_f1, Recall = rf_recall, Precision = rf_precision))

# Print and plot confusion matrix for Random Forest
rf_confusion <- confusionMatrix(rf_pred, test_data$Room_Occupancy_Count)
print("Random Forest Confusion Matrix:")
print(rf_confusion)
plot_confusion_matrix(rf_confusion, "Random Forest")



# 4. Gradient Boosting Machine (GBM)
set.seed(123)
gbm_model <- train(Room_Occupancy_Count ~ S1_Temp + hour + is_weekend,
                   data = train_data_smote,
                   method = "gbm",
                   trControl = trainControl(method = "cv", number = 5),
                   verbose = FALSE)
gbm_pred <- predict(gbm_model, newdata = test_data)
gbm_accuracy <- mean(gbm_pred == test_data$Room_Occupancy_Count)
print(paste("GBM Accuracy:", gbm_accuracy))

cat("Confusion Matrix for GBM:\n")
print(confusionMatrix(gbm_pred, test_data$Room_Occupancy_Count))

# GBM F1 Score, Precision and Recall
gbm_f1 <- calculate_f1(gbm_pred, test_data$Room_Occupancy_Count)
gbm_recall <- calculate_recall(gbm_pred, test_data$Room_Occupancy_Count)
gbm_precision <- calculate_precision(gbm_pred, test_data$Room_Occupancy_Count)
model_metrics <- rbind(model_metrics, data.frame(Model = "GBM", Accuracy = gbm_accuracy, F1_Score = gbm_f1, Recall = gbm_recall, Precision = gbm_precision))

# Print and plot confusion matrix for GBM
gbm_confusion <- confusionMatrix(gbm_pred, test_data$Room_Occupancy_Count)
print("GBM Confusion Matrix:")
print(gbm_confusion)
plot_confusion_matrix(gbm_confusion, "GBM")



# 5. Support Vector Machine (SVM)
svm_model <- svm(Room_Occupancy_Count ~ S1_Temp + hour + is_weekend, data = train_data_smote, kernel = "radial")
svm_pred <- predict(svm_model, newdata = test_data)
svm_accuracy <- mean(svm_pred == test_data$Room_Occupancy_Count)
print(paste("SVM Accuracy:", svm_accuracy))
cat("Confusion Matrix for SVM:\n")
print(confusionMatrix(svm_pred, test_data$Room_Occupancy_Count))

# SVM F1 Score, Precision and Recall
svm_f1 <- calculate_f1(svm_pred, test_data$Room_Occupancy_Count)
svm_recall <- calculate_recall(svm_pred, test_data$Room_Occupancy_Count)
svm_precision <- calculate_precision(svm_pred, test_data$Room_Occupancy_Count)
model_metrics <- rbind(model_metrics, data.frame(Model = "SVM", Accuracy = svm_accuracy, F1_Score = svm_f1, Recall = svm_recall, Precision = svm_precision))


# Print and plot confusion matrix for SVM
svm_confusion <- confusionMatrix(svm_pred, test_data$Room_Occupancy_Count)
print("SVM Confusion Matrix:")
print(svm_confusion)
plot_confusion_matrix(svm_confusion, "SVM")




# Model Evaluation
# Model Comparison Summary
cat("Model Accuracies:\n")
cat("Poisson Regression Accuracy:", poisson_accuracy, "\n")
cat("Decision Tree Accuracy:", tree_accuracy, "\n")
cat("Random Forest Accuracy:", rf_accuracy, "\n")
cat("GBM Accuracy:", gbm_accuracy, "\n")
cat("SVM Accuracy:", svm_accuracy, "\n")

# Print model metrics for comparison
print(model_metrics)

# Visualization: Model Metrics Comparison
model_metrics_long <- melt(model_metrics, id.vars = "Model", variable.name = "Metric", value.name = "Value")

ggplot(model_metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Model Metrics Comparison (Accuracy, F1 Score, Recall, Precision)") +
  xlab("Model") +
  ylab("Score") +
  theme_minimal()