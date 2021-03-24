library(rpart)
library(lattice)
library(ggplot2)
library(caret)
library(corrplot)
library(caTools)
library(rpart.plot)
library(randomForest)
library(DMwR) 
library(gridExtra)

setwd("C:/Users/SB052807/Desktop/AI/DataSet-Diabetes/")
problem4_diabetes <- read.csv("problem4_diabetes.csv")

# Create training and testing set

set.seed(12345)
ratio = sample(1:nrow(problem4_diabetes), size = 0.30*nrow(problem4_diabetes))
testing_diabetes = problem4_diabetes[ratio,] #Test dataset 30% of total
training_diabetes = problem4_diabetes[-ratio,] #Train dataset 70% of total

## Dealing with zeros
missing_data <- problem4_diabetes[,setdiff(names(problem4_diabetes), c('Outcome', 'Pregnancies'))]
features_miss_num <- apply(missing_data, 2, function(x) sum(x <= 0))
features_miss <- names(missing_data)[ features_miss_num > 0]

rows_miss <- apply(missing_data, 1, function(x) sum(x <= 0) >= 1) 
sum(rows_miss)

missing_data[missing_data <= 0] <- NA
problem4_diabetes[, names(missing_data)] <- missing_data

#Data Pre-Processing

colSums(is.na(problem4_diabetes))#Checking for invalid data

#Since Skin Thickness and Glucose had 227 and 374 zero-values respectively 
# KNN imputation
#So performing knn Imputation to correct the zero values.
problem4_diabetes[,c(-8,-9)] <- knnImputation(problem4_diabetes[,c(-8,-9)], k = 5)
colSums(is.na(problem4_diabetes))#Checking for invalid data

#Plotting the features against Outcome

p2 <- ggplot(problem4_diabetes, aes(x = Glucose, color = Outcome, fill = as.factor(Outcome))) + geom_density(alpha = 0.8) + theme(legend.position = "bottom") +labs(x = "Glucose", y = "Density", title = "Density plot of glucose")
p1 <- ggplot(problem4_diabetes, aes(x = Outcome, y = Glucose,fill = as.factor(Outcome))) + geom_boxplot() + theme(legend.position = "bottom") + ggtitle("Variation of glucose Vs Diabetes")

gridExtra::grid.arrange(p1,p2, ncol=2)

p4 <- ggplot(problem4_diabetes, aes(x = BloodPressure, color = Outcome, fill = as.factor(Outcome))) + geom_density(alpha = 0.8) + theme(legend.position = "bottom") +labs(x = "BloodPressure", y = "Density", title = "Density plot of BloodPressure")
p3 <- ggplot(problem4_diabetes, aes(x = Outcome, y = BloodPressure,fill = as.factor(Outcome))) + geom_boxplot() + theme(legend.position = "bottom") + ggtitle("Variation of BloodPressure Vs Diabetes")

gridExtra::grid.arrange(p3,p4, ncol=2)


#Correlation Graph

corMat = cor (problem4_diabetes[, -9])
diag (corMat) = 0 #Remove self correlations
corrplot.mixed(corMat,tl.pos = "lt") 


# MODEL- Logistic Regression Model(GLM)

diabetes_glm_model <- glm (Outcome ~ ., data = training_diabetes, family = binomial)
step_model <- step(diabetes_glm_model) 

#glm model with the lowest AIC value 
#diabetes_glm_model_final <- glm (Outcome ~ Pregnancies + Glucose + BMI + DiabetesPedigreeFunction, data = training_diabetes, family = binomial)

#Prediction 

diabetes_predicted <- predict(diabetes_glm_model,testing_diabetes, type="response")
rounded_diabetes_predicted<-round(diabetes_predicted)

#Confusion Matrix

cm_glm<-confusionMatrix(rounded_diabetes_predicted,testing_diabetes$Outcome )
print(cm_glm)



# MODEL- Random Forest

diabetes_model_RandomForest <- randomForest(Outcome ~ .,data=training_diabetes,importance =TRUE)

#Prediction
diabetes_predicted_rf<-predict(diabetes_model_RandomForest,testing_diabetes,type ="class")
rounded_diabetes_predicted_rf<-round(diabetes_predicted_rf)

#Confusion Matrix
cm_rf <- confusionMatrix(rounded_diabetes_predicted_rf,testing_diabetes$Outcome)
print(cm_rf)

# MODEL- Decision Tree

diabetes_model_DecisionTree <- rpart(Outcome~., data=training_diabetes, method="class")
rpart.plot(diabetes_model_DecisionTree)

#Prediction
diabetes_predicted_DT<- predict(diabetes_model_DecisionTree, testing_diabetes, type = "class")

#CONFUSION MATRIX

cm_dt <- confusionMatrix(diabetes_predicted_DT,testing_diabetes$Outcome)
print(cm_dt)
