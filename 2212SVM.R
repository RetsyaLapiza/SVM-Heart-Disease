###KLASIFIKASI DENGAN SVM

###install dan called packages
install.packages("devtools")
library(e1071)
library(devtools)
library(caTools)
library(caret)
library(dplyr)
library(ggplot2)

###Preprosessing dan statistika deskriptif
data <- read.csv(file.choose())
summary(data)
sample_n(10, data)
anyNA(data)
dim(data)
colnames(data)
col <- c("ï..age", "sex", "target")

##khusus prepossesing ubah format data 
#sex
data$sex <- factor(data$sex, 
                   levels=c(0,1), 
                   labels=c("wanita","pria"))
#kadar gula darah (fbs)
data$fbs <- factor(data$fbs, 
                   levels=c(0,1), 
                   labels=c("tinggi","rendah"))
#eletrocardio (restecg)
data$restecg <- factor(data$restecg, 
                       levels=c(0,1,2), 
                       labels=c("normal","tidak normal","pembesaran bilik"))
#target(y)
data$target <- factor(data$target, 
                      levels=c(0,1), 
                      labels=c("penyakit jantung","tidak penyakit jantung"))
###EDA dan Visualisasi Data
#barplot jenis kelamin terhadap penyakit 
ggplot(data = data, mapping = aes(fill = target, x = sex)) + 
  geom_bar()
#histogram usia
ggplot(data = data, mapping = aes(ï..age)) + 
  geom_histogram(fill="blue")

###Split data dengan train 80% dan test 20%
set.seed(12)
split = sample.split(data$target, SplitRatio = 0.8)

training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

##cek split data
tail(training_set)
tail(test_set)

dim(training_set)
dim(test_set)

summary(training_set)

###Pemodelan dengan trik kernel
##kernel linier acc 86,36%
svm1 <- svm(target~., data = training_set, kernel = "linear")
prediksi1 <- predict(svm1, training_set)
confusionMatrix(prediksi1, training_set$target)

##kernel polynomial acc 92,98% 
svm2 <- svm(target~., data = training_set, kernel = "polynomial")
prediksi2 <- predict(svm2, training_set)
confusionMatrix(prediksi2, training_set$target)

##kernel radial acc 92,15%
svm3 <- svm(target~., data = training_set, kernel = "radial")
prediksi3 <- predict(svm3, training_set)
confusionMatrix(prediksi3, training_set$target)

##kernel sigmoid acc 81,82% 
svm4 <- svm(target~., data = training_set, kernel = "sigmoid")
prediksi4 <- predict(svm4, training_set)
confusionMatrix(prediksi4, training_set$target)

#oleh karena itu kernel trik terbaik berdasarkan akurasi adalah polynomial
#dengan parameter gamma = 1/14; degree = 3; coef = 0.

#melakukan tuning parameter dengan CV untuk mengetahui nilai cost terbaik
###K-Fold CrossValidatonValidation
###Tunning dengan CV
tuned <- tune(svm,target~.,data=training_set,kernel= "polynomial",
              ranges = list(degree = 3, gamma = 1/14,
                            coef= 0, cost=seq(from = 0.05, to = 4,by = 0.05)))
summary(tuned)
#hasil menunjukkan cost = 2 yang terbaik

##kernel polynomial acc 92,98% 
svm2 <- svm(target~., data = training_set, kernel = "polynomial", cost =2)
prediksi2 <- predict(svm2, training_set)
confusionMatrix(prediksi2, training_set$target)

####Testing model
test_pred <- predict(svm2, newdata = test_set)

###Akurasi terhadap 
confusionMatrix(table(test_pred, test_set$target))

###Plot SVM
plot(svm2,training_set[,data])


###Berdasarkan hasil perhitungan kernel trik terbaik adalah polynomial dengan
###parameter gamma = 1/14 ; coef = 0; degree = 3; dan cost = 2
###hasil klasifikasi dengan SVM dan parameter diatas menunjukkan akurasi sebesar 78,69%
##hal ini menunjukkan model SVM yang dibangun tergolong baik