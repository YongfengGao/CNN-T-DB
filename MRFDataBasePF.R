library(randomForest)
library(pROC)
library(tree)
#library(reprtree)
library(plyr)
library(caret)
library(PRROC)
library(ggplot2)
library(latex2exp)

set.seed(123)
theme_set(theme_bw())

load_my_data = function(x) {
  data = read.csv(x, header = FALSE)
  data[,1] = as.factor(data[,1])
  data[,10] = as.factor(data[,10])

  data=data[,1:13]
  
  data
}

train_model = function(data) {
  model = randomForest(V1 ~ ., data = data[,-c(2:5)],
                       ntree = 500,
                       importance = T,
                       proximity =F)
  
  # reprtree:::plot.getTree(model, 1)
  
  # varImpPlot(model)
  
  # MDSplot(model, data$Label, 2)
  
  model
}


test_model = function(model, test) {
  p_test = predict(model, test, type = 'prob')
  roc = roc(test$label, p_test[,2])
  plot(roc,
       type = "S",
       main = paste("ROC, AUC = ", format(roc$auc,digits=3)),
       print.thres = c(0.9),
       print.thres.pattern = " Thrs = %.2f (Spec = %.4f, Sens = %.4f)", 
       print.thres.cex = 0.8)
  auc = auc(roc)
  print(auc)
  
  pr = pr.curve(scores.class0 = p_test[test$label==1,2], scores.class1 = p_test[test$label==0,2], curve = TRUE)
  plot(pr)
  
  pr
}


test_model2 = function(test,model) {
  p_test = predict(model, test[,-c(2:5)], type = 'prob')
  p_test_sort = sort(p_test[,2],decreasing = T, index.return = T)
  p_id <- test$V2[p_test_sort$ix[1:20]]
  s_id <- test$V3[p_test_sort$ix[1:20]]
  p_id_can <- test$V4[p_test_sort$ix[1:20]]
  s_id_can <- test$V5[p_test_sort$ix[1:20]]
  label_can <- test$V1[p_test_sort$ix[1:20]]
  out_result <- cbind(p_id, s_id, p_id_can,s_id_can, label_can, p_test[p_test_sort$ix[1:20],2])
  out_result <- t(out_result)
  print(out_result)
  return(out_result)
  
}


#############################################start the main function###########################

root1 = "input\\Training\\"
setwd(root1);

files = list.files(path=root1, pattern="*.csv", full.names=T, recursive=FALSE)
data_list1 = lapply(files,load_my_data)
train = do.call("rbind", data_list1)

# allocate partial data for training ####
ind = sample(2, nrow(train), replace = T, prob = c(0.80, 0.20))
train_sample = train[ind==1,]

# start the training 
# MRFCanPG.rf = train_model(train_sample)
MRFCanPG80.rf = train_model(train_sample)
# save the trained model 
root3 = "models"
setwd(root3);
save(MRFCanPG80.rf, file = "MRFDB80.RData")
# use the following sytax to load the previous trained model
#super_model <- readRDS("./final_model.rds")
#load("MRFDB.RData")
 

# set up for test 
root2 = "input\\Testing\\"
setwd(root2);
files_test = list.files(path=root2, pattern="*.csv", full.names=T, recursive=FALSE)


# testing scheme 1
pre<- c()
for (i in 1:2)
  {data_test = load_my_data(files_test[i])
  p_test = predict(MRFCanPG.rf,data_test, type = 'prob')
  p_test_sort = sort(p_test[,2],decreasing = T, index.return = T)
  p_id_can <- data_test$V4[p_test_sort$ix[1]]
  s_id_can <- data_test$V5[p_test_sort$ix[1]]
  label_can <- data_test$V1[p_test_sort$ix[1]]
  pre_temp <- c(p_id_can,s_id_can, label_can)
  pre <- rbind(pre,pre_temp,byrow = T)}

  
# testing scheme 2 
data_list2_temp = lapply(files_test,load_my_data)
pre_1 = lapply(data_list2_temp, test_model2,MRFCanPG.rf)
pre_vec <- matrix(unlist(pre_1),ncol = 6, byrow = T)
#plot(pre_vec[,3])
write.table(pre_vec,"PrePositive.txt", append = FALSE, sep = "\t ", dec = ".",
            row.names = FALSE, col.names = TRUE)


pre1 = lapply(data_list2_temp, test_model2,model)
pre_vec1 <- matrix(unlist(pre),ncol = 3, byrow = T)
plot(pre_vec1[,3])
write.table(pre_vec1,"PrePositive.txt", append = FALSE, sep = "\t ", dec = ".",
            row.names = FALSE, col.names = TRUE)











