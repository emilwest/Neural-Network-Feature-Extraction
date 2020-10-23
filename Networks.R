library(keras)
library(tidyverse)
library(glmnet)
library(caret)
library(doParallel) # for parallel computing
library(parallel) # for automatic detection of cores etc
library(kableExtra)

####
### importing data ####
#base_dir <- "C:\\Users\\emwe9516\\Desktop\\HW2\\img_align_celeba\\img_align_celeba"
#train_dir <- file.path(base_dir, "train")
#fnames <- list.files(base_dir, full.names = TRUE)

#### transforming the data ####
################# change the number of observations later!!!!!!!!!!!!!!!!!!!!! ####
# data <- array(0, c(8000,150,150,3))
# 
# for(i in 1:8000){
#   img_path <- fnames[[i]]
#   
#   # Convert it to an array with shape (150, 150, 3)
#   img <- image_load(img_path, target_size = c(150, 150))
#   img_array <- image_to_array(img)
#   data[i,,,] <- array_reshape(img_array/255, c(150, 150, 3))
# }
# getwd()
#save(data,file="data_8000.RData")


# LOADS THE DATA FILE WITH ThE IMAGE ARRAYS
load(file = "C:\\Users\\emwe9516\\Desktop\\HW2\\GROUP\\data_8000.RData")

#### importing the labels ###
labs <- "C:\\Users\\emwe9516\\Desktop\\HW2\\img_align_celeba\\list_attr_celeba.txt"
d <- as.tibble(read.table(labs, header=T)) 
d <- d[1:8000,]

# converts columns 2:41 from -1 to 0 otherwise 1 and cbind with id column
# so we have it in binary format: 0 or 1
labels <- as.tibble(cbind(id=d$id,ifelse(d[,2:41]==-1,0,1)))

#### START !!! ####
###Dividing train/test/validation data
set.seed(40)
train.validation.ind <- sample(1:8000, 2000)

train.validation <- data[train.validation.ind,,,]
train.validation.labels <- labels$Male[train.validation.ind]
train.validation.labels.full <- labels[train.validation.ind,]

test <- data[-train.validation.ind,,,]
test.labels.full <- labels[-train.validation.ind,]
test.labels <- labels$Male[-train.validation.ind]

set.seed(40)
train.ind <- sample(1:2000, 0.8*2000)

training <- train.validation[train.ind,,,]
training.labels <- train.validation.labels[train.ind]
training.labels.full <- train.validation.labels.full[train.ind,]

validation <- train.validation[-train.ind,,,]
validation.labels <- train.validation.labels[-train.ind]
validation.labels.full <- train.validation.labels.full[-train.ind,]

#### checking if the splits are balanced
sum(training.labels)/1600
sum(validation.labels)/400
sum(test.labels)/6000


#### data augmention ####
datagen = image_data_generator(
  rotation_range = 30,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_generator <- flow_images_from_data(
  x=array_reshape(training[,,,],c(1600,150,150,3)),
  y=training.labels,
  generator = datagen,
  batch_size = 30,
  shuffle=F
)

###preparing validation data(validation data shouldn't be augmented)
validation_generator <- flow_images_from_data(
  x=array_reshape(validation[,,,],c(400,150,150,3)),
  y=validation.labels,
  batch_size = 30
)


#####################################
#Simple CNN on unaugmented data######
#####################################
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

freeze_weights(conv_base)

model.baseline <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model.baseline %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

not.augment <- model.baseline %>% fit(
  x=training,y=training.labels,
  validation_data=list(validation,validation.labels),
  epochs = 50,
  batch_size = 100,
  workers=16
)

plot(not.augment)



###############################################
#CNN on augmented data with regularisations ###
###############################################

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

freeze_weights(conv_base)

model.final.ep50 <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 500, activation = "relu") %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model.final.ep50 %>% compile(
  optimizer = optimizer_rmsprop(lr=0.00001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

final <- model.final.ep50 %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data=validation_generator,
  validation_steps = 50,
  workers=16
)
plot(final)


model.final.ep50 %>% save_model_hdf5("bestfitNN.h5")

################### results of our final NN-model #####################
result.ep50 <- model.final.ep50 %>% evaluate(test, test.labels)
test.features.1 <- model.final.ep50 %>% predict(x=test)

Confusion.Matrix.BestfitNN <- table(test.labels,ifelse(test.features.1>0.5,1,0))

kable(Confusion.Matrix.BestfitNN, "latex")


########################################
####### feature extraction #############
########################################

################ pop the last layer ###################################
model.final.ep50 %>% pop_layer()
train.validation.features.10 <- model.final.ep50 %>% predict(x=train.validation)
test.features.10 <- model.final.ep50 %>% predict(x=test)

train.valid.data.10 <- cbind(y=train.validation.labels,train.validation.features.10)
test.data.10 <- cbind(y=test.labels,test.features.10)

###LASSO###
lasso.10 <- cv.glmnet(train.validation.features.10, y=train.validation.labels, family = "binomial", type.measure = "class", alpha=1)
lasso_model.10 <- glmnet(x = train.validation.features.10, y = train.validation.labels, family = "binomial",  alpha = 1, lambda = lasso.10$lambda.min)

predictions_lasso.10 <- predict(object =  lasso_model.10, newx = as.matrix(test.features.10), type = "class", s = "lambda.min") %>% as.vector() %>% as.integer()

Confusion.Matrix.lasso.10 <- table(test.labels,predictions_lasso.10)
kable(Confusion.Matrix.lasso.10, "latex")

###RF######

cl <- makePSOCKcluster(parallel::detectCores()) # detects number of cores automatically
registerDoParallel(cl) # register the cores to use 100% of the CPU


control <- trainControl(method='cv', 
                        number=10)

rf.10<-caret::train(as.factor(y) ~., 
                    data = as.data.frame(train.valid.data.10),
                    method = 'rf', # random forest
                    metric = 'Accuracy',
                    trControl = control)


stopCluster(cl)

predict.rf.10 <-predict(rf.10, newdata = test.data.10, type = "raw")

Confusion.Matrix.rf.10 <-table(test.labels,predict.rf.10)#####
kable(Confusion.Matrix.rf.10, "latex")



##############pop the 2nd last layer######################################################
model.final.ep50 %>% pop_layer()
train.validation.features.20 <- model.final.ep50 %>% predict(x=train.validation)
test.features.20 <- model.final.ep50 %>% predict(x=test)
#dim(test.features.20)


train.valid.data.20 <- cbind(y=train.validation.labels,train.validation.features.20)
test.data.20 <- cbind(y=test.labels,test.features.20)

####LASSO
lasso.20 <- cv.glmnet(train.validation.features.20, y=train.validation.labels, family = "binomial", type.measure = "class", alpha=1)
lasso_model.20 <- glmnet(x = train.validation.features.20, y = train.validation.labels, family = "binomial",  alpha = 1, lambda = lasso.20$lambda.min)

predictions_lasso.20 <- predict(object =  lasso_model.20, newx = as.matrix(test.features.20), type = "class", s = "lambda.min") %>% as.vector() %>% as.integer()

Confusion.Matrix.lasso.20 <- table(test.labels,predictions_lasso.20)

####Random Forest
cl <- makePSOCKcluster(parallel::detectCores()) # detects number of cores automatically
registerDoParallel(cl) # register the cores to use 100% of the CPU


control <- trainControl(method='cv', 
                        number=10)

rf.20<-caret::train(as.factor(y) ~., 
                    data = as.data.frame(train.valid.data.20),
                    method = 'rf', # random forest
                    metric = 'Accuracy',
                    trControl = control)


stopCluster(cl)

predict.rf.20 <-predict(rf.20, newdata = test.data.20, type = "raw")

Confusion.Matrix.rf.20 <- table(test.labels,predict.rf.20)#####




##############pop the 3nd last layer######################################################
model.final.ep50 %>% pop_layer()
train.validation.features.100 <- model.final.ep50 %>% predict(x=train.validation)
test.features.100 <- model.final.ep50 %>% predict(x=test)

train.valid.data.100 <- cbind(y=train.validation.labels,train.validation.features.100)
test.data.100 <- cbind(y=test.labels,test.features.100)

###LASSO###
lasso.100 <- cv.glmnet(train.validation.features.100, y=train.validation.labels, family = "binomial", type.measure = "class", alpha=1)
lasso_model.100 <- glmnet(x = train.validation.features.100, y = train.validation.labels, family = "binomial",  alpha = 1, lambda = lasso.100$lambda.min)

predictions_lasso.100 <- predict(object =  lasso_model.100, newx = as.matrix(test.features.100), type = "class", s = "lambda.min") %>% as.vector() %>% as.integer()

Confusion.Matrix.lasso.100 <- table(test.labels,predictions_lasso.100)


###RF######

cl <- makePSOCKcluster(parallel::detectCores()) # detects number of cores automatically
registerDoParallel(cl) # register the cores to use 100% of the CPU


control <- trainControl(method='cv', 
                        number=10)

rf.100<-caret::train(as.factor(y) ~., 
                    data = as.data.frame(train.valid.data.100),
                    method = 'rf', # random forest
                    metric = 'Accuracy',
                    trControl = control)


stopCluster(cl)

predict.rf.100 <-predict(rf.100, newdata = test.data.100, type = "raw")

Confusion.Matrix.rf.100 <- table(test.labels,predict.rf.100)#####



##############pop the 4nd last layer######################################################
model.final.ep50 %>% pop_layer()
train.validation.features.500 <- model.final.ep50 %>% predict(x=train.validation)
test.features.500 <- model.final.ep50 %>% predict(x=test)

train.valid.data.500 <- cbind(y=train.validation.labels,train.validation.features.500)
test.data.500 <- cbind(y=test.labels,test.features.500)

####LASSO
lasso.500 <- cv.glmnet(train.validation.features.500, y=train.validation.labels, family = "binomial", type.measure = "class", alpha=1)
lasso_model.500 <- glmnet(x = train.validation.features.500, y = train.validation.labels, family = "binomial",  alpha = 1, lambda = lasso.500$lambda.min)

predictions_lasso.500 <- predict(object =  lasso_model.500, newx = as.matrix(test.features.500), type = "class", s = "lambda.min") %>% as.vector() %>% as.integer()

Confusion.Matrix.lasso.500 <- table(test.labels,predictions_lasso.500)

####Random Forest
cl <- makePSOCKcluster(parallel::detectCores()) # detects number of cores automatically
registerDoParallel(cl) # register the cores to use 100% of the CPU


control <- trainControl(method='cv', 
                        number=10)

rf.500<-caret::train(as.factor(y) ~., 
                    data = as.data.frame(train.valid.data.500),
                    method = 'rf', # random forest
                    metric = 'Accuracy',
                    trControl = control)


stopCluster(cl)

predict.rf.500 <-predict(rf.500, newdata = test.data.500, type = "raw")

Confusion.Matrix.rf.500 <- table(test.labels,predict.rf.500)#####

### all confusion matrices

lasso <- cbind(Confusion.Matrix.lasso.10, Confusion.Matrix.lasso.20, Confusion.Matrix.lasso.100, Confusion.Matrix.lasso.500)
rf <- cbind(Confusion.Matrix.rf.10, Confusion.Matrix.rf.20, Confusion.Matrix.rf.100, Confusion.Matrix.rf.500)
combined <- rbind(lasso,rf)

Confusion.Matrix.BestfitNN  %>% kable(format = "latex", caption = "Confusion matrix for Neural Network.
        The rows represent actual values, the columns represent the predicted values.", label = "confusionmatrices", booktabs=T) 


combined %>%  
  kable(format = "latex", caption = "Confusion matrices.
        The rows represent actual values, the columns represent the predicted values.", label = "confusionmatrices", booktabs=T) %>%
  kable_styling(latex_options = "striped") %>%
  add_header_above(header = c("","10"=2,"20"=2, "100"=2,"500"=2) ) %>%
  add_header_above(header = c("", "Number of features in last layer"=8) ) %>% 
  pack_rows("LASSO", 1, 2, latex_gap_space = "0.5em") %>%
  pack_rows("Random Forest", 3, 4, latex_gap_space = "0.5em")