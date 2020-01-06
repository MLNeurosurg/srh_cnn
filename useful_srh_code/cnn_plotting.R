

library(readxl)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
library(corrplot)
library(tibble)
library(caret)
library(e1071)
library(scales)
library(viridis)
library(RColorBrewer)
library(forcats)
library(wesanderson)
library(pROC)
library(onehot)
library(ROCR)


# NIO_numbers_test = {
# "ependymoma": [],
# "glioblastoma": [],
# "greymatter": [],
# "lowgradeglioma": [],
# "lymphoma": [],
# "medulloblastoma": [],
# "meningioma": [],
# "metastasis": [],
# "nondiagnostic": [],
# "pilocyticastrocytoma": [],
# "pituitaryadenoma": [],
# "pseudoprogression": [],
# "schwannoma": [], 
# "whitematter": []
# }

setwd("")
# tumor_names <- c("ependymoma", "glioblastoma", "lowgradeglioma", "lymphoma", "medulloblastoma", "meningioma", "metastasis", "pilocyticastrocytoma", "pituitaryadenoma", "schwannoma")
tumor_names <- c("glioma", "ependymoma", "glioblastoma","greymatter", "lowgradeglioma", "lymphoma", "medulloblastoma", "meningioma", "metastasis","nondiagnostic", "pilocyticastrocytoma", "pituitaryadenoma", "pseudoprogression", "schwannoma", "whitematter")
normal_names <- c("greymatter", "whitematter", "pseudoprogression")
glial_names <- c("glioma", "ependymoma", "lowgradeglioma", "glioblastoma", "pilocyticastrocytoma")
nonglial_names <- c("medulloblastoma", "meningioma", "metastasis", "pituitaryadenoma", "schwannoma")
nonsurgical_names <- c("lymphoma")

# import tumor cases
df_importer <- function(filename, sheet_num){
  df <- read_excel(filename, sheet = sheet_num, col_names = TRUE)
  return(df)
}
umich_tumors <- df_importer("/Users/toddhollon/Desktop/CNN_SRH/patient_softmax.xlsx", sheet = 1)
columbia_tumors <- df_importer("/Users/toddhollon/Desktop/CNN_SRH/patient_softmax.xlsx", sheet = 4)

# expand the dataframe to fit with normals, this is purely to allow for dataframe building 
expand_tumors_df <- function(df_tumors){
  column_names <- c("greymatter", "nondiagnostic", "pseudoprogression", "whitematter")
  columns <- c(4, 10, 13, 15)
  for (i in seq_along(columns)){
    df_tumors <- add_column(df_tumors, rep(0, nrow(df_tumors)), .before = columns[i])
    colnames(df_tumors)[columns[i]] <- column_names[i]
  }
  return(df_tumors)
}
columbia_tumors <- expand_tumors_df(columbia_tumors)
umich_tumors <- expand_tumors_df(umich_tumors)

#### import normal cases 
umich_normals <- df_importer("/Users/toddhollon/Desktop/CNN_SRH/patient_softmax.xlsx", sheet = 2)
columbia_normals <- df_importer("/Users/toddhollon/Desktop/CNN_SRH/patient_softmax.xlsx", sheet = 5)
add_label_column <- function(df, label){
  label_vect <- rep(label, nrow(df))
  return(cbind(df, label_vect))
}

# build vectors of labels
umich_normals <- add_label_column(umich_normals, "umich_normals")
umich_tumors <- add_label_column(umich_tumors, "umich_tumors")
columbia_normals <- add_label_column(columbia_normals, "columbia_normals")
columbia_tumors <- add_label_column(columbia_tumors, "columbia_tumors")

# concatenate all the cases in the trial
full_df <- rbind(umich_normals, umich_tumors, columbia_normals, columbia_tumors)
# full_df <- filter(full_df, condensed_truelabel != "schwannoma")

# find the maximum value for correct case
argmax_function <- function(dataframe) {
  maxes <- c()
  for (i in 1:nrow(dataframe)){
    if (dataframe$condensed_predlabel[i] == dataframe$condensed_truelabel[i]){
      if (dataframe$condensed_truelabel[i] == "glioma"){ #sum the softmax values for glioma classes
        maxes = c(maxes, sum(dataframe$ependymoma[i],
                             dataframe$glioblastoma[i],
                             dataframe$lowgradeglioma[i],
                             dataframe$pilocyticastrocytoma[i]))
      } else{maxes = c(maxes, max(dataframe[i, 2:15]))}
    } 
    else{maxes = c(maxes, -max((dataframe[i, 2:15])))}
  }
  
  correct_status <- c()
  for (i in 1:nrow(dataframe)){
    if (dataframe$condensed_predlabel[i] == dataframe$condensed_truelabel[i]){correct_status = c(correct_status, "correct")}
    else{correct_status = c(correct_status, "wrong")}
  } 
  df = data.frame(dataframe$inv_case, dataframe$condensed_predlabel, dataframe$condensed_truelabel, dataframe$label_vect, maxes)
  colnames(df) <- c("inv_case", "condensed_predlabel", "condensed_truelabel", "label_vect", "maxes")
  return(df)
}
condense_classes <- function(full_df){
  condense_classes <- c()
  for (i in 1:nrow(full_df)){
    if (full_df$true_label[i] %in% glial_names){condense_classes <- c(condense_classes, "glial")}
    else if (full_df$true_label[i] %in% nonglial_names){condense_classes <- c(condense_classes, "nonglial")}
    else if (full_df$true_label[i] %in% nonsurgical_names){condense_classes <- c(condense_classes, "nonsurgical")}
    else if (full_df$true_label[i] %in% normal_names){condense_classes <- c(condense_classes, "normal")}
  }
  return(condense_classes)
}
  
full_df %>%
  argmax_function() %>%
  ggplot(aes(x=reorder(inv_case, maxes), y=maxes, fill=condense_classes(full_df))) +
  geom_bar(stat = "identity") + 
  # coord_flip() +
  # scale_fill_manual(values=wes_palette(n=4, name="GrandBudapest1"))
  scale_fill_brewer(palette = "RdBu")
  # facet_grid(.~condensed_truelabel)

overall_accuracy <- function(df){
  correct <- 0
  for (i in seq_along(df$condensed_predlabel)){
    if (df$condensed_predlabel[i] == df$condensed_truelabel[i]){correct = correct + 1}
  }
  print(correct/dim(df[1]))
  print((dim(df[1])-correct))
}
overall_accuracy(full_df)

###### probability distribution plots
true_prob_values <- function(dataframe) {
  probs <- c()
  for (i in 1:nrow(dataframe)){
    label <- dataframe$condensed_truelabel[i]
    if (label == "glioma"){ #sum the softmax values for glioma classes
      probs <- c(probs, sum(dataframe$ependymoma[i],
                            dataframe$glioblastoma[i],
                            dataframe$lowgradeglioma[i],
                            dataframe$pilocyticastrocytoma[i]))
    } 
    else{probs <- c(probs, dataframe[[label]][i])}
  } 
  
  inference_class <- c() # add inference node_vector
  for (i in 1:nrow(dataframe)){
    if (dataframe$condensed_truelabel[i] == "glioma" | dataframe$condensed_truelabel[i] == "ependymoma" | dataframe$condensed_truelabel[i] == "glioblastoma" | dataframe$condensed_truelabel[i] == "lowgradeglioma" | dataframe$condensed_truelabel[i] == "pilocyticastrocytoma"){
      inference_class <- c(inference_class, "glial")
    } else if(dataframe$condensed_truelabel[i] == "lymphoma") inference_class <- c(inference_class, "nonsurgical")
    else if(dataframe$condensed_truelabel[i] == "greymatter" | dataframe$condensed_truelabel[i] == "whitematter" | dataframe$condensed_truelabel[i] == "pseudoprogression") inference_class <- c(inference_class, "normal") 
    else inference_class <- c(inference_class, "nonglial")
  } 
  
  correct_status <- c() # add correct or incorrect vector
  for (i in 1:nrow(dataframe)){
    if (dataframe$condensed_predlabel[i] == dataframe$condensed_truelabel[i]){correct_status = c(correct_status, "correct")}
    else{correct_status = c(correct_status, "wrong")}
  }  
  df = data.frame(dataframe$inv_case, dataframe$condensed_predlabel, dataframe$condensed_truelabel, dataframe$frozen, dataframe$label_vect, correct_status, inference_class, probs, dataframe$iou_label)
  colnames(df) <- c("inv_case", "condensed_predlabel", "condensed_truelabel", "frozen", "label_vect", "correct_status", "inference_class", "probs", "iou")
  return(df)
}

umich_df <- filter(full_df, label_vect == "umich_tumors" | label_vect == "umich_normals")
columbia_df <- filter(full_df, label_vect == "columbia_tumors" | label_vect == "columbia_normals")

umich_df$condensed_predlabel <- factor(umich_df$condensed_predlabel, levels = tumor_names)
umich_df$condensed_truelabel <- factor(umich_df$condensed_truelabel, levels = tumor_names)
columbia_df$condensed_predlabel <- factor(columbia_df$condensed_predlabel, levels = tumor_names)
columbia_df$condensed_truelabel <- factor(columbia_df$condensed_truelabel, levels = tumor_names)

true_prob_df <- true_prob_values(columbia_df)

ggplot(true_prob_df, aes(x=reorder(inv_case, desc(probs)), y = probs, fill = inference_class)) + 
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "RdBu") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

columbia_df %>%
  true_prob_values() %>%
  ggplot(aes(x=reorder(inv_case, desc(probs)), y = probs, fill = correct_status)) + 
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "RdBu")

# umich_df %>%
#   true_prob_values() %>%
#   ggplot(aes(probs, fill = inference_class)) + 
#     geom_histogram(position = "dodge", bins = 100) +
#     scale_fill_brewer(palette = "RdBu") +  
#     scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12)) + 
#     facet_grid(.~inference_class)


###########################Condense glioma classes
condense_glioma_softmaxes <- function(dataframe) {
  maxes <- c()
  for (i in 1:nrow(dataframe)){
    # consolidate the glioma classes
    if (dataframe$condensed_truelabel[i] == "glioma" | dataframe$condensed_truelabel[i] == "ependymoma" | dataframe$condensed_truelabel[i] == "glioblastoma" | dataframe$condensed_truelabel[i] == "lowgradeglioma" | dataframe$condensed_truelabel[i] == "pilocyticastrocytoma"){maxes = c(maxes, sum(dataframe$ependymoma[i],
                                                                                         dataframe$glioblastoma[i],
                                                                                         dataframe$lowgradeglioma[i],
                                                                                         dataframe$pilocyticastrocytoma[i]))} 
    # leave non glioma classes alone
    else {maxes = c(maxes, max(dataframe[i, 2:15]))}
  }  
  return(maxes)
}

# return glioma diagnoses
return_glial_dx <- function(dataframe_column){
  dx <- c()
  for (i in 1:length(dataframe_column)){
    if (dataframe_column[i] == "glioma" | dataframe_column[i] == "ependymoma" | dataframe_column[i] == "glioblastoma" | dataframe_column[i] == "lowgradeglioma" |dataframe_column[i] == "pilocyticastrocytoma"){dx = c(dx, "glioma")}
    else {dx = c(dx, dataframe_column[i])}
  }
  return(dx)
}

# Define the glioma dataframe
glioma_df <- data.frame(full_df$inv_case, return_glial_dx(full_df$condensed_predlabel), return_glial_dx(full_df$condensed_truelabel), condense_glioma_softmaxes(full_df))
colnames(glioma_df) <- c("inv_case","condensed_predlabel", "condensed_truelabel", "maxes") 
find_incorrect_cases <- function(df){
  for (i in 1:nrow(df)){
    if (df$condensed_predlabel[i] != df$condensed_truelabel[i]){df$maxes[i] = -df$maxes[i]}
  }
  return(df)
}
glioma_df = find_incorrect_cases(glioma_df)

condense_classes_glial <- function(full_df){
  condense_classes <- c()
  for (i in 1:nrow(full_df)){
    if (full_df$condensed_truelabel[i] %in% glial_names){condense_classes <- c(condense_classes, "glial")}
    else if (full_df$condensed_truelabel[i] %in% nonglial_names){condense_classes <- c(condense_classes, "nonglial")}
    else if (full_df$condensed_truelabel[i] %in% nonsurgical_names){condense_classes <- c(condense_classes, "nonsurgical")}
    else if (full_df$condensed_truelabel[i] %in% normal_names){condense_classes <- c(condense_classes, "normal")}
  }
  return(condense_classes)
}

ggplot(glioma_df, aes(x=reorder(inv_case, maxes), y=maxes, fill = condense_classes_glial(glioma_df))) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  scale_fill_brewer(palette = "RdBu")
  # facet_grid(.~condensed_truelabel)

foo = data_frame(c("col_normal ", "umich", ), c(51, 153))
names(foo) = c("center", "num")
ggplot(foo, aes(x = "", y = num, fill = center)) +
  geom_bar(width = 1, stat="identity") +
  coord_polar("y")











