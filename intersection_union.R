

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
library(onehot)
library(MASS)

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

setwd("/home/todd/Desktop/IOU_spreadsheets/")
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
umich_tumors <- df_importer("/home/todd/Desktop/IOU_spreadsheets/patient_softmax.xlsx", sheet = 1)
columbia_tumors <- df_importer("/home/todd/Desktop/IOU_spreadsheets/patient_softmax.xlsx", sheet = 4)

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
umich_normals <- df_importer("/home/todd/Desktop/IOU_spreadsheets/patient_softmax.xlsx", sheet = 2)
columbia_normals <- df_importer("/home/todd/Desktop/IOU_spreadsheets/patient_softmax.xlsx", sheet = 5)
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


ggplot(full_df, aes(x = condensed_truelabel, y = iou_label)) +
  geom_point(stat = "identity")

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
  df = data.frame(dataframe$inv_case, dataframe$condensed_predlabel, dataframe$condensed_truelabel, dataframe$frozen, dataframe$label_vect, correct_status, inference_class, probs, dataframe$iou_label, dataframe$iou_tumor, dataframe$iou_nontumor, dataframe$iou_nondiag)
  colnames(df) <- c("inv_case", "condensed_predlabel", "condensed_truelabel", "frozen", "label_vect", "correct_status", "inference_class", "probs", "iou", "iou_tumor", "iou_normal", "iou_nondiag")
  return(df)
}
true_probs_df = true_prob_values(full_df)

posn.j = position_jitter(0.05)
ggplot(true_probs_df, aes(x = condensed_truelabel, y = iou, col = correct_status)) +
  geom_jitter(position = posn.j) +
  scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1)) + 
  xlab("") +
  ylab("True class IOU") +
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.3, size = 2) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.1, col = "black", alpha = 0.3)

`%not_in%` <- purrr::negate(`%in%`)
true_probs_df %>% 
  filter(condensed_truelabel %not_in% normal_names) %>%
  # filter(condensed_truelabel %in% normal_names) %>%
  ggplot(aes(x = condensed_truelabel, y = iou_nondiag, col = correct_status)) +
    geom_jitter(position = posn.j) +
    scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1)) + 
    xlab("") +
    ylab("True class IOU") +
    stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.3, size = 2) +
    stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.1, col = "black", alpha = 0.3)

sum_columns <- function(x, y){
  if (is.na(x)){return(y)}
  else if(is.na(y)){return(x)}
  else {
    return((x + y)/2)
  }
}
normal.tumor.vect = map2_dbl(true_probs_df$iou_tumor, true_probs_df$iou_normal, sum_columns)
test <- kde2d(true_probs_df$probs, true_probs_df$iou)
contour(test)


# plotting intersection of union against prediction probabilities 
ggplot(true_probs_df, aes(x = iou, y = probs, col = inference_class, shape = correct_status)) + 
  geom_point() + 
  geom_smooth(method=lm)
  
ggplot(true_probs_df, aes(x = iou, y = probs)) + 
  geom_point() + 
  geom_smooth(method=lm)


cor(true_probs_df$iou, true_probs_df$probs, use = "complete.obs")


