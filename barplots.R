

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


setwd("/home/todd/Desktop/")
# import tumor cases
df_importer <- function(filename, sheet_num){
  df <- read_excel(filename, sheet = sheet_num, col_names = TRUE)
  # df <- as.data.frame(df[complete.cases(df),])
  return(df)
}

# import tumor cases
df_importer <- function(filename, sheet_num){
  df <- read_excel(filename, sheet = sheet_num, col_names = TRUE)
  df <- as.data.frame(df[complete.cases(df),])
  return(df)
}
umich_tumors <- df_importer("/home/todd/Desktop/patient_softmax.xlsx", sheet = 1)
columbia_tumors <- df_importer("/home/todd/Desktop/patient_softmax.xlsx", sheet = 4)

# expand the dataframe to fit with normals, this is purely to allow for dataframe building 
add_column <- function(df, label){
  label_vect <- rep(label, nrow(df))
  return(cbind(df, label_vect))
}

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
umich_normals <- df_importer("/home/todd/Desktop/patient_softmax.xlsx", sheet = 2)
columbia_normals <- df_importer("/home/todd/Desktop/patient_softmax.xlsx", sheet = 5)
add_label_column <- function(df, label){
  label_vect <- rep(label, nrow(df))
  return(cbind(df, label_vect))
}

# build vectors of labels
umich_normals <- add_label_column(umich_normals, "umich_tumors")
umich_tumors <- add_label_column(umich_tumors, "umich_tumors")
columbia_normals <- add_label_column(columbia_normals, "columbia_normals")
columbia_tumors <- add_label_column(columbia_tumors, "columbia_normals")

# concatenate all the cases in the trial
full_df <- rbind(umich_normals, umich_tumors, columbia_normals, columbia_tumors)

ggplot(full_df, aes(condensed_truelabel)) +
  geom_bar(aes(fill = label_vect))

#reorder the table and reset the factor to that ordering
full_df %>%
  group_by(condensed_truelabel) %>%                              # calculate the counts
  summarize(counts = n()) %>%
  arrange(-counts) %>%
  mutate(condensed_truelabel = factor(condensed_truelabel, condensed_truelabel)) %>%   # reset factor
  ggplot(aes(x=condensed_truelabel, y=counts)) +                 # plot
  geom_bar(stat="identity")                         # plot histogram


###############
# training patch numbers 
df_importer("CNN_dataset.xlsx", 1) %>%
  arrange(-num) %>%
  mutate(class = factor(class, class)) %>% 
  ggplot(aes(x = class, y = num)) +
  geom_bar(stat = "identity")

# training case numbers
df_importer("CNN_dataset.xlsx", 2)%>%
  arrange(-num) %>%
  mutate(class = factor(class, class)) %>% 
  ggplot(aes(x = class, y = num)) +
  geom_bar(stat = "identity")

# validation case numbers
df_importer("CNN_dataset.xlsx", 4)%>%
  arrange(-num) %>%
  mutate(class = factor(class, class)) %>% 
  ggplot(aes(x = class, y = num)) +
  geom_bar(stat = "identity")

# training patch numbers 
df_importer("CNN_dataset.xlsx", 5) %>%
  arrange(-num) %>%
  mutate(class = factor(class, class)) %>% 
  ggplot(aes(x = class, y = num)) +
  geom_bar(stat = "identity")






