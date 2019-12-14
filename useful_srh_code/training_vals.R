

library(readxl)
library(ggplot2)
library(tidyr)

setwd("/home/todd/Desktop/spreadsheets/")
cnn_df <- read_excel("/home/todd/Desktop/spreadsheets/cnn_training_final.xlsx")

plotting_function <- function(df){
  df %>%
    gather(key, val, -Epoch) %>%
    ggplot(aes(Epoch, val, col=key)) +
    geom_point() +
    geom_line() + 
    ylim(c(0,1))
}

# plot cross entropy loss 
loss_df <- data_frame(cnn_df$Epoch, cnn_df$train_loss, cnn_df$val_loss)
colnames(loss_df) <- c("Epoch", "loss", "val_loss")
plotting_function(loss_df)

# plot classification accuracy
acc_df <- data_frame(cnn_df$Epoch, cnn_df$train_acc, cnn_df$val_acc)
colnames(acc_df) <- c("Epoch", "acc", "val_acc")
plotting_function(acc_df)
