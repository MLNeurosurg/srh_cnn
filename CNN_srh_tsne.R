

library(ggplot2)
library(dplyr)
library(tidyr)
library(readxl)
library(purrr)
library(stringr)

tsne_df <- read_excel("tsne_df.xlsx")
tumors <- unique(tsne_df$labels) # tumor classes
glial <- c("ependymoma", "glioblastoma", "lowgradeglioma", "pilocyticastrocytoma")
nonglial <- c("lymphoma", "medulloblastoma", "meningioma", "metastasis", "pituitaryadenoma", "schwannoma")

string_splitter <- function(filename){
    test <- strsplit(filename, "/")[[1]][2]
    bar <- strsplit(test, "_")[[1]][1]
    return(bar)
}

glial_nonglial <- function(x){
  if (x %in% nonglial){return("nonglial")}
  else {return("glial")}
}

tsne_df["patients"] <- map_chr(tsne_df$filenames, string_splitter)
tsne_df["glial_nonglial"] <- map_chr(tsne_df$labels, glial_nonglial)
tsne_df["patient_labels"] <- paste(tsne_df$patients, tsne_df$labels, sep="_")

# 
patient_tsne_df <- tsne_df %>%
  group_by(patient_labels) %>%
  summarise(means_xs = mean(xs), means_ys = mean(ys)) 

patient_tsne_df$labels = map_chr(patient_tsne_df$patient_labels, string_splitter2)
patient_tsne_df["glial_nonglial"] <- map_chr(patient_tsne_df$labels, glial_nonglial)


# main tsne
ggplot() +
  geom_point(data = tsne_df, aes(xs, ys, col = labels), size = 0.4, alpha = 0.3) + 
  geom_point(data = patient_tsne_df, aes(means_xs, means_ys, col = labels), size = 1) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggplot() +
  geom_point(data = tsne_df, aes(xs, ys, col = glial_nonglial), size = 0.4, alpha = 0.3) + 
  geom_point(data = patient_tsne_df, aes(means_xs, means_ys, col = glial_nonglial), size = 1) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))



ggplot(tsne_df, aes(xs, ys, col = glial_nonglial)) +
  geom_point()




tsne_whiteplot = function(df){
  ggplot(df, aes(xs, ys, col = labels)) + 
    geom_jitter() + 
    theme_bw() + 
    theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
}
tsne_whiteplot(tsne_df)

mean_tumor_positions = function(df){
  tumors = unique(df$labels)
  means = data_frame(tumors)
  xs <- c()
  ys <- c()

  for (tumor in tumors){
      fitered_df <- filter(df, labels == tumor) 
      xs <- c(xs, mean(fitered_df$xs))
      ys <- c(ys, mean(fitered_df$ys))
  }
  return(cbind(means, xs, ys))
}
# function to select tumors to plot
plot_tumors = function(df, vector_tumors){
  tumor_df = filter(df, labels == list_tumors)
  return(tumor_df)
}


tsne_whiteplot(plot_tumors(tsne_df, "meningioma"))

ggplot() +
  geom_point(data=tsne_df, aes(xs, ys, color=labels)) +
  geom_point(data=foo, aes(xs, ys))

ggplot(tsne_df, aes(xs, ys, col = labels)) +
  geom_point() + 
  geom_point(aes(foo$x, foo$ys))




