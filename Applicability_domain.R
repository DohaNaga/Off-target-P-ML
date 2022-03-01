#loading necessary libraries
library(dplyr)
library(rcdk)
library(rcdklibs)
library(ggplot2)
library(ggpubr)
library(caret)
library(xlsx)
library(readxl)
library(stats)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

####the part of parsing smiles is adopted from the script fingerprints_preparation.R


#####function to remove duplicated smiles per target
#not removing by id in case user is using a non unique identifier and in case duplicated ids encode different structures

dataset_curated = data.frame()
curate <- function(d_f) {
  for (i in 1: length(unique(d_f %>% pull(OFF_TARGET)))) {
    d_f_1 <- unique (d_f %>% filter(OFF_TARGET == unique(d_f %>% pull(OFF_TARGET) )[i]))
    d_f_2 <- d_f_1[ !duplicated(d_f_1$SMILES),]
    dataset_curated = rbind(dataset_curated,d_f_2)  
    
  }
  return(dataset_curated)
}  

####function to calculate the ranges of molecular descriptors
range_props <- function(descs_df,desc_list){
  ranges <- data.frame()
  for (i in 1:length(desc_list)){
    desc_values <- descs_df[,desc_list[i]]
    min_desc <- min(desc_values)
    max_desc <- max(desc_values)
    if(substr(desc_list[i],1,1)  == "n") {
      mean_desc <- round(mean(desc_values))
    }else{
      mean_desc <- mean(desc_values)  
    }
    ranges_2 <- data.frame(desc_list[i],min_desc,max_desc,mean_desc)
    ranges <- rbind(ranges,ranges_2)
    
  }  
  return(ranges)
}  






#reading unique smiles from the main datasests and converting to a dataframe
compounds_smiles <- (read_excel(paste0("Datasets/", args[1]) ) ) %>% data.frame


#removing duplicated compounds per smile for each target smiles (using curate function)
compounds_smiles_unique <- curate (compounds_smiles)


#parsing total unique smiles to the "mol" format (since smiles might be repeated between targets) for efficency
# if smiles are not parsed, they stay in the list

compounds_mols <- sapply(compounds_smiles_unique[!duplicated(compounds_smiles_unique$SMILES),] %>% pull(SMILES) ,parse.smiles)

#lipinksifailures
# number HB donors
# number HB acceptors
# number rotatable bonds
# Mwt
# LogP
# Molecular refractivity

#choose above descs
dnames <- get.desc.names(type = "all") [c(9,13,14,27,28,50,49)]

#calculate descs
descs <- eval.desc(compounds_mols, dnames, verbose=TRUE) 
desc_list <- colnames(descs)

ranges_fulldataset <- range_props(descs_df = descs_1,desc_list = desc_list)

write.csv(ranges_fulldataset,"ranges_fulldataset.csv")

