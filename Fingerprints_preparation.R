#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

#loading necessary libraries
library(readxl)
library(readr)
library(dplyr)
library(rcdk)
library(rcdklibs)
library(fingerprint)
library(xlsx)

#functions

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


#reading unique smiles from the main datasests and converting to a dataframe
compounds_smiles <- (read_excel(paste0("Datasets/", args[1]) ) ) %>% data.frame


#removing duplicated compounds per smile for each target smiles (using curate function)
compounds_smiles_unique <- curate (compounds_smiles)


#parsing total unique smiles to the "mol" format (since smiles might be repeated between targets) for efficency
# if smiles are not parsed, they stay in the list

compounds_mols <- sapply(compounds_smiles_unique[!duplicated(compounds_smiles_unique$SMILES),] %>% pull(SMILES) ,parse.smiles)

#saving compound ids 
names(compounds_mols) <- compounds_smiles_unique[!duplicated(compounds_smiles_unique$SMILES),] %>% pull(COMPOUND_ID)

#getting the names of the compound ids that were not parsed and resulted in nulls

#not using the indices because mols were calculated on total unique smiles
names_null <-   names(compounds_mols[ which (sapply(compounds_mols, is.null))] )


if(length(names_null) != 0){
compounds_smiles_curated <- compounds_smiles_unique[!compounds_smiles_unique$COMPOUND_ID %in% names_null,] 
#removing nulls from compounds_mols
compounds_mols <- compounds_mols[-which(sapply(compounds_mols, is.null))]

 }else{ 
compounds_smiles_curated <- compounds_smiles_unique

}


#rematching ids in compounds_mols with compounds_smiles_curated since there might be duplicated troubled smiles with different ids
compounds_smiles_curated <- compounds_smiles_curated %>% filter(COMPOUND_ID %in% names(compounds_mols))


 
#calculating actives/inactives after removal of errored smiles 
actives_inactives <- compounds_smiles_curated  %>% group_by(OFF_TARGET) %>% dplyr::count(BINARY_VALUE) %>% data.frame()

#writing the final curated dataset and the actives/inactives
write.xlsx(compounds_smiles_curated,"Datasets/dataset1_curated.xlsx")
write.xlsx(actives_inactives,"Datasets/actives_inactives.xlsx")


###generate compounds circular finger prints from the "mol" format
fps_ecfp4 <- lapply(compounds_mols[1:length(compounds_mols)], get.fingerprint,type = 'circular', circular.type = 'ECFP4') 


#add compound ids & convert fps objects into  matrix 
fps_ecfp4_binary <- fp.to.matrix(fps_ecfp4)

rownames(fps_ecfp4_binary) <- compounds_smiles_curated[!duplicated(compounds_smiles_curated$SMILES),]  %>% pull(COMPOUND_ID)

#removing duplicated compound ids (which are screened with more than one target)
fps_ecfp4_binary_unique <- fps_ecfp4_binary[unique(rownames(fps_ecfp4_binary)),]

##writing down fingerprints matrix  

write.table(fps_ecfp4_binary_unique, "Datasets/dataset_2.csv",row.names = TRUE, col.names = TRUE )
##########################################################################################################################################################


