
#loading necessary libraries
library(readxl)
library(dplyr)
library(rcdk)
library(rcdklibs)

#reading smiles that has been standardized by pipeline pilot and converting to a dataframe
compounds_smiles <- read_excel("/Datasets/dataset_1") %>% data.frame 

#parsing smiles to the "mol" format
compounds_mols <- sapply(compounds_smiles$SMILES,parse.smiles)


###generate compounds circular finger prints from the "mol" format
fps_ecfp4<- lapply(compounds_mols[1:length(compounds_mols)], get.fingerprint,type = 'circular', circular.type = 'ECFP4') 


#add cas numbers  & convert fps objects into  matrix 
fps_ecfp4_binary <- fp.to.matrix(fps_ecfp4)
rownames(fps_ecfp4_binary) <- compounds_smiles$CAS.Number


##writing down fingerprints file 

write.xls(fps_ecpf4_binary, "/Datasets/dataset_2.xls")
##########################################################################################################################################################
