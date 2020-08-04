
files_auto<-function(path){
  filenames <- list.files(path, pattern="*.csv", full.names=TRUE)
  if(length(filenames)==1){
    single_file<-read.csv(filenames,check.names=FALSE)
    #single_file<-data.frame(single_file)
    return(single_file)
  }else{
    ldf <- lapply(filenames, read.csv,check.names=FALSE)
    res <- lapply(ldf, rbind)
    return(res)
  }
  
}
