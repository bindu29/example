stop=c("Accept","add","admire","admit","advise","afford","agree","alert","bring","extremely","awful","sauce","not","br","put","soon","make","br","perhaps","go","recommend"
       ,"old","send","many","start","ever","try","href","type","ounce","later","allow","amuse","analyse","analyze","announce","annoy","answer","apologise","appear","applaud"
       ,"appreciate","approve","argue","arrange","arrest","arrive","ask","attach","attack","attempt","attend","avoid","back","bake"
       ,"balance","ban","bang","bare","bat","bathe","battle","beam","beg","behave","belong","bleach","bless","blind","blink","blot","blush","boast","boil","bolt","bomb","book","bore"
       ,"borrow","bounce","bow","box","brake","branch","breathe","bruise","brush","bubble","bump","burn","bury","buzz","calculate","call","camp","care","boil","bolt","bomb","book","bore","borrow","bounce","bow","box"
       ,"brake","carry","carve","cause","challenge","change","charge","chase","cheat","check","cheer","chew","choke","chop","claim","clap"
       ,"clean","clear","clip","close","coach","coil","collect","colour","comb","command","communicate","compare","compete","complain","complete","concentrate",'concern',"confess","confuse","connect","consider","consist"
       ,"contain","continue","copy","correct"
       ,"choke"
       ,"chop"
       ,"claim"
       ,"clap",
       "attractive",
       "bald",
       "beautiful",
       "chubby",
       "clean",
       "dazzling",
       "drab",
       "elegant",
       "fancy",
       "fit",
       "flabby",
       "glamorous",
       "gorgeous",
       "handsome",
       "long",
       "magnificent",
       "muscular",
       "plain",
       "plump",
       "quaint",
       "scruffy",
       "shapely",
       "short",
       "skinny",
       "stocky",
       "ugly",
       "unkempt",
       "unsightly"
       , "aggressive"
       ,"agreeable"
       ,"ambitious"
       ,"brave"
       ,"calm"
       ,"delightful"
       ,"eager"
       ,"faithful"
       ,"gentle"
       ,"happy"
       ,"jolly"
       ,"kind"
       ,"lively"
       ,"nice"
       ,"obedient"
       ,"polite"
       ,"proud"
       ,"silly"
       ,"thankful"
       ,"victorious"
       ,"witty"
       ,"wonderful"
       ,"zealous"
       ,"clean"
       ,"clear"
       ,"clip"
       ,"close"
       ,"coach"
       ,"coil"
       ,"collect"
       ,"colour"
       ,"comb"
       ,"command"
       ,"communicate"
       ,"cough"
       ,"count"
       ,"cover"
       ,"crack"
       ,"crash"
       ,"crawl"
       ,"cross"
       ,"crush"
       ,"cry"
       ,"cure"
       ,"curl"
       ,"curve"
       ,"cycle"
       ,"dam"
       ,"damage"
       ,"dance"
       ,"dare"
       ,"decay"
       ,"deceive"
       ,"decide"
       ,"decorate"
       ,"delay"
       ,"delight"
       ,"deliver"
       ,"depend"
       ,"describe"
       ,"desert"
       ,"deserve"
       ,"destroy"
       ,"detect"
       ,"develop"
       ,"disapprove"
       ,"disarm"
       ,"discover"
       ,"dislike"
       ,"divide"
       ,"double"
       ,"doubt"
       ,"drag"
       ,"drain"
       ,"dream"
       ,"earn"
       ,"educate"
       ,"embarrass"
       ,"employ"
       ,"empty"
       ,"encourage"
       ,"end"
       ,"enjoy"
       ,"enter"
       ,"entertain"
       ,"escape"
       ,"examine"
       ,"excite"
       ,"excuse"
       ,"exercise"
       ,"exist"
       ,"expand"
       ,"expect"
       ,"dress"
       ,"drip"
       ,"drop"
       ,"drown"
       ,"drum"
       ,"dry"
       ,"dust"
       ,"face"
       ,"fade"
       ,"fail"
       ,"fancy"
       ,"fasten"
       ,"fax"
       ,"fetch"
       ,"file"
       ,"fill"
       ,"film"
       ,"fire"
       ,"fit"
       ,"fix"
       ,"flap"
       ,"flash"
       ,"float"
       ,"flood"
       ,"flow"
       ,"flower"
       ,"fold"
       ,"follow"
       ,"fool","broad","chubby","crooked"
       ,"curved"
       ,"deep"
       ,"flat"
       ,"high"
       ,"hollow"
       ,"low"
       ,"narrow"
       ,"refined"
       ,"round"
       ,"shallow"
       ,"skinny"
       ,"square"
       ,"steep"
       ,"straight"
       ,"wide",
       "abundant",
       "billions"
       ,"enough"
       ,"few"
       ,"full"
       ,"hundreds"
       ,"incalculable"
       ,"limited"
       ,"little"
       ,"many"
       ,"most"
       ,"millions"
       ,"numerous"
       ,"scarce"
       ,"some"
       ,"sparse"
       ,"substantial"
       ,"thousands","br","not","very","beat",	"beat","beaten",
       "become",	"became",	"become",
       "begin",	"began"	,"begun",
       "bend",	"bent"	,"bent","bet",
       "bite",	"bit",	"bitten",
       "blow",	"blew",	"blown",
       "break",	"broke"	,"broken",
       "bring",	"brought"	,"brought",
       "build"	,"built"	,
       "burn",	"burned","burnt",
       "buy"	,"bought"	,"bought",
       "catch",	"caught"	,
       "choose",	"chose",	"chosen",
       "come"	,"came"	,"come",
       "cost",	
       "cut","dug",
       "dive",	"dove",	"dived",
       "do"	,"did"	,"done",
       "draw",	"drew",	"drawn",
       "dream"	,"dreamed","dreamt"	,"dreamed","dreamt",
       "drive",	"drove",	"driven",
       "drink"	,"drank",	"drunk",
       "eat"	,"ate"	,"eaten",
       "fall",	"fell",	"fallen",
       "feel"	,"felt","do","not")

# clean_text<-function(data,Text){
#   # create a document term matrix to clean
#   corpus <- Corpus(VectorSource(data[,"Text"]))
#   corpus <- tm_map(corpus, content_transformer(tolower)) 
#   corpus <- tm_map(corpus, removePunctuation) 
#   corpus <- tm_map(corpus,removeWords,stopwords("english")) 
#   corpus <- tm_map(corpus,removeWords,stop)
#   corpus <- tm_map(corpus, removeNumbers) 
#   corpus <- tm_map(corpus,stripWhitespace)
#   reviewsDTM <- DocumentTermMatrix(corpus)
#   #listed<-list(corpus,reviewsDTM)
#   return(reviewsDTM)
# }

clean_text<-function(data,Text){
  # create a document term matrix to clean
  corpus <- Corpus(VectorSource(data[,"Text"]))
  corpus <- tm_map(corpus, content_transformer(tolower)) 
  corpus <- tm_map(corpus, removePunctuation) 
  corpus <- tm_map(corpus,removeWords,stopwords("english")) 
  corpus <- tm_map(corpus,removeWords,stop)
  corpus <- tm_map(corpus, removeNumbers) 
  corpus <- tm_map(corpus,stripWhitespace)
  #reviewsDTM <- DocumentTermMatrix(corpus)
  #listed<-list(corpus,reviewsDTM)
  return(corpus)
}

#clean_data<-clean_text(reviews,"Text")

burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

top_terms_by_topic_LDA <- function(data,corpus,Text, # return a plot? TRUE by defult
                                   k = 4,nstart,best,burnin,seed) # number of topics (4 by default)
{    
  #corpus<-clean_text(data,"Text")
  clean_data <- DocumentTermMatrix(corpus)
  #clean_data<-reviewsDTM
  # create a corpus (type of object expected by tm) and document term matrix
  # preform LDA & get the words/topic in a tidy text format
  ldaOut <-LDA(clean_data,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
  ldaOut.topics <- as.matrix(topics(ldaOut))
  data$topic_no<-ldaOut.topics
  ldaOut.terms <- data.frame(terms(ldaOut,5))
  for(i in unique(data$topics)){
    data$topics[data$topic_no==i]<-list(levels(ldaOut.terms[,i]))
  }
  # if the user asks for a plot (TRUE by default)
  # if the user does not request a plot
  # return a list of sorted terms instead
  listed<-list(data,ldaOut.terms,ldaOut)
  return(listed)
}

#topp<-top_terms_by_topic_LDA(reviews,"Text",k = 4,nstart,best,burnin)
#corpus=clean_text1(reviews,"Text")

sentiment<-function(data,Text,topic,ldaOut.terms,corpus){
  topiccount=length(levels(as.factor(data$topic_no)))
  text_eachdocument=table(data$topic_no)
  #topic_data<-subset(data,topic==topic)
  clean_data<-clean_text(reviews,Text)
  #sentiments<-topic_data[,"Text"] %>% get_sentiments("afinn")
  text<-data[,Text]
  pol = qdap::polarity(text)   # Calc polarity from qdap dictionary
    pos.words  = pol$all[,4]                  # Positive words info
  neg.words  = pol$all[,5] 
  data$poswords<-pol$all[,4]
  data$negwords<-pol$all[,5]
  qdap_outp <- text %>% 
    data_frame() %>%
    data.frame(., pol$all) %>%
    select(text.var, wc, polarity, pos.words, neg.words) 
  pos.words  = pol$all[,4]                  # Positive words info
  neg.words  = pol$all[,5] 
  polarity<-qdap_outp$polarity
  positive_words = unique(setdiff(unlist(pos.words),"-")) 
  data[,"polarity"]<-qdap_outp$polarity
  dtm <- TermDocumentMatrix(corpus)
  m <- as.matrix(dtm)
  v <- sort(rowSums(m),decreasing=TRUE)
  d <- data.frame(word = names(v),freq=v)
  negative_words = unique(setdiff(unlist(neg.words),"-"))  
  ap_topics <- data.frame(tidy(LDA(clean_data,4, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin)), matrix = "beta"))
  df2 <- ap_topics[-grep('^\\d+$', ap_topics$term),]
  listed<-list(data,topiccount,negative_words,ap_topics,polarity,d,df2,positive_words)
  return(listed)
}


#line to run 
#corpus=clean_text(reviews,Text)
#topp<-top_terms_by_topic_LDA(reviews,corpus,"Text",k = 4,nstart,best,burnin)
#djfd=sentiment(data=reviews,Text="Text",topic=1,ldaOut.terms=topp[2],corpus = corpus)



