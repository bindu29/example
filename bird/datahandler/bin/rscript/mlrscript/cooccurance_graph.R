

if (!require(tm)) {
  install.packages("tm")
}
if (!require(wordcloud)) {
  install.packages("wordcloud")
}
if (!require(igraph)) {
  install.packages("igraph")
}
if (!require(ggraph)) {
  install.packages("ggraph")
}
if (!require(textstem)) {
  install.packages("textstem")
} 


library(tm)
library(tidyverse)
library(tidytext)
library(wordcloud)
library(igraph)
library(ggraph)
library(widyr)

text_clean = function(x,
                      # x=text_corpus
                      remove_numbers = TRUE,
                      # whether to drop numbers? Default is TRUE
                      remove_stopwords = TRUE)
  # whether to drop stopwords? Default is TRUE
  
{
  library(tm)
  library(textstem) 
  
  x  =  gsub("<.*?>", " ", x)               # regex for removing HTML tags
  x  =  iconv(x, "latin1", "ASCII", sub = "") # Keep only ASCII characters
  x  =  gsub("[^[:alnum:]]", " ", x)        # keep only alpha numeric
  x  =  tolower(x)                          # convert to lower case characters
  
  if (remove_numbers) {
    x  =  removeNumbers(x)
  }    # removing numbers
  
  x  =  stripWhitespace(x)                  # removing white space
  x  =  gsub("^\\s+|\\s+$", "", x)          # remove leading and trailing white space. Note regex usage
  
  # evlauate condn
  if (remove_stopwords) {
    # read std stopwords list from my git
    stpw1 = readLines(
      'https://raw.githubusercontent.com/sudhir-voleti/basic-text-analysis-shinyapp/master/data/stopwords.txt'
    )
    
    # tm package stop word list; tokenizer package has the same name function, hence 'tm::'
    stpw2 = tm::stopwords('english')
    comn  = unique(c(stpw1, stpw2))         # Union of the two lists
    stopwords = unique(gsub("'", " ", comn))  # final stop word list after removing punctuation
    
    # removing stopwords created above
    x  =  removeWords(x, stopwords)
  }  # if condn ends
  
  x  =  stripWhitespace(x)                  # removing white space
  # x  =  stemDocument(x)                   # can stem doc if needed. For Later.
  x  = lemmatize_strings(x) 
  return(x)
}  # func ends

## test-driving on ibm data
#ibm = readLines('https://raw.githubusercontent.com/sudhir-voleti/sample-data-sets/master/International%20Business%20Machines%20(IBM)%20Q3%202016%20Results%20-%20Earnings%20Call%20Transcript.txt')  #IBM Q3 2016 analyst call transcript
#system.time({ ibm.clean =  text.clean(ibm, remove_numbers=FALSE) })  # 0.26 secs

# +++

dtm_build <- function(raw_corpus, tfidf = FALSE)
{
  # func opens
  
  require(tidytext)
  require(tibble)
  require(tidyverse)
  
  # converting raw corpus to tibble to tidy DF
  textdf = data_frame(text = raw_corpus)
  
  tidy_df = textdf %>%
    dplyr::mutate(doc = row_number()) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words) %>%
    group_by(doc) %>%
    dplyr::count(word, sort = TRUE)
  #tidy_df
  
  # evaluating IDF wala DTM
  if (tfidf == "TRUE") {
    textdf1 = tidy_df %>%
      group_by(doc) %>%
      count(word, sort = TRUE) %>% ungroup() %>%
      bind_tf_idf(word, doc, nn) %>%   # 'nn' is default colm name
      dplyr::rename(value = tf_idf)
  } else {
    textdf1 = tidy_df %>% dplyr::rename(value = n)
  }
  
  textdf1
  
  dtm = textdf1 %>% cast_sparse(doc, word, value)
  dtm[1:9, 1:9]
  
  # order rows and colms putting max mass on the top-left corner of the DTM
  colsum = apply(dtm, 2, sum)
  col.order = order(colsum, decreasing = TRUE)
  row.order = order(rownames(dtm) %>% as.numeric())
  
  dtm1 = dtm[row.order, col.order]
  dtm1[1:8, 1:8]
  
  return(dtm1)
}   # func ends

# testing func 2 on ibm data
#system.time({ dtm_ibm_tf = dtm_build(ibm) })    # 0.02 secs

# +++

build_wordcloud <- function(dtm,
                            max.words1 = 150,
                            # max no. of words to accommodate
                            min.freq = 5,
                            # min.freq of words to consider
                            plot.title = "wordcloud") {
  # write within double quotes
  
  require(wordcloud)
  if (ncol(dtm) > 20000) {
    # if dtm is overly large, break into chunks and solve
    
    tst = round(ncol(dtm) / 100)  # divide DTM's cols into 100 manageble parts
    a = rep(tst, 99)
    b = cumsum(a)
    rm(a)
    b = c(0, b, ncol(dtm))
    
    ss.col = c(NULL)
    for (i in 1:(length(b) - 1)) {
      tempdtm = dtm[, (b[i] + 1):(b[i + 1])]
      s = colSums(as.matrix(tempdtm))
      ss.col = c(ss.col, s)
      print(i)
    } # i loop ends
    
    tsum = ss.col
    
  } else {
    tsum = apply(dtm, 2, sum)
  }
  
  tsum = tsum[order(tsum, decreasing = T)]       # terms in decreasing order of freq
  head(tsum)
  tail(tsum)
  
  # windows()  # Opens a new plot window when active
  wordcloud(
    names(tsum),
    tsum,
    # words, their freqs
    scale = c(3.5, 0.5),
    # range of word sizes
    min.freq,
    # min.freq of words to consider
    max.words = max.words1,
    # max #words
    colors = brewer.pal(8, "Dark2")
  )    # Plot results in a word cloud
  title(sub = plot.title)     # title for the wordcloud display
  listed <- list(names(tsum), tsum)
  return(listed)
} # func ends


distill.cog = function(dtm,
                       # input dtm
                       title = "COG",
                       # title for the graph
                       central.nodes = 4,
                       # no. of central nodes
                       max.connexns = 5) {
  # max no. of connections
  
  # first convert dtm to an adjacency matrix
  dtm1 = as.matrix(dtm)   # need it as a regular matrix for matrix ops like %*% to apply
  adj.mat = t(dtm1) %*% dtm1    # making a square symmatric term-term matrix
  diag(adj.mat) = 0     # no self-references. So diag is 0.
  a0 = order(apply(adj.mat, 2, sum), decreasing = T)   # order cols by descending colSum
  mat1 = as.matrix(adj.mat[a0[1:50], a0[1:50]])
  
  # now invoke network plotting lib igraph
  library(igraph)
  
  a = colSums(mat1) # collect colsums into a vector obj a
  b = order(-a)     # nice syntax for ordering vector in decr order
  
  mat2 = mat1[b, b]     # order both rows and columns along vector b
  diag(mat2) =  0
  
  ## +++ go row by row and find top k adjacencies +++ ##
  
  wc = NULL
  
  for (i1 in 1:central.nodes) {
    thresh1 = mat2[i1, ][order(-mat2[i1,])[max.connexns]]
    mat2[i1, mat2[i1, ] < thresh1] = 0   # neat. didn't need 2 use () in the subset here.
    mat2[i1, mat2[i1, ] > 0] = 1
    word = names(mat2[i1, mat2[i1, ] > 0])
    mat2[(i1 + 1):nrow(mat2), match(word, colnames(mat2))] = 0
    wc = c(wc, word)
  } # i1 loop ends
  
  
  mat3 = mat2[match(wc, colnames(mat2)), match(wc, colnames(mat2))]
  ord = colnames(mat2)[which(!is.na(match(colnames(mat2), colnames(mat3))))]  # removed any NAs from the list
  mat4 = mat3[match(ord, colnames(mat3)), match(ord, colnames(mat3))]
  
  # building and plotting a network object
  graph <-
    graph.adjacency(mat4, mode = "undirected", weighted = T)    # Create Network object
  graph = simplify(graph)
  V(graph)$color[1:central.nodes] = "green"
  V(graph)$color[(central.nodes + 1):length(V(graph))] = "pink"
  
  graph = delete.vertices(graph, V(graph)[degree(graph) == 0]) # delete singletons?
  
  plot(graph,
       layout = layout.kamada.kawai,
       main = title)
  return(graph)
} # distill.cog func ends

# testing COG on ibm data
#system.time({ distill.cog(dtm_ibm_tf, "COG for IBM TF") })    # 0.27 secs

# +++

build_cog_ggraph <- function(corpus,
                             # text colmn only
                             max_edges = 100,
                             drop.stop_words = TRUE,
                             new.stopwords = NULL) {
  # invoke libraries
  library(tidyverse)
  library(tidytext)
  library(widyr)
  library(ggraph)
  
  # build df from corpus
  corpus_df = data.frame(
    docID = seq(1:length(corpus)),
    text = corpus,
    stringsAsFactors = FALSE
  )
  
  # eval stopwords condn
  if (drop.stop_words == TRUE) {
    stop.words = unique(c(stop_words$word, new.stopwords)) %>%
      as_tibble() %>% dplyr::rename(word = value)
  } else {
    stop.words = stop_words[2, ]
  }
  
  # build word-pairs
  tokens <- corpus_df %>%
    
    # tokenize, drop stop_words etc
    unnest_tokens(word, text) %>% anti_join(stop.words)
  
  # pairwise_count() counts #token-pairs co-occuring in docs
  word_pairs = tokens %>% pairwise_count(word, docID, sort = TRUE, upper = FALSE)# %>% # head()
  
  word_counts = tokens %>% dplyr::count(word, sort = T) %>% dplyr::rename(wordfr = n)
  
  word_pairs = word_pairs %>% left_join(word_counts, by = c("item1" = "word"))
  
  row_thresh = min(nrow(word_pairs), max_edges)
  
  # now plot
  set.seed(1234)
  # windows()
  plot_d <- word_pairs %>%
    filter(n >= 3) %>%
    top_n(row_thresh) %>%   igraph::graph_from_data_frame()
  if (length(names(V(plot_d))) != 0) {
    dfwordcloud = data_frame(vertices = names(V(plot_d))) %>% left_join(word_counts, by = c("vertices" = "word"))
    
    plot_obj = plot_d %>%   # graph object built!
      
      ggraph(layout = "fr") +
      geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "orange")  +
      # geom_node_point(size = 5) +
      geom_node_point(size = log(dfwordcloud$wordfr)) +
      geom_node_text(
        aes(label = name),
        repel = TRUE,
        point.padding = unit(0.2, "lines"),
        size = 1 + log(dfwordcloud$wordfr)
      ) +
      theme_void()
  } else{
    dfwordcloud = "No sufficient data"
    plot_obj = "No sufficient data"
  }
  
  listed <- list(word_pairs, dfwordcloud, plot_obj)
  return(listed)    # must return func output
  
}  # func ends


co_occurrence <- function(text) {
  text <- text_clean(text)
  dtm_ibm_tf <- dtm_build(text)
  pol=polarity(text)
  pos_words  = data.frame(table(unlist(pol$all[,4])))# Positive words info
  names(pos_words)<-c("name","value")
  neg_words  = data.frame(table(unlist(pol$all[,5])))
  names(neg_words)<-c("name","value")
  word_cloud <- build_wordcloud(dtm_ibm_tf)
  words <- data.frame(word_cloud[1])
  names(words) <- "name"
  word_freq <- data.frame(word_cloud[2])
  word_freq$text <- rownames(word_freq)
  names(word_freq)[1] <- "weight"
  co_occur <- build_cog_ggraph(text)
  dfwordcloud <- data.frame(co_occur[1])
  vertices_frequency <- data.frame(co_occur[2])
  listed <- list(words, word_freq, dfwordcloud, vertices_frequency,pos_words,neg_words)
  return(listed)
}
