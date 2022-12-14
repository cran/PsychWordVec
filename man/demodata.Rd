% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/00-utils.R
\name{demodata}
\alias{demodata}
\title{Demo data (pre-trained using word2vec on Google News; 8000 vocab, 300 dims).}
\format{
A \code{data.table} (of new class \code{wordvec}) with two variables \code{word} and \code{vec},
transformed from the raw data (see the URL in Source) into \code{.RData}
using the \code{\link{data_transform}} function.
}
\source{
Google Code - word2vec (\url{https://code.google.com/archive/p/word2vec/})
}
\usage{
data(demodata)
}
\description{
This demo data contains a sample of 8000 English words
with 300-dimension word vectors pre-trained
using the "word2vec" algorithm based on the Google News corpus.
Most of these words are from the Top 8000 frequent wordlist,
whereas a few are selected from less frequent words and appended.
}
\examples{
class(demodata)
demodata

embed = as_embed(demodata, normalize=TRUE)
class(embed)
embed

}
