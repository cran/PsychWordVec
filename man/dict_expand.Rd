% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/01-basic.R
\name{dict_expand}
\alias{dict_expand}
\title{Expand a dictionary from the most similar words.}
\usage{
dict_expand(data, words, threshold = 0.5, iteration = 5, verbose = TRUE)
}
\arguments{
\item{data}{A \code{\link[PsychWordVec:as_wordvec]{wordvec}} (data.table) or
\code{\link[PsychWordVec:as_embed]{embed}} (matrix),
see \code{\link{data_wordvec_load}}.}

\item{words}{A single word or a list of words,
used to calculate the
\link[PsychWordVec:sum_wordvec]{sum vector}.}

\item{threshold}{Threshold of cosine similarity,
used to find all words with similarities higher than this value.
Defaults to \code{0.5}. A low threshold may lead to failure of convergence.}

\item{iteration}{Number of maximum iterations. Defaults to \code{5}.}

\item{verbose}{Print information to the console? Defaults to \code{TRUE}.}
}
\value{
An expanded list (character vector) of words.
}
\description{
Expand a dictionary from the most similar words.
}
\section{Download}{

Download pre-trained word vectors data (\code{.RData}):
\url{https://psychbruce.github.io/WordVector_RData.pdf}
}

\examples{
\donttest{dict = dict_expand(demodata, "king")
dict

dict = dict_expand(demodata, cc("king, queen"))
dict

most_similar(demodata, dict)

dict.cn = dict_expand(demodata, "China")
dict.cn  # too inclusive if setting threshold = 0.5

dict.cn = dict_expand(demodata,
                      cc("China, Chinese"),
                      threshold=0.6)
dict.cn  # adequate to represent "China"
}
}
\seealso{
\code{\link{sum_wordvec}}

\code{\link{most_similar}}

\code{\link{dict_reliability}}
}
