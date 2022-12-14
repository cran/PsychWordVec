% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/03-dynamic.R
\name{text_to_vec}
\alias{text_to_vec}
\title{Extract contextualized word embeddings from transformers (pre-trained language models).}
\usage{
text_to_vec(
  text,
  model,
  layers = "all",
  layer.to.token = "concatenate",
  token.to.word = TRUE,
  token.to.text = TRUE,
  encoding = "UTF-8",
  ...
)
}
\arguments{
\item{text}{Can be:
\itemize{
  \item{a character string or vector of text (usually sentences)}
  \item{a data frame with at least one character variable
  (for text from all character variables in a given data frame)}
  \item{a file path on disk containing text}
}}

\item{model}{Model name at \href{https://huggingface.co/models}{HuggingFace}.
See \code{\link{text_model_download}}.
If the model has not been downloaded, it would automatically download the model.}

\item{layers}{Layers to be extracted from the \code{model},
which are then aggregated in the function
\code{\link[text:textEmbedLayerAggregation]{text::textEmbedLayerAggregation()}}.
Defaults to \code{"all"} which extracts all layers.
You may extract only the layers you need (e.g., \code{11:12}).
Note that layer 0 is the \emph{decontextualized} input layer
(i.e., not comprising hidden states).}

\item{layer.to.token}{Method to aggregate hidden layers to each token.
Defaults to \code{"concatenate"},
which links together each word embedding layer to one long row.
Options include \code{"mean"}, \code{"min"}, \code{"max"}, and \code{"concatenate"}.}

\item{token.to.word}{Aggregate subword token embeddings (if whole word is out of vocabulary)
to whole word embeddings. Defaults to \code{TRUE}, which sums up subword token embeddings.}

\item{token.to.text}{Aggregate token embeddings to each text.
Defaults to \code{TRUE}, which averages all token embeddings.
If \code{FALSE}, the text embedding will be the token embedding of \code{[CLS]}
(the special token that is used to represent the beginning of a text sequence).}

\item{encoding}{Text encoding (only used if \code{text} is a file).
Defaults to \code{"UTF-8"}.}

\item{...}{Other parameters passed to
\code{\link[text:textEmbed]{text::textEmbed()}}.}
}
\value{
A \code{list} of:
\describe{
  \item{\code{token.embed}}{
    Token (roughly word) embeddings}
  \item{\code{text.embed}}{
    Text embeddings, aggregated from token embeddings}
}
}
\description{
Extract hidden layers from a language model and aggregate them to
get token (roughly word) embeddings and text embeddings
(all reshaped to \code{\link[PsychWordVec:as_embed]{embed}} matrix).
It is a wrapper function of \code{\link[text:textEmbed]{text::textEmbed()}}.
}
\examples{
\dontrun{
# text_init()  # initialize the environment

text = c("Download models from HuggingFace",
         "Chinese are East Asian",
         "Beijing is the capital of China")
embed = text_to_vec(text, model="bert-base-cased", layers=c(0, 12))
embed

embed1 = embed$token.embed[[1]]
embed2 = embed$token.embed[[2]]
embed3 = embed$token.embed[[3]]

View(embed1)
View(embed2)
View(embed3)
View(embed$text.embed)

plot_similarity(embed1, value.color="grey")
plot_similarity(embed2, value.color="grey")
plot_similarity(embed3, value.color="grey")
plot_similarity(rbind(embed1, embed2, embed3))
}

}
\seealso{
\code{\link{text_init}}

\code{\link{text_model_download}}

\code{\link{text_model_remove}}

\code{\link{text_unmask}}
}
