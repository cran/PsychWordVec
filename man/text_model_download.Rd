% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/03-dynamic.R
\name{text_model_download}
\alias{text_model_download}
\title{Download pre-trained language models from HuggingFace.}
\usage{
text_model_download(model = NULL)
}
\arguments{
\item{model}{Character string(s) specifying the
pre-trained language model(s) to be downloaded.
For a full list of options, see
\href{https://huggingface.co/models}{HuggingFace}.
Defaults to download nothing and check currently downloaded models.

Example choices:
\itemize{
  \item{\code{"gpt2"} (50257 vocab, 768 dims, 12 layers)}
  \item{\code{"openai-gpt"} (40478 vocab, 768 dims, 12 layers)}
  \item{\code{"bert-base-uncased"} (30522 vocab, 768 dims, 12 layers)}
  \item{\code{"bert-large-uncased"} (30522 vocab, 1024 dims, 24 layers)}
  \item{\code{"bert-base-cased"} (28996 vocab, 768 dims, 12 layers)}
  \item{\code{"bert-large-cased"} (28996 vocab, 1024 dims, 24 layers)}
  \item{\code{"bert-base-chinese"} (21128 vocab, 768 dims, 12 layers)}
  \item{\code{"bert-base-multilingual-cased"} (119547 vocab, 768 dims, 12 layers)}
  \item{\code{"distilbert-base-uncased"} (30522 vocab, 768 dims, 6 layers)}
  \item{\code{"distilbert-base-cased"} (28996 vocab, 768 dims, 6 layers)}
  \item{\code{"distilbert-base-multilingual-cased"} (119547 vocab, 768 dims, 6 layers)}
  \item{\code{"albert-base-v2"} (30000 vocab, 768 dims, 12 layers)}
  \item{\code{"albert-large-v2"} (30000 vocab, 1024 dims, 24 layers)}
  \item{\code{"roberta-base"} (50265 vocab, 768 dims, 12 layers)}
  \item{\code{"roberta-large"} (50265 vocab, 1024 dims, 24 layers)}
  \item{\code{"xlm-roberta-base"} (250002 vocab, 768 dims, 12 layers)}
  \item{\code{"xlm-roberta-large"} (250002 vocab, 1024 dims, 24 layers)}
  \item{\code{"xlnet-base-cased"} (32000 vocab, 768 dims, 12 layers)}
  \item{\code{"xlnet-large-cased"} (32000 vocab, 1024 dims, 24 layers)}
  \item{\code{"microsoft/deberta-v3-base"} (128100 vocab, 768 dims, 12 layers)}
  \item{\code{"microsoft/deberta-v3-large"} (128100 vocab, 1024 dims, 24 layers)}
  \item{\code{...} (see \url{https://huggingface.co/models})}
}}
}
\value{
Invisibly return the names of all downloaded models.
}
\description{
Download pre-trained language models (Transformers Models,
such as GPT, BERT, RoBERTa, DeBERTa, DistilBERT, etc.)
from \href{https://huggingface.co/models}{HuggingFace} to
your local ".cache" folder ("C:/Users/[YourUserName]/.cache/").
The models will never be removed unless you run
\code{\link{text_model_remove}}.
}
\examples{
\dontrun{
# text_init()  # initialize the environment

text_model_download()  # check downloaded models
text_model_download(c(
  "bert-base-uncased",
  "bert-base-cased",
  "bert-base-multilingual-cased"
))
}

}
\seealso{
\code{\link{text_init}}

\code{\link{text_model_remove}}

\code{\link{text_to_vec}}

\code{\link{text_unmask}}
}
