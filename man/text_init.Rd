% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/03-dynamic.R
\name{text_init}
\alias{text_init}
\title{Install required Python modules
in a new conda environment
and initialize the environment,
necessary for all \code{text_*} functions
designed for contextualized word embeddings.}
\usage{
text_init()
}
\description{
Install required Python modules
in a new conda environment
and initialize the environment,
necessary for all \code{text_*} functions
designed for contextualized word embeddings.
}
\details{
Users may first need to manually install
\href{https://www.anaconda.com/}{Anaconda} or
\href{https://docs.conda.io/en/main/miniconda.html}{Miniconda}.

The R package \code{text} (\url{https://www.r-text.org/}) enables users access to
\href{https://huggingface.co/models}{HuggingFace Transformers models} in R,
through the R package \code{reticulate} as an interface to Python
and the Python modules \code{torch} and \code{transformers}.

For advanced usage, see
\itemize{
  \item{\code{\link[text:textrpp_install]{text::textrpp_install()}}}
  \item{\code{\link[text:textrpp_install]{text::textrpp_install_virtualenv()}}}
  \item{\code{\link[text:textrpp_uninstall]{text::textrpp_uninstall()}}}
  \item{\code{\link[text:textrpp_initialize]{text::textrpp_initialize()}}}
}
}
\examples{
\dontrun{
text_init()

# You may need to specify the version of Python:
# RStudio -> Tools -> Global/Project Options
# -> Python -> Select -> Conda Environments
# -> Choose ".../textrpp_condaenv/python.exe"
}

}
\seealso{
\code{\link{text_model_download}}

\code{\link{text_model_remove}}

\code{\link{text_to_vec}}

\code{\link{text_unmask}}
}
