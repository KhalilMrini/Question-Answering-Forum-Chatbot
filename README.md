# Building a Question-Answering Chatbot using Forum Data in the Semantic Space

[Khalil Mrini](https://KhalilMrini.github.io/), [Marc Laperrouza](https://people.epfl.ch/marc.laperrouza/bio?lang=en&cvlang=en), [Pierre Dillenbourg](https://people.epfl.ch/cgi-bin/people?id=155704&op=bio&lang=en&cvlang=en)

Best Presentation Award at SwissText 2018 (Winterthur, Switzerland): [Paper](https://infoscience.epfl.ch/record/256467?&ln=en), [Presentation](https://www.youtube.com/watch?v=ht03dVRmYmQ).

## Abstract

We build a conversational agent which knowledge base is an online forum for parents of autistic children. We collect about 35,000 threads totalling some 600,000 replies, and label 1% of them for usefulness using Amazon Mechanical Turk. We train a Random Forest Classifier using sent2vec features to label the remaining thread replies. Then, we use word2vec to match user queries conceptually with a thread, and then a reply with a predefined context window.

## How to use

The jupyter notebook file details the process to build the chatbot. The `README` file in the Chatbot folder details how to launch the Django-based interface to chat with the Chatbot. **However**, pickle files are needed for it to run, that are too large to put on GitHub. Please contact the author to get them. You will also need the word2vec model `GoogleNews-vectors-negative300.bin`.
