# A Forum-based Chatbot for Parents of Autistic Children

## Abstract

We build a conversational agent which knowledge base is an online forum for parents of autistic children. We col- lect about 35,000 threads totalling some 600,000 replies, and label 1% of them for usefulness using Amazon Mechanical Turk. We train a Random Forest Clas- sifier using sent2vec features to label the remaining thread replies. Then, we use word2vec to match user queries conceptu- ally with a thread, and then a reply with a predefined context window.

## How to use

The jupyter notebook file details the process to build the chatbot. The `README` file in the Chatbot folder details how to launch the Django-based interface to chat with the Chatbot. **However**, pickle files are needed for it to run, that are too large to put on GitHub. Please contact the author to get them.