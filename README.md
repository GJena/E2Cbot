# E2Cbot (CIS-700-7_Chatbot-Project)
This is the final project for CIS 700-7. It continues on our previous assignment which can be found at https://github.com/vmansi/HW4.
## Checkpoint 1 : 3/14

### Introduction
We are trying to infuse personality to our chat bot. For this, we are using the dialogues of Star Trek to enable our bot to speak in similar mannerisms. We intend to use different models and analyze the results.

### Goals
- To build a persona-based model.
- To have day to day conversation in StarTrek style .

### Previous work
In Assignment 4, we implemented:
- Chatbot using seq2seq model. Trained on 99,687 post response pairs from Startrek movie and TV series scripts. Incorporated context. Augmented data since initial dataset was very small.
- Chatbot using seq2seq model. Trained on Cornell Movie dataset. 199,455 post response pairs. Created a dataset of 120 very commonly used dialogs in Star Trek series. Used Star Trek dataset to find a dialog that has the maximum likelihood (unigrams) of following the dialog the Seq2Seq model generates.

### Milestones
1. Train seq2seq model on Cornell movie dataset. Fine tune using Star Trek dialogs.
2. Implement A Persona-Based Neural Conversation Model by Jiwei Li, Michel Galley, Chris Brockett, Georgios P. Spithourakis, Jianfeng Gao, Bill Dolan.
3. Train a seq2seq model using StarTrek data. Train another seq2seq model using Cornell Movie data. Combine both outputs as done in MemoryNN (O: (output feature map) – produces a new output (in the feature representation space), given the op1 and op2.) , and generate response R from O. r = argmaxw∈W sR([x, op1 , op2 ], w) 
4. Incorporate Star Trek style using style shifting. Inspired from the neural storyteller, it uses skip thought vectors to transfer style from one corpus to another. Reference: https://github.com/ryankiros/neural-storyteller.




