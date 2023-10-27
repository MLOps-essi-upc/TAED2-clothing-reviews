---
authors:
  - Valèria Caro Via
  - Esther Fanyanàs i Ropero
  - Claudia Len Manero
language: en
name: Sentiment Analysis for Clothing Reviews
model_type: Pretrained/Fine-Tuned
dataset:
   name: Women's E-Commerce Clothing Reviews
   url: [Dataset Card](https://github.com/MLOps-essi-upc/TAED2-clothing-reviews/blob/main/datasetcard.md)
---

# Model Card for Women's E-Commerce Clothing Reviews

The model used is a pretrained BERT model on English language using a masked language modeling (MLM) objective. A transfer learning with this model has been designed to analyze product reviews and predict whether people are likely to recommend the product or not. 

## Model Details

### Model Description

This model is a fine-tuned version of BERT (Bidirectional Encoder Representations), which is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. The key characteristics of BERT are as follows:

- Bidirectional - to understand the text  you're looking you'll have to look back (at the previous words) and forward (at the next words)
- Transformers - The Transformer reads entire sequences of tokens at once. In a sense, the model is non-directional, while LSTMs read sequentially (left-to-right or right-to-left). The attention mechanism allows for learning contextual relations between words (e.g. `his` in a sentence refers to Jim).
- (Pre-trained) contextualized word embeddings - encode words based on their meaning/context. Nails has multiple meanings - fingernails and metal nails.

BERT was trained by masking 15% of the tokens with the goal to guess them. An additional objective was to predict the next sentence.

Consequently, the model can make binary recommendations by considering both the sentiment and
content of input reviews.


- **Developed by:** Valèria Caro Via, Esther Fanyanàs i Ropero, Claudia Len Manero
- **Model type:** Transformer
- **Language(s) (NLP):** English
- **Finetuned from model Bert Base Cased:** This model was fine-tuned from the "bert-base-cased" model, which is available at https://huggingface.co/bert-base-uncased. 

### Model Sources

The model finetuned can be downlad in the following link:

- **Repository:** https://dagshub.com/claudialen/TAED2-clothing-reviews/src/main/model/transfer-learning.pt

## Uses

The model is designed to analyze customer reviews and comments in order to understand sentiment and determine whether customers are likely to recommend a product positively or not. 

### Direct Use

This is useful for companies to make data-driven decisions, without having to read all the reviews. An idea of the product recommendations will be available and thus be able to make improvements through a global view of all the reviews.

### Downstream Use

This model is versatile and can be applied to a range of downstream tasks, including but not limited to:

- Sentiment analysis: It can be used to determine sentiment in text data, making it valuable for social media analytics, product reviews, and customer feedback analysis.

- Text classification: The model's ability to understand context makes it suitable for tasks like topic classification, spam detection, and content categorization.

The model can be easily integrated into various natural language processing pipelines and frameworks, making it a valuable tool for a wide range of applications.

### Out-of-Scope Use

The model has been trained from reviews, thus there is a subjective opinion in the text.

## Bias, Risks, and Limitations

### Bias
- The opinion on a product depends on several subjective aspects that are not covered by the model, such as the size of women.

### Risks
It's important to be aware of the following risks and limitations when using this model:
- **Robustness:** The model might be sensitive to input phrasing and might produce different results for slight variations in input text.
- **Data Quality:** The model's performance depends on the quality and representativeness of the training data. If your data is noisy or unrepresentative, the model's predictions may be less reliable.
- **Ethical Use:** Users should employ the model in an ethical and responsible manner. Avoid using it for harmful or malicious purposes, such as generating harmful content, fake news, or spam.


### Limitations 
- The model may be less accurate for text data that is significantly different from the e-commerce reviews it was trained on.
- The model may be less accurate with male review text data, as all instances of the model are female.
- The model has been trained in English, so it does not support the input of text in other languages.
- The model does not interpret emojis.

### Recommendations

The model is recommended for analyzing overall women's trends in customer sentiment and identifying areas for improvement based on customer feedback.

## How to Get Started with the Model


To get started with this model, follow the steps below:

1. **Install the Required Libraries:**
    ```bash
    pip install torchvision
    pip install transformers
    pip install pandas
    pip install datasets

2. **Load the model:**
   ```
   model = torch.load("transfer-learning.pt", map_location = "cpu")
   ``` 
3. **Perform Inference**
   ```
   text = "I love this dress."
   words = text.split()
   data = pd.DataFrame({'Review Text': words})
   hg_data = Dataset.from_pandas(data)
   tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
   dataset = hg_data.map(tokenizer(
        hg_data['Review Text'], max_length=128, truncation=True, padding="max_length"
    ))
   dataset = dataset.remove_columns(["Review Text"])
   dataset.set_format("torch")
   text_dataloader = DataLoader(dataset=dataset_text, shuffle=True, batch_size=4)
   ```

## Training Details

### Training Data

The processed data underwent a split, allocating 80% of the data for the training dataset with a random seed of 2023.

[Dataset Card](https://github.com/MLOps-essi-upc/TAED2-clothing-reviews/blob/main/datasetcard.md)

### Training Procedure 

During the training process, this model underwent the following training configuration:

- **Number of Epochs:** 3
- **Total Training Steps:** 384

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The input data for this model underwent the following preprocessing steps:

1. **Text Tokenization:** The input text is tokenized into subword units using the "bert-base-cased" tokenizer.

2. **Data Preparation:** The input text is split into individual words, resulting in a list of words. This list is then converted into a Pandas DataFrame with a column labeled "Review Text."

4. **Tokenization and Formatting:** The data is tokenized using the "bert-base-cased" tokenizer with a maximum sequence length of 128, truncation enabled, and padding applied to ensure consistent input lengths. The resulting dataset was then formatted for PyTorch.

5. **DataLoader Creation:** A PyTorch DataLoader is created for the tokenized and formatted dataset, with a batch size of 4 and shuffle set to True for training.

These preprocessing steps are crucial to ensure that input data is in the correct format for the model's input requirements. Users who plan to use the model should consider these preprocessing steps when preparing their data.


#### Training Hyperparameters

The hyperparameteres have been evaluated through experiments in MlFlow and the best results
obtained have been with the follows:

- **Learning rate:** 5e-6
- **Epochs:** 10

#### Speeds, Sizes, Times

- **Training Time:** 1 hour
- **Inference Time:** 1 minute
- **Model Size:** 433,4 MB
- **Tokenizer Max Length:** 128 - the maximum sequence length that the tokenizer is configured to handle.

## Evaluation

This section describes the evaluation protocols and provides the results.

### Testing Data, Factors & Metrics

#### Testing Data

For our test dataset, we employed a 15% split of the preprocessed data. This split ensured that a portion of the data was reserved exclusively for evaluating the model's performance.
[Dataset Card](https://github.com/MLOps-essi-upc/TAED2-clothing-reviews/blob/main/datasetcard.md)


#### Factors

The evaluation of this model takes into account the following factors, with a focus on women's clothing reviews:

- **Clothing Categories:** The model's effectiveness is evaluated for various categories of women's clothing, including but not limited to dresses, tops, bottoms, and accessories.

- **Sentiment Analysis:** The model's performance in understanding sentiment and emotions expressed in reviews, which is especially important in the context of women's clothing where preferences and emotions can vary widely.

- **Fashion Trends:** Consideration of the model's ability to capture and adapt to evolving fashion trends and styles that are prominent in women's clothing.

- **Seasonality:** Evaluation accounts for seasonal variations in fashion, assessing how well the model handles reviews related to different seasons and occasions.

#### Metrics

In assessing the model's performance, we have focused on the following primary metric:

- **Accuracy:** The choice of the "Accuracy" metric aligns with our objective of ensuring
comprehensive representation of opinions. We aim to accurately classify reviews, categorizing products
as either recommended or not recommended, encompassing both positive and negative sentiments. Accuracy measures the overall correctness of these classifications.


### Results

Subsequently, we conducted an evaluation of the model’s performance, resulting in the following
performance metrics:

- **Accuracy:** 79.81%


## Model Examination 

In our examination of the model's behavior, we considered the following:

- **Dropout Regularization:** Dropout is a technique used in our model architecture for regularization. It involves randomly deactivating a fraction of neurons during training to prevent overfitting. Our analysis included investigating the impact of dropout on the model's performance, robustness, and generalization capabilities.
- **Attention Mask:** Examination of attention masks generated by the model's self-attention mechanisms, revealing which parts of the input text the model focuses on when making predictions.

## Environmental Impact

Our machine learning experiments were conducted using a private infrastructure, Kaggle, which has a carbon efficiency of 0.432 kgCO$_2$eq/kWh. A cumulative of 1 hours of computation was performed on hardware of type Tesla P100 (TDP of 250W).

The total estimated carbon emissions from our machine learning computations amounted to 0.11 kgCO$_2$eqof which 0 percents were directly offset. 

The estimations were carried out using the [MachineLearning Impact calculator](https://mlco2.github.io/impact#compute), a tool that quantifies the environmental impact of machine learning computations.


- **Hardware Type:** Tesla P100 (Kaggle T100)
- **Hours Used:** 1 hour
- **Compute Region:** Catalonia, Europe
- **Carbon Emitted:** 0.11 kgCO₂eq

This section provides a clear overview of the environmental considerations and actions taken to offset carbon emissions associated with our machine learning experiments.

## Technical Specifications 

### Model Architecture and Objective

Our model is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, which is a transformer-based model pretrained on a large corpus of English data in a self-supervised manner. BERT is renowned for its bidirectional understanding of text, allowing it to consider both previous and subsequent words when processing language.

**Objective:** The primary objective of our model is to accurately classify reviews in the context of women's clothing. It aims to categorize products as recommended or not recommended based on the sentiment and content expressed in reviews, encompassing both positive and negative sentiments.

BERT was pretrained with the goal of predicting masked tokens within text and also predicting the next sentence. This pretraining helps the model understand contextual relations between words, making it highly effective for various natural language processing tasks.


## Model Card Authors

- **Valèria Caro Via**
- **Esther Fanyanàs I Ropero**
- **Claudia Len Manero**

## Model Card Contact

- **Contact Name:** Esther Fanyanàs I Ropero

- **Contact Email:** esther.fanyanas.i@estudiantat.upc.edu
