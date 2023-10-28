---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for Women's E-Commerce Clothing Reviews

## Dataset Description

- **Homepage:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- **Repository:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)


### Dataset Summary

This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its nine supportive features offer a great environment to parse out the text through its multiple dimensions. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.

### Supported Tasks and Leaderboards

The Women’s Clothing E-Commerce dataset is designed to support:
- **Sentiment Analysis:** Researchers and data analysts can use this dataset to perform sentiment analysis to understand customer sentiment towards clothing and if they recommend it.
- **Consumer Insights:** Businesses can gain valuable insights into customer preferences and areas for improvement based on the reviews and ratings.
- **Natural Language Processing (NLP):** NLP practitioners can utilize textual reviews for various NLP tasks, such as text classification, summarization, and topic modelling.

### Languages

The Women’s Clothing E-Commerce dataset is primarily focused on the English language. All data instances within this dataset are in English.


## Dataset Structure

### Data Instances

Here are a few sample data instances from the Women’s Clothing E-Commerce dataset to illustrate its content and structure:

| ID              | Clothing ID             | Age                    | Title | Review Text | Rating | Recommended IND | Positive Feedback Count| Division Name | Department Name | Class Name |
| :---------------- | :------:              | :----:                  | :----: | :----: | :----------: | :----------: |:----------: |:----------: |:----------: | ----------: |
|0|767|33|  |Absolutely wonderful - silky and sexy and comfortable|4|1|0|Initmates|Intimate|Intimate|
|1|1080|34|  |Love this dress! it's sooo pretty. i happened to find it in a store, and i'm glad i did bc i never...|5|1|4|General|Dresses|Dresses|
|2|1077|60|Some major design flaws |I had such high hopes for this dress and really wanted it to work for me. i initially ordered the pe...|3|0|0|General|Dresses|Dresses|

### Data Fields

The Women’s Clothing E-Commerce dataset includes the following key data fields, each serving a specific purpose in organizing and describing the data; it includes 23486 rows and 10 feature variables:

1. **Clothing ID:** Integer Categorical variable that refers to the specific piece being reviewed.
2. **Age:** Positive Integer variable that refers to the reviewers age.
3. **Title:** String variable that refers to the title of the review.
4. **Review Text:** String variable for the review body, its content itself.
5. **Rating:** Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
6. **Recommended IND:** Binary variable stating if the customer recommends the product; 1 is recommended, 0 is not recommended.
7. **Positive Feedback Count:** Positive Integer documenting the number of other customers who found this review positive.

8. **Division Name:** Categorical name of the product high level division.
9. **Department Name:** Categorical name of the product department name.
10. **Class Name:** Categorical name of the product class name.


### Data Splits

Data was split for training and testing. All data was derived from the used dataset and not manually generated.
The split was done randomly with 80% of the data used for training, 20% for testing.

## Dataset Creation

### Curation Rationale

The dataset was created for Natural Language Processing and Sentiment Analysis. Providing ht review text and Rating, Recommended IND attributes the model can learn which words have a positive connotation and which have a noegative one.

### Source Data

#### Initial Data Collection and Normalization

Data was collected from a women's clothes store website review section and anonymized to keep the consumer's and store's privacy. All references to the store's name were replaced with the word retailer.

#### Data Transformations and Final Dataset

The project's objective was Sentiment Analysis, involving the analysis of textual data and the identification of variables contributing to accurate word classification.

In the process of determining which dataset variables affected the positivity of reviews, an examination was conducted on the impact of each variable on the "Recommended IND" and "rating" attributes as these are primary indicators of sentiment in the reviews. The analysis concluded that, aside from these two attributes, no other variables had a significant influence on the classification outcome. This conclusion was drawn from the observation that positive and negative values were evenly distributed across all categories of each analyzed attribute. Consequently, the decision was made to remove them as they did not provide additional value to the dataset.

As part of efforts to refine the dataset and establish a more balanced model, a new attribute, "Top Product", was introduced. This attribute was created to address the need for data balance. It categorizes products as "Top Product" (Top Product=1) when they have a rating of 5 and as non-top products (Top Product=0) when their rating falls below 5.

This data processing step resulted in the curation of a dataset with a more balanced and pertinent set of attributes, contributing to the success of the Sentiment Analysis project.

The final dataset comprises three columns. The first two columns represent text data, one with stemming applied and the other without stemming, based on user-defined preferences. The third column contains the corresponding product ratings, indicating whether the product is classified as the best product or not.

| Review Text                                  | Top Product | Stemmed Review Text          |
|---------------------------------------------|------------|-----------------------------|
| absolutely wonderful silky sexy comfortable | 0          | absolut wonder silki sexi comfort |


#### Who are the source language producers?

Data was not computer generated, instead it was produced by costumers as it consists of real clothes' reviews from a real unknown retailer.
Data was collected by @nicapotato and uploaded to kaggle.

### Personal and Sensitive Information

The dataset has erased identity categories to avoid conflicts with personal and sensitive information.

## Considerations for Using the Data

### Known Limitations

The dataset is not balanced which creates the need to balance it manually that causes some distorsion on the dataset as values have to be modified to reach a balance.

The dataset is also written in a unique language which doesn't contribute to enrichening NLP in multiple languages.

## Additional Information

### Licensing Information

[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

### Contributions

Thanks to @mrc03, @nicapotato and @BurhanYKiyakoglu for adding this dataset.
